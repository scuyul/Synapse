package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	stateFile    = "logs/reefscape_visualizer_state.json"
	metricsFile  = "logs/desktop_training_metrics.jsonl"
	defaultModel = "models/reefscape_ppo"
	defaultBest  = "models/best_reefscape_ppo"
	maxLogLines  = 2000
)

const windowsCreateNoWindow = 0x08000000

type App struct {
	ctx context.Context

	repoRoot string
	python   string
	pyErr    error

	mu                   sync.Mutex
	process              *exec.Cmd
	running              bool
	current              string
	logs                 []string
	lastCode             *int
	missingTrainingCache []string
	missingTrainingAt    time.Time
	metricsCache         []MetricRow
	metricsAt            time.Time
	artifactsCache       []ArtifactRow
	artifactsAt          time.Time
}

type StateResponse struct {
	RepoRoot        string         `json:"repoRoot"`
	Python          string         `json:"python"`
	PythonAvailable bool           `json:"pythonAvailable"`
	PythonError     string         `json:"pythonError,omitempty"`
	MissingTraining []string       `json:"missingTraining,omitempty"`
	Running         bool           `json:"running"`
	Current         string         `json:"current"`
	Logs            []string       `json:"logs"`
	LastCode        *int           `json:"lastCode,omitempty"`
	Metrics         []MetricRow    `json:"metrics"`
	Artifacts       []ArtifactRow  `json:"artifacts"`
	Snapshot        map[string]any `json:"snapshot,omitempty"`
}

type LiveStateResponse struct {
	Running  bool           `json:"running"`
	Current  string         `json:"current"`
	LastCode *int           `json:"lastCode,omitempty"`
	Snapshot map[string]any `json:"snapshot,omitempty"`
}

type TrainConfig struct {
	Timesteps       string `json:"timesteps"`
	ModelOut        string `json:"modelOut"`
	ResumeFrom      string `json:"resumeFrom"`
	Device          string `json:"device"`
	RobotProfile    string `json:"robotProfile"`
	Preview         string `json:"preview"`
	NEnvs           string `json:"nEnvs"`
	NSteps          string `json:"nSteps"`
	BatchSize       string `json:"batchSize"`
	LearningRate    string `json:"learningRate"`
	CheckpointEvery string `json:"checkpointEvery"`
	Pretrain        bool   `json:"pretrain"`
	AdvantageScope  bool   `json:"advantageScope"`
}

type MetricRow struct {
	Step   int     `json:"step"`
	Reward float64 `json:"reward"`
	Loss   float64 `json:"loss"`
	FPS    float64 `json:"fps"`
	Event  string  `json:"event"`
}

type ArtifactRow struct {
	Path      string `json:"path"`
	Size      string `json:"size"`
	SizeBytes int64  `json:"sizeBytes"`
	Modified  string `json:"modified"`
}

func NewApp() (*App, error) {
	repoRoot, err := findRepoRoot()
	if err != nil {
		return nil, err
	}
	python, pyErr := findPython(repoRoot)
	app := &App{
		repoRoot: repoRoot,
		python:   python,
		pyErr:    pyErr,
	}
	app.appendLog("Reefscape RL React desktop app started.")
	app.appendLog("Repo: " + repoRoot)
	if pyErr != nil {
		app.appendLog("Python: " + pyErr.Error())
	} else {
		app.appendLog("Python: " + python)
	}
	return app, nil
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
}

func (a *App) shutdown(ctx context.Context) {
	_ = a.Stop()
}

func (a *App) GetState() StateResponse {
	a.mu.Lock()
	logs := append([]string(nil), a.logs...)
	running := a.running
	current := a.current
	lastCode := a.lastCode
	a.mu.Unlock()

	response := StateResponse{
		RepoRoot:        a.repoRoot,
		Python:          a.python,
		PythonAvailable: a.pyErr == nil,
		MissingTraining: a.missingTrainingPackagesCached(),
		Running:         running,
		Current:         current,
		Logs:            logs,
		LastCode:        lastCode,
		Metrics:         a.metricsCached(),
		Artifacts:       a.artifactsCached(),
		Snapshot:        loadSnapshotMap(filepath.Join(a.repoRoot, stateFile)),
	}
	if a.pyErr != nil {
		response.PythonError = a.pyErr.Error()
	}
	return response
}

func (a *App) GetLiveState() LiveStateResponse {
	a.mu.Lock()
	running := a.running
	current := a.current
	lastCode := a.lastCode
	a.mu.Unlock()

	return LiveStateResponse{
		Running:  running,
		Current:  current,
		LastCode: lastCode,
		Snapshot: loadSnapshotMap(filepath.Join(a.repoRoot, stateFile)),
	}
}

func (a *App) RunAction(id string) error {
	switch id {
	case "setup_deps":
		return a.runManagedSequence(
			"Setup + Install Training Dependencies",
			[][]string{setupVenvArgs(a.repoRoot), installTrainingDepsArgs(a.repoRoot)},
		)
	case "doctor":
		return a.runPython("Run Doctor", "-m", "reefscape_rl.doctor")
	case "checks":
		return a.runPython("Run Full Checks", "-c", "import manage; manage.run_full_checks()")
	case "build":
		return a.runManaged("Build Portable App", []string{a.pythonOrFallback(), "build.py"})
	case "installer":
		return a.runManaged("Build Installer", []string{a.pythonOrFallback(), "build.py", "--installer"})
	case "folder":
		return a.OpenFolder()
	case "advantagescope":
		return a.OpenAdvantageScope()
	default:
		return fmt.Errorf("unknown action: %s", id)
	}
}

func (a *App) StartTraining(config TrainConfig) error {
	if err := a.ensureTrainingReady(); err != nil {
		a.appendLog("Train blocked: " + err.Error())
		return err
	}
	args := a.trainingArgs(config, false)
	return a.runManaged("Train PPO", append([]string{a.pythonOrFallback(), "scripts/train_ppo.py"}, args...))
}

func (a *App) StartSmokeRun() error {
	if err := a.ensureTrainingReady(); err != nil {
		a.appendLog("Smoke test blocked: " + err.Error())
		return err
	}
	args := []string{
		"scripts/train_ppo.py", "--timesteps", "256", "--model-out", "models/studio_smoke",
		"--device", "auto", "--robot-profile", "sim", "--n-envs", "2", "--n-steps", "64",
		"--batch-size", "128", "--checkpoint-every-steps", "0", "--skip-heuristic-pretrain",
		"--visualization-backend", "custom-ui", "--custom-ui-state", stateFile,
		"--metrics-out", "logs/studio/smoke_metrics.jsonl", "--metrics-every-steps", "64",
	}
	return a.runManaged("Smoke Test Run", append([]string{a.pythonOrFallback()}, args...))
}

func (a *App) RunCLI(command string) error {
	command = strings.TrimSpace(command)
	if command == "" {
		return errors.New("empty command")
	}
	if runtime.GOOS == "windows" {
		return a.runManaged("CLI Command", []string{"cmd", "/c", command})
	}
	return a.runManaged("CLI Command", []string{"sh", "-c", command})
}

func (a *App) Stop() error {
	a.mu.Lock()
	cmd := a.process
	a.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	a.appendLog("stop requested")
	if runtime.GOOS == "windows" {
		stopCmd := exec.Command("taskkill", "/PID", strconv.Itoa(cmd.Process.Pid), "/T", "/F")
		hideConsoleWindow(stopCmd)
		if err := stopCmd.Run(); err != nil {
			a.appendLog("taskkill failed: " + err.Error())
			_ = cmd.Process.Kill()
		}
		return nil
	}
	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		a.appendLog("interrupt failed: " + err.Error())
		return cmd.Process.Kill()
	}
	return nil
}

func (a *App) OpenFolder() error {
	if runtime.GOOS == "windows" {
		return exec.Command("explorer", a.repoRoot).Start()
	}
	if runtime.GOOS == "darwin" {
		return exec.Command("open", a.repoRoot).Start()
	}
	return exec.Command("xdg-open", a.repoRoot).Start()
}

func (a *App) OpenAdvantageScope() error {
	candidates := []string{}
	if env := strings.TrimSpace(os.Getenv("ADVANTAGESCOPE_PATH")); env != "" {
		candidates = append(candidates, env)
	}
	if path, err := exec.LookPath("advantagescope"); err == nil {
		candidates = append(candidates, path)
	}
	candidates = append(candidates,
		filepath.Join(os.Getenv("LOCALAPPDATA"), "Programs", "AdvantageScope", "AdvantageScope.exe"),
		`C:\Program Files\AdvantageScope\AdvantageScope.exe`,
		`C:\Program Files (x86)\AdvantageScope\AdvantageScope.exe`,
	)
	for _, candidate := range candidates {
		if fileExists(candidate) {
			a.appendLog("Opened AdvantageScope: " + candidate)
			return exec.Command(candidate).Start()
		}
	}
	a.appendLog("AdvantageScope was not found. Open it manually and connect to 127.0.0.1:5810.")
	return errors.New("AdvantageScope was not found")
}

func (a *App) EvaluateArtifact(path string) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return errors.New("select a model first")
	}
	return a.runPython("Evaluate Model", "scripts/evaluate_model.py", "--model", path)
}

func (a *App) ReplayArtifact(path string) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return errors.New("select a model first")
	}
	if err := a.ensureTrainingReady(); err != nil {
		a.appendLog("Model replay blocked: " + err.Error())
		return err
	}
	return a.runPython(
		"Run Model in Field",
		"scripts/run_trained_model.py",
		"--model", path,
		"--fixed-start",
		"--loop",
		"--speed", "2.0",
		"--visualization-backend", "custom-ui",
		"--custom-ui-state", stateFile,
	)
}

func (a *App) trainingArgs(config TrainConfig, smoke bool) []string {
	preview := fallback(config.Preview, "desktop")
	backend := map[string]string{
		"desktop":        "custom-ui",
		"both":           "both",
		"advantagescope": "advantagescope",
		"none":           "none",
	}[preview]
	if backend == "" {
		backend = "custom-ui"
	}
	args := []string{
		"--timesteps", fallback(config.Timesteps, "100000"),
		"--model-out", fallback(config.ModelOut, defaultModel),
		"--device", fallback(config.Device, "cuda"),
		"--robot-profile", fallback(config.RobotProfile, "2025-robot"),
		"--n-envs", fallback(config.NEnvs, "8"),
		"--n-steps", fallback(config.NSteps, "512"),
		"--batch-size", fallback(config.BatchSize, "1024"),
		"--learning-rate", fallback(config.LearningRate, "0.0003"),
		"--checkpoint-dir", "models/checkpoints",
		"--checkpoint-every-steps", fallback(config.CheckpointEvery, "10000"),
		"--keep-checkpoints", "2",
		"--visualization-backend", backend,
		"--advantage-port", "5810",
		"--custom-ui-port", "8775",
		"--custom-ui-state", stateFile,
		"--viz-every-steps", "512",
		"--viz-preview-steps", "25",
		"--metrics-out", metricsFile,
		"--metrics-every-steps", "512",
		"--eval-checkpoints",
		"--eval-episodes", "5",
		"--best-model-out", defaultBest,
	}
	if resume := strings.TrimSpace(config.ResumeFrom); resume != "" {
		args = append(args, "--resume-from", resume)
	}
	if !config.Pretrain {
		args = append(args, "--skip-heuristic-pretrain")
	}
	if backend == "none" || (backend == "advantagescope" && !config.AdvantageScope) {
		args = append(args, "--no-advantagescope")
	}
	return args
}

func (a *App) runPython(label string, args ...string) error {
	return a.runManaged(label, append([]string{a.pythonOrFallback()}, args...))
}

func (a *App) runManaged(label string, args []string) error {
	if len(args) == 0 {
		return errors.New("no command configured")
	}
	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		return errors.New("another command is already running")
	}
	a.running = true
	a.current = label
	a.lastCode = nil
	a.mu.Unlock()

	a.appendLog("")
	a.appendLog("==> " + label)
	a.appendLog("$ " + strings.Join(args, " "))

	go a.runProcess(args)
	return nil
}

func (a *App) runManagedSequence(label string, commands [][]string) error {
	if len(commands) == 0 {
		return errors.New("no command configured")
	}
	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		return errors.New("another command is already running")
	}
	a.running = true
	a.current = label
	a.lastCode = nil
	a.missingTrainingAt = time.Time{}
	a.mu.Unlock()

	a.appendLog("")
	a.appendLog("==> " + label)
	go func() {
		for index, command := range commands {
			a.appendLog(fmt.Sprintf("-- step %d/%d", index+1, len(commands)))
			code := a.runProcessSync(command)
			if code != 0 {
				a.finishProcess(code, fmt.Sprintf("sequence stopped at step %d with exit code %d.", index+1, code))
				return
			}
		}
		a.finishProcess(0, "setup and dependencies finished successfully.")
	}()
	return nil
}

func (a *App) runProcess(args []string) {
	code := a.runProcessSync(args)
	a.finishProcess(code, fmt.Sprintf("finished with exit code %d.", code))
}

func (a *App) runProcessSync(args []string) int {
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = a.repoRoot
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	hideConsoleWindow(cmd)
	a.appendLog("$ " + strings.Join(args, " "))
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		a.appendLog("stdout error: " + err.Error())
		return -1
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		a.appendLog("stderr error: " + err.Error())
		return -1
	}
	if err := cmd.Start(); err != nil {
		a.appendLog("start error: " + err.Error())
		return -1
	}
	a.mu.Lock()
	a.process = cmd
	a.mu.Unlock()

	var wg sync.WaitGroup
	wg.Add(2)
	go streamLines(&wg, stdout, a.appendLog)
	go streamLines(&wg, stderr, a.appendLog)
	wg.Wait()

	code := 0
	if err := cmd.Wait(); err != nil {
		code = 1
		if exitErr, ok := err.(*exec.ExitError); ok {
			code = exitErr.ExitCode()
		}
	}
	return code
}

func hideConsoleWindow(cmd *exec.Cmd) {
	if runtime.GOOS != "windows" {
		return
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP | windowsCreateNoWindow,
		HideWindow:    true,
	}
}

func (a *App) finishProcess(code int, message string) {
	python, pyErr := findPython(a.repoRoot)
	a.mu.Lock()
	a.running = false
	a.current = ""
	a.process = nil
	a.lastCode = &code
	a.python = python
	a.pyErr = pyErr
	a.missingTrainingAt = time.Time{}
	a.metricsAt = time.Time{}
	a.artifactsAt = time.Time{}
	a.mu.Unlock()
	a.appendLog(message)
}

func (a *App) metricsCached() []MetricRow {
	a.mu.Lock()
	if time.Since(a.metricsAt) < 250*time.Millisecond {
		cached := append([]MetricRow(nil), a.metricsCache...)
		a.mu.Unlock()
		return cached
	}
	a.mu.Unlock()

	metrics := loadMetrics(filepath.Join(a.repoRoot, metricsFile))
	a.mu.Lock()
	a.metricsCache = append([]MetricRow(nil), metrics...)
	a.metricsAt = time.Now()
	a.mu.Unlock()
	return metrics
}

func (a *App) artifactsCached() []ArtifactRow {
	a.mu.Lock()
	if time.Since(a.artifactsAt) < 2*time.Second {
		cached := append([]ArtifactRow(nil), a.artifactsCache...)
		a.mu.Unlock()
		return cached
	}
	a.mu.Unlock()

	artifacts := scanArtifacts(a.repoRoot)
	a.mu.Lock()
	a.artifactsCache = append([]ArtifactRow(nil), artifacts...)
	a.artifactsAt = time.Now()
	a.mu.Unlock()
	return artifacts
}

func (a *App) ensureTrainingReady() error {
	if a.pyErr != nil {
		return fmt.Errorf("%s. Click Setup + Deps", a.pyErr)
	}
	missing := a.missingTrainingPackages()
	if len(missing) == 0 {
		return nil
	}
	venvPython := filepath.Join(a.repoRoot, ".venv", "Scripts", "python.exe")
	if !fileExists(venvPython) {
		return fmt.Errorf("training environment is not set up. Click Setup + Deps")
	}
	return fmt.Errorf("training dependencies missing: %s. Click Setup + Deps and wait for it to finish", strings.Join(missing, ", "))
}

func (a *App) missingTrainingPackagesCached() []string {
	a.mu.Lock()
	if time.Since(a.missingTrainingAt) < 5*time.Second {
		cached := append([]string(nil), a.missingTrainingCache...)
		a.mu.Unlock()
		return cached
	}
	a.mu.Unlock()
	missing := a.missingTrainingPackages()
	a.mu.Lock()
	a.missingTrainingCache = append([]string(nil), missing...)
	a.missingTrainingAt = time.Now()
	a.mu.Unlock()
	return missing
}

func (a *App) missingTrainingPackages() []string {
	if a.pyErr != nil || a.python == "" {
		return []string{"python"}
	}
	code := strings.Join([]string{
		"import importlib.util",
		"checks={'stable-baselines3':'stable_baselines3','gymnasium':'gymnasium','torch':'torch','robotpy':'robotpy','pygame':'pygame','numpy':'numpy'}",
		"print(','.join(name for name,mod in checks.items() if importlib.util.find_spec(mod) is None))",
	}, "; ")
	cmd := exec.Command(a.python, "-c", code)
	cmd.Dir = a.repoRoot
	hideConsoleWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return []string{"training package check failed"}
	}
	text := strings.TrimSpace(string(out))
	if text == "" {
		return nil
	}
	return strings.Split(text, ",")
}

func streamLines(wg *sync.WaitGroup, reader io.Reader, appendLog func(string)) {
	defer wg.Done()
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		appendLog(scanner.Text())
	}
}

func (a *App) appendLog(line string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	stamp := time.Now().Format("15:04:05")
	a.logs = append(a.logs, "["+stamp+"] "+line)
	if len(a.logs) > maxLogLines {
		a.logs = a.logs[len(a.logs)-maxLogLines:]
	}
}

func (a *App) pythonOrFallback() string {
	if a.pyErr == nil && a.python != "" {
		return a.python
	}
	return "python"
}

func fallback(value string, defaultValue string) string {
	if strings.TrimSpace(value) == "" {
		return defaultValue
	}
	return strings.TrimSpace(value)
}

func loadSnapshotMap(path string) map[string]any {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil
	}
	return payload
}

func loadMetrics(path string) []MetricRow {
	file, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer file.Close()
	rows := []MetricRow{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var raw map[string]any
		if json.Unmarshal(scanner.Bytes(), &raw) != nil {
			continue
		}
		step := int(number(raw["num_timesteps"]))
		if step == 0 {
			step = int(number(raw["step"]))
		}
		rows = append(rows, MetricRow{
			Step:   step,
			Reward: firstNumber(raw, "rollout/ep_rew_mean", "sim/latest_episode_reward", "episode_reward", "reward"),
			Loss:   number(raw["train/loss"]),
			FPS:    number(raw["time/fps"]),
			Event:  fmt.Sprint(raw["event"]),
		})
	}
	if len(rows) > 300 {
		rows = rows[len(rows)-300:]
	}
	return rows
}

func firstNumber(row map[string]any, keys ...string) float64 {
	for _, key := range keys {
		if value, ok := row[key]; ok {
			return number(value)
		}
	}
	return 0
}

func scanArtifacts(repoRoot string) []ArtifactRow {
	rows := []ArtifactRow{}
	for _, dir := range []string{"models", "runs"} {
		root := filepath.Join(repoRoot, dir)
		_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() || !strings.HasSuffix(strings.ToLower(path), ".zip") {
				return nil
			}
			info, err := d.Info()
			if err != nil {
				return nil
			}
			rel, _ := filepath.Rel(repoRoot, path)
			rows = append(rows, ArtifactRow{
				Path:      rel,
				Size:      byteSize(info.Size()),
				SizeBytes: info.Size(),
				Modified:  info.ModTime().Format("2006-01-02 15:04"),
			})
			return nil
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Modified > rows[j].Modified })
	return rows
}

func number(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case string:
		n, _ := strconv.ParseFloat(t, 64)
		return n
	default:
		return 0
	}
}

func byteSize(size int64) string {
	if size > 1024*1024 {
		return fmt.Sprintf("%.1f MB", float64(size)/(1024*1024))
	}
	if size > 1024 {
		return fmt.Sprintf("%.1f KB", float64(size)/1024)
	}
	return fmt.Sprintf("%d B", size)
}
