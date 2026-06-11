package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

type appState struct {
	repoRoot string
	mu       sync.Mutex
	running  bool
	current  string
	logs     []string
}

type commandSpec struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Description string `json:"description"`
	Disabled    string `json:"disabled,omitempty"`
	Kind        string `json:"kind"`
	Args        []string
}

type stateResponse struct {
	RepoRoot        string        `json:"repoRoot"`
	Python          string        `json:"python"`
	PythonAvailable bool          `json:"pythonAvailable"`
	PythonError     string        `json:"pythonError,omitempty"`
	Running         bool          `json:"running"`
	Current         string        `json:"current"`
	Logs            []string      `json:"logs"`
	Commands        []commandSpec `json:"commands"`
}

func main() {
	repoRoot, err := findRepoRoot()
	if err != nil {
		showStartupError(err)
		os.Exit(1)
	}

	state := &appState{repoRoot: repoRoot}
	state.appendLog("REEFSCAPE RL app launcher started.")
	state.appendLog("Repo: " + repoRoot)

	mux := http.NewServeMux()
	mux.HandleFunc("/", state.handleIndex)
	mux.HandleFunc("/api/state", state.handleState)
	mux.HandleFunc("/api/run", state.handleRun)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		showStartupError(err)
		os.Exit(1)
	}
	url := "http://" + listener.Addr().String()
	state.appendLog("GUI: " + url)
	_ = openBrowser(url)

	if err := http.Serve(listener, mux); err != nil && !errors.Is(err, http.ErrServerClosed) {
		showStartupError(err)
		os.Exit(1)
	}
}

func (s *appState) handleIndex(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := pageTemplate.Execute(w, nil); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *appState) handleState(w http.ResponseWriter, _ *http.Request) {
	python, pythonErr := findPython(s.repoRoot)

	s.mu.Lock()
	response := stateResponse{
		RepoRoot:        s.repoRoot,
		Python:          python,
		PythonAvailable: pythonErr == nil,
		Running:         s.running,
		Current:         s.current,
		Logs:            append([]string(nil), s.logs...),
		Commands:        s.commands(python, pythonErr),
	}
	s.mu.Unlock()

	if pythonErr != nil {
		response.PythonError = pythonErr.Error()
	}

	writeJSON(w, response)
}

func (s *appState) handleRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	id := r.URL.Query().Get("id")
	python, pythonErr := findPython(s.repoRoot)
	var selected *commandSpec
	for _, command := range s.commands(python, pythonErr) {
		if command.ID == id {
			copy := command
			selected = &copy
			break
		}
	}
	if selected == nil {
		http.Error(w, "unknown command", http.StatusNotFound)
		return
	}
	if selected.Disabled != "" {
		http.Error(w, selected.Disabled, http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		http.Error(w, "another command is already running", http.StatusConflict)
		return
	}
	s.running = true
	s.current = selected.Label
	s.mu.Unlock()

	go s.runCommand(*selected)
	writeJSON(w, map[string]string{"status": "started"})
}

func (s *appState) commands(python string, pythonErr error) []commandSpec {
	pythonDisabled := ""
	if pythonErr != nil {
		pythonDisabled = pythonErr.Error()
	}

	return []commandSpec{
		{
			ID:          "setup",
			Label:       "Set Up Python Environment",
			Description: "Create or update .venv using scripts/setup_venv.ps1.",
			Kind:        "setup",
			Args:        setupVenvArgs(s.repoRoot),
		},
		{
			ID:          "studio",
			Label:       "Open Training Studio",
			Description: "Launch the browser training UI for configuring and running PPO training.",
			Disabled:    pythonDisabled,
			Kind:        "launch",
			Args:        []string{python, "scripts/training_studio.py"},
		},
		{
			ID:          "visualizer",
			Label:       "Open Custom Visualizer",
			Description: "Launch the REEFSCAPE 2025 visualizer.",
			Disabled:    pythonDisabled,
			Kind:        "launch",
			Args:        []string{python, "scripts/reefscape_visualizer.py"},
		},
		{
			ID:          "deps",
			Label:       "Install Training Dependencies",
			Description: "Install CUDA/PyTorch/RL dependencies. This is the slow download-heavy step.",
			Disabled:    pythonDisabled,
			Kind:        "setup",
			Args:        installTrainingDepsArgs(s.repoRoot),
		},
		{
			ID:          "doctor",
			Label:       "Run Doctor",
			Description: "Check the Python environment and simulator setup.",
			Disabled:    pythonDisabled,
			Kind:        "check",
			Args:        []string{python, "-m", "reefscape_rl.doctor"},
		},
		{
			ID:          "checks",
			Label:       "Run Full Local Checks",
			Description: "Run Ruff, format check, unit tests, and compile checks.",
			Disabled:    pythonDisabled,
			Kind:        "check",
			Args: []string{
				python,
				"-c",
				"import manage; manage.run_full_checks()",
			},
		},
		{
			ID:          "release",
			Label:       "Build Release Artifacts",
			Description: "Build wheel/sdist and release manifest.",
			Disabled:    pythonDisabled,
			Kind:        "build",
			Args:        releaseBuildArgs(s.repoRoot, python),
		},
		{
			ID:          "installer",
			Label:       "Rebuild Installer",
			Description: "Run build.bat --installer to recreate the portable zip and Setup.exe.",
			Kind:        "build",
			Args:        buildInstallerArgs(s.repoRoot),
		},
		{
			ID:          "folder",
			Label:       "Open Project Folder",
			Description: "Open the installed app folder in File Explorer.",
			Kind:        "open",
			Args:        openFolderArgs(s.repoRoot),
		},
	}
}

func (s *appState) runCommand(command commandSpec) {
	defer func() {
		s.mu.Lock()
		s.running = false
		s.current = ""
		s.mu.Unlock()
	}()

	s.appendLog("")
	s.appendLog("==> " + command.Label)
	s.appendLog(strings.Join(command.Args, " "))

	if len(command.Args) == 0 {
		s.appendLog("No command configured.")
		return
	}

	cmd := exec.Command(command.Args[0], command.Args[1:]...)
	cmd.Dir = s.repoRoot

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		s.appendLog("stdout error: " + err.Error())
		return
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		s.appendLog("stderr error: " + err.Error())
		return
	}

	if err := cmd.Start(); err != nil {
		s.appendLog("start error: " + err.Error())
		return
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go copyLines(&wg, stdout, s.appendLog)
	go copyLines(&wg, stderr, s.appendLog)
	wg.Wait()

	if err := cmd.Wait(); err != nil {
		s.appendLog("finished with error: " + err.Error())
		return
	}
	s.appendLog("finished successfully.")
}

func copyLines(wg *sync.WaitGroup, reader io.Reader, appendLog func(string)) {
	defer wg.Done()
	buffer := make([]byte, 4096)
	var pending strings.Builder
	for {
		n, err := reader.Read(buffer)
		if n > 0 {
			pending.Write(buffer[:n])
			text := pending.String()
			lines := strings.Split(text, "\n")
			for _, line := range lines[:len(lines)-1] {
				appendLog(strings.TrimRight(line, "\r"))
			}
			pending.Reset()
			pending.WriteString(lines[len(lines)-1])
		}
		if err != nil {
			if pending.Len() > 0 {
				appendLog(strings.TrimRight(pending.String(), "\r"))
			}
			return
		}
	}
}

func (s *appState) appendLog(line string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	timestamp := time.Now().Format("15:04:05")
	s.logs = append(s.logs, "["+timestamp+"] "+line)
	if len(s.logs) > 600 {
		s.logs = s.logs[len(s.logs)-600:]
	}
}

func findRepoRoot() (string, error) {
	exe, err := os.Executable()
	if err == nil {
		if root, ok := walkForFile(filepath.Dir(exe), "pyproject.toml"); ok {
			return root, nil
		}
	}

	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	if root, ok := walkForFile(cwd, "pyproject.toml"); ok {
		return root, nil
	}

	return "", errors.New("pyproject.toml was not found above the executable or working directory")
}

func walkForFile(start string, name string) (string, bool) {
	current, err := filepath.Abs(start)
	if err != nil {
		return "", false
	}
	for {
		if _, err := os.Stat(filepath.Join(current, name)); err == nil {
			return current, true
		}
		parent := filepath.Dir(current)
		if parent == current {
			return "", false
		}
		current = parent
	}
}

func findPython(repoRoot string) (string, error) {
	if configured := strings.TrimSpace(os.Getenv("REEFSCAPE_PYTHON")); configured != "" {
		if fileExists(configured) {
			return configured, nil
		}
		return "", fmt.Errorf("REEFSCAPE_PYTHON points to missing file: %s", configured)
	}

	candidates := []string{
		filepath.Join(repoRoot, ".venv", "Scripts", "python.exe"),
		filepath.Join(repoRoot, ".venv", "bin", "python"),
	}
	for _, candidate := range candidates {
		if fileExists(candidate) {
			return candidate, nil
		}
	}

	if path, err := exec.LookPath("python"); err == nil && fileExists(path) {
		return path, nil
	}
	if path, err := exec.LookPath("py"); err == nil && fileExists(path) {
		return path, nil
	}

	return "", errors.New("click Set Up Python Environment, install Python, or set REEFSCAPE_PYTHON")
}

func releaseBuildArgs(repoRoot string, python string) []string {
	script := filepath.Join(repoRoot, "scripts", "build_release.ps1")
	return powershellArgs(script, "-PythonExe", python)
}

func setupVenvArgs(repoRoot string) []string {
	return powershellArgs(filepath.Join(repoRoot, "scripts", "setup_venv.ps1"), "-BootstrapPython", "-SkipRequirements")
}

func installTrainingDepsArgs(repoRoot string) []string {
	return powershellArgs(filepath.Join(repoRoot, "scripts", "install_training_deps.ps1"))
}

func buildInstallerArgs(repoRoot string) []string {
	if runtime.GOOS == "windows" {
		return []string{"cmd", "/c", filepath.Join(repoRoot, "build.bat"), "--installer"}
	}
	return []string{"python", filepath.Join(repoRoot, "build.py"), "--installer"}
}

func openFolderArgs(repoRoot string) []string {
	if runtime.GOOS == "windows" {
		return []string{"explorer", repoRoot}
	}
	if runtime.GOOS == "darwin" {
		return []string{"open", repoRoot}
	}
	return []string{"xdg-open", repoRoot}
}

func powershellArgs(script string, extra ...string) []string {
	if runtime.GOOS == "windows" {
		if pwsh, err := exec.LookPath("pwsh"); err == nil {
			return append([]string{pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script}, extra...)
		}
		if powershell, err := exec.LookPath("powershell"); err == nil {
			return append([]string{powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script}, extra...)
		}
	}
	return append([]string{"pwsh", "-NoProfile", "-File", script}, extra...)
}

func openBrowser(url string) error {
	switch runtime.GOOS {
	case "windows":
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		return exec.Command("open", url).Start()
	default:
		return exec.Command("xdg-open", url).Start()
	}
}

func showStartupError(err error) {
	if runtime.GOOS == "windows" {
		_ = exec.Command("powershell", "-NoProfile", "-Command",
			"Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show($args[0], 'Reefscape RL')",
			err.Error(),
		).Run()
		return
	}
	fmt.Fprintln(os.Stderr, err)
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(value); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

var pageTemplate = template.Must(template.New("page").Parse(`<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reefscape RL</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101418;
      --panel: #171d22;
      --panel-2: #1f272e;
      --text: #e8eef2;
      --muted: #8ea0ad;
      --cyan: #24d8db;
      --orange: #ff9f2f;
      --green: #45d475;
      --red: #ff5d5d;
      --border: #30404b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 "Segoe UI", system-ui, sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 24px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      background: #12181d;
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(360px, 1fr);
      min-height: calc(100vh - 70px);
    }
    aside {
      padding: 18px;
      border-right: 1px solid var(--border);
      background: var(--panel);
    }
    section { padding: 18px; }
    .status {
      display: grid;
      gap: 8px;
      margin-bottom: 16px;
      color: var(--muted);
      word-break: break-word;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
    }
    .ok { color: var(--green); }
    .bad { color: var(--red); }
    .commands { display: grid; gap: 10px; }
    button {
      width: 100%;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      text-align: left;
      padding: 12px;
      cursor: pointer;
      border-radius: 6px;
    }
    button:hover:not(:disabled) { border-color: var(--cyan); }
    button:disabled {
      cursor: not-allowed;
      opacity: .48;
    }
    .button-title {
      display: block;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .button-desc {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .log-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }
    pre {
      height: calc(100vh - 140px);
      margin: 0;
      overflow: auto;
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #080b0d;
      color: #d5e8d9;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 Consolas, "Cascadia Mono", monospace;
    }
    @media (max-width: 840px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--border); }
      pre { height: 45vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Reefscape RL</h1>
    <span id="run-state" class="pill">Loading...</span>
  </header>
  <main>
    <aside>
      <div class="status">
        <div><strong>Repo</strong><br><span id="repo"></span></div>
        <div><strong>Python</strong><br><span id="python"></span></div>
      </div>
      <div id="commands" class="commands"></div>
    </aside>
    <section>
      <div class="log-head">
        <strong>Live Log</strong>
        <span id="current" class="pill">Idle</span>
      </div>
      <pre id="logs"></pre>
    </section>
  </main>
  <script>
    async function refresh() {
      const res = await fetch('/api/state');
      const data = await res.json();
      document.getElementById('repo').textContent = data.repoRoot;
      const py = document.getElementById('python');
      py.textContent = data.pythonAvailable ? data.python : data.pythonError;
      py.className = data.pythonAvailable ? 'ok' : 'bad';
      document.getElementById('run-state').textContent = data.running ? 'Running' : 'Ready';
      document.getElementById('current').textContent = data.running ? data.current : 'Idle';
      document.getElementById('logs').textContent = data.logs.join('\n');

      const container = document.getElementById('commands');
      container.innerHTML = '';
      for (const command of data.commands) {
        const button = document.createElement('button');
        button.disabled = data.running || Boolean(command.disabled);
        button.title = command.disabled || command.description;
        button.innerHTML =
          '<span class="button-title">' + escapeHtml(command.label) + '</span>' +
          '<span class="button-desc">' + escapeHtml(command.disabled || command.description) + '</span>';
        button.onclick = async () => {
          button.disabled = true;
          await fetch('/api/run?id=' + encodeURIComponent(command.id), { method: 'POST' });
          await refresh();
        };
        container.appendChild(button);
      }
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[c]));
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>`))
