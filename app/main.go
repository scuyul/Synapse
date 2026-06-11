package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	cyan    = "\033[38;5;51m"
	orange  = "\033[38;5;208m"
	green   = "\033[38;5;46m"
	magenta = "\033[38;5;201m"
	blue    = "\033[38;5;39m"
	yellow  = "\033[38;5;226m"
	red     = "\033[38;5;196m"
	bold    = "\033[1m"
	reset   = "\033[0m"
)

type commandOption struct {
	key         string
	label       string
	description string
	args        []string
	disabled    string
	color       string
}

func main() {
	repoRoot, err := findRepoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%sCould not find repo root:%s %v\n", red, reset, err)
		pauseForExplorer()
		os.Exit(1)
	}

	python, pythonErr := findPython(repoRoot)
	pythonDisabled := ""
	if pythonErr != nil {
		pythonDisabled = pythonErr.Error()
	}

	options := []commandOption{
		{
			key:         "1",
			label:       "Training menu",
			description: "Open the existing REEFSCAPE RL control menu.",
			args:        []string{python, "menu.py"},
			disabled:    pythonDisabled,
			color:       cyan,
		},
		{
			key:         "2",
			label:       "Management menu",
			description: "Open checks, release builds, tags, and git helpers.",
			args:        []string{python, "manage.py"},
			disabled:    pythonDisabled,
			color:       orange,
		},
		{
			key:         "3",
			label:       "Training Studio",
			description: "Start the browser-based training UI.",
			args:        []string{python, "scripts/training_studio.py"},
			disabled:    pythonDisabled,
			color:       green,
		},
		{
			key:         "4",
			label:       "Custom visualizer",
			description: "Start the REEFSCAPE 2025 browser visualizer.",
			args:        []string{python, "scripts/reefscape_visualizer.py"},
			disabled:    pythonDisabled,
			color:       magenta,
		},
		{
			key:         "5",
			label:       "Doctor",
			description: "Run environment diagnostics.",
			args:        []string{python, "-m", "reefscape_rl.doctor"},
			disabled:    pythonDisabled,
			color:       blue,
		},
		{
			key:         "6",
			label:       "Full local checks",
			description: "Run Ruff, format check, unit tests, and compile checks.",
			args: []string{
				python,
				"-c",
				"import manage; manage.run_full_checks()",
			},
			disabled: pythonDisabled,
			color:    yellow,
		},
		{
			key:         "7",
			label:       "Build release artifacts",
			description: "Run scripts/build_release.ps1 with the selected Python.",
			args:        releaseBuildArgs(repoRoot, python),
			disabled:    pythonDisabled,
			color:       red,
		},
		{
			key:         "8",
			label:       "Set up Python environment",
			description: "Create/update .venv with the repo setup script.",
			args:        setupVenvArgs(repoRoot),
			color:       green,
		},
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		printMenu(repoRoot, python, pythonErr, options)
		choice, err := reader.ReadString('\n')
		if errors.Is(err, io.EOF) && strings.TrimSpace(choice) == "" {
			pauseForExplorer()
			return
		}
		choice = strings.TrimSpace(choice)
		if choice == "9" || strings.EqualFold(choice, "q") || strings.EqualFold(choice, "quit") {
			return
		}

		option, ok := lookupOption(options, choice)
		if !ok {
			fmt.Println("Invalid option.")
			continue
		}

		if option.disabled != "" {
			fmt.Printf("%s%s is unavailable:%s %s\n", red, option.label, reset, option.disabled)
			pause(reader)
			continue
		}

		if err := run(repoRoot, option); err != nil {
			fmt.Fprintf(os.Stderr, "%sCommand failed:%s %v\n", red, reset, err)
		}
		pause(reader)
	}
}

func printMenu(repoRoot string, python string, pythonErr error, options []commandOption) {
	fmt.Println()
	fmt.Printf("%s%sREEFSCAPE RL App%s\n", bold, orange, reset)
	fmt.Printf("%s================%s\n", orange, reset)
	fmt.Printf("%sRepo:%s   %s\n", cyan, reset, repoRoot)
	if pythonErr == nil {
		fmt.Printf("%sPython:%s %s\n", green, reset, python)
	} else {
		fmt.Printf("%sPython:%s unavailable - %s\n", red, reset, pythonErr)
	}
	fmt.Println()
	for _, option := range options {
		itemColor := option.color
		if itemColor == "" {
			itemColor = cyan
		}
		state := ""
		if option.disabled != "" {
			itemColor = red
			state = " [unavailable]"
		}
		fmt.Printf("%s%s%s.%s %s%s%s\n", itemColor, bold, option.key, reset, itemColor, option.label, reset)
		if option.description != "" {
			fmt.Printf("   %s%s%s\n", blue, option.description, reset)
		}
		if state != "" {
			fmt.Printf("   %s%s%s\n", red, strings.TrimSpace(state), reset)
		}
	}
	fmt.Printf("%s%s9.%s %sExit%s\n", magenta, bold, reset, magenta, reset)
	fmt.Printf("%sSelect option:%s ", orange, reset)
}

func lookupOption(options []commandOption, key string) (commandOption, bool) {
	for _, option := range options {
		if option.key == key {
			return option, true
		}
	}
	return commandOption{}, false
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

	if path, err := exec.LookPath("python"); err == nil {
		return path, nil
	}
	if path, err := exec.LookPath("py"); err == nil {
		return path, nil
	}

	return "", errors.New("activate .venv, install Python, or set REEFSCAPE_PYTHON")
}

func releaseBuildArgs(repoRoot string, python string) []string {
	script := filepath.Join(repoRoot, "scripts", "build_release.ps1")
	return powershellArgs(script, "-PythonExe", python)
}

func setupVenvArgs(repoRoot string) []string {
	return powershellArgs(filepath.Join(repoRoot, "scripts", "setup_venv.ps1"))
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

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func run(repoRoot string, option commandOption) error {
	if len(option.args) == 0 {
		return errors.New("empty command")
	}

	fmt.Println()
	fmt.Printf("%sRunning:%s\n", orange, reset)
	fmt.Println(strings.Join(option.args, " "))
	fmt.Println()

	cmd := exec.Command(option.args[0], option.args[1:]...)
	cmd.Dir = repoRoot
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func pause(reader *bufio.Reader) {
	fmt.Printf("\n%sPress Enter to return to the menu...%s", orange, reset)
	_, _ = reader.ReadString('\n')
}

func pauseForExplorer() {
	if stat, err := os.Stdin.Stat(); err == nil && (stat.Mode()&os.ModeCharDevice) != 0 {
		fmt.Printf("\n%sPress Enter to close...%s", orange, reset)
		_, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	}
}
