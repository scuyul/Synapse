package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

type commandOption struct {
	key         string
	label       string
	description string
	args        []string
}

func main() {
	repoRoot, err := findRepoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not find repo root: %v\n", err)
		os.Exit(1)
	}

	python, err := findPython(repoRoot)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not find Python: %v\n", err)
		os.Exit(1)
	}

	options := []commandOption{
		{
			key:         "1",
			label:       "Training menu",
			description: "Open the existing REEFSCAPE RL control menu.",
			args:        []string{python, "menu.py"},
		},
		{
			key:         "2",
			label:       "Management menu",
			description: "Open checks, release builds, tags, and git helpers.",
			args:        []string{python, "manage.py"},
		},
		{
			key:         "3",
			label:       "Training Studio",
			description: "Start the browser-based training UI.",
			args:        []string{python, "scripts/training_studio.py"},
		},
		{
			key:         "4",
			label:       "Custom visualizer",
			description: "Start the REEFSCAPE 2025 browser visualizer.",
			args:        []string{python, "scripts/reefscape_visualizer.py"},
		},
		{
			key:         "5",
			label:       "Doctor",
			description: "Run environment diagnostics.",
			args:        []string{python, "-m", "reefscape_rl.doctor"},
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
		},
		{
			key:         "7",
			label:       "Build release artifacts",
			description: "Run scripts/build_release.ps1 with the selected Python.",
			args:        releaseBuildArgs(repoRoot, python),
		},
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		printMenu(repoRoot, python, options)
		choice, _ := reader.ReadString('\n')
		choice = strings.TrimSpace(choice)
		if choice == "8" || strings.EqualFold(choice, "q") || strings.EqualFold(choice, "quit") {
			return
		}

		option, ok := lookupOption(options, choice)
		if !ok {
			fmt.Println("Invalid option.")
			continue
		}

		if err := run(repoRoot, option); err != nil {
			fmt.Fprintf(os.Stderr, "Command failed: %v\n", err)
		}
	}
}

func printMenu(repoRoot string, python string, options []commandOption) {
	fmt.Println()
	fmt.Println("REEFSCAPE RL App")
	fmt.Println("================")
	fmt.Printf("Repo:   %s\n", repoRoot)
	fmt.Printf("Python: %s\n", python)
	fmt.Println()
	for _, option := range options {
		fmt.Printf("%s. %s\n", option.key, option.label)
		if option.description != "" {
			fmt.Printf("   %s\n", option.description)
		}
	}
	fmt.Println("8. Exit")
	fmt.Print("Select option: ")
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
	if runtime.GOOS == "windows" {
		if pwsh, err := exec.LookPath("pwsh"); err == nil {
			return []string{pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-PythonExe", python}
		}
		if powershell, err := exec.LookPath("powershell"); err == nil {
			return []string{powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-PythonExe", python}
		}
	}
	return []string{"pwsh", "-NoProfile", "-File", script, "-PythonExe", python}
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
	fmt.Println("Running:")
	fmt.Println(strings.Join(option.args, " "))
	fmt.Println()

	cmd := exec.Command(option.args[0], option.args[1:]...)
	cmd.Dir = repoRoot
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
