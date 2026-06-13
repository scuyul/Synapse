package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

func findRepoRoot() (string, error) {
	if exe, err := os.Executable(); err == nil {
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
		filepath.Join(repoRoot, ".python", "python.exe"),
		filepath.Join(repoRoot, ".python", "bin", "python"),
	}
	for _, candidate := range candidates {
		if fileExists(candidate) {
			return candidate, nil
		}
	}
	for _, name := range []string{"python", "py"} {
		if path, err := exec.LookPath(name); err == nil && fileExists(path) {
			return path, nil
		}
	}
	return "", errors.New("run setup, install Python, or set REEFSCAPE_PYTHON")
}

func setupVenvArgs(repoRoot string) []string {
	packagedPython := filepath.Join(repoRoot, ".python", "python.exe")
	if runtime.GOOS != "windows" {
		packagedPython = filepath.Join(repoRoot, ".python", "bin", "python")
	}
	if fileExists(packagedPython) {
		return powershellArgs(filepath.Join(repoRoot, "scripts", "setup_venv.ps1"), "-Python", packagedPython, "-SkipRequirements", "-FastAppInstall")
	}
	return powershellArgs(filepath.Join(repoRoot, "scripts", "setup_venv.ps1"), "-SkipRequirements")
}

func installTrainingDepsArgs(repoRoot string) []string {
	return powershellArgs(filepath.Join(repoRoot, "scripts", "install_training_deps.ps1"))
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

func showStartupError(err error) {
	_ = os.MkdirAll("logs", 0o755)
	_ = os.WriteFile(filepath.Join("logs", "desktop_startup_error.txt"), []byte(err.Error()), 0o644)
	if runtime.GOOS == "windows" {
		_ = exec.Command("powershell", "-NoProfile", "-Command",
			"Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show($args[0], 'Reefscape RL')",
			err.Error(),
		).Run()
		return
	}
	fmt.Fprintln(os.Stderr, err)
}
