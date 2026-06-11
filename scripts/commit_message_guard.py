"""Clean or reject low-quality local commit messages.

Git hooks call this script from `.githooks/commit-msg` and `.githooks/pre-push`.
It intentionally uses deterministic rules instead of an external service.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


VAGUE_EXACT = {
    "3w",
    "asdf",
    "changes",
    "code",
    "codee",
    "fix",
    "fixed",
    "hopefully this works",
    "idk",
    "mid",
    "more code",
    "ok",
    "ok so its great now!",
    "pog",
    "pog code",
    "stuff",
    "test",
    "update",
    "updates",
    "wip",
    "work",
    "works",
    "works and is pog",
}

VAGUE_PATTERNS = [
    re.compile(r"^havent tested", re.IGNORECASE),
    re.compile(r"^work in progress", re.IGNORECASE),
    re.compile(r"^.+\b(jiber|gibber|bs)\b.*$", re.IGNORECASE),
]


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def is_vague(subject: str) -> bool:
    normalized = subject.strip().lower()
    if len(normalized) < 8:
        return True
    if normalized in VAGUE_EXACT:
        return True
    return any(pattern.match(subject.strip()) for pattern in VAGUE_PATTERNS)


def generated_subject(files: list[str]) -> str:
    paths = [path.replace("\\", "/") for path in files if path.strip()]
    if not paths:
        return "Update project files"

    if all(path.startswith(("docs/", "README")) or path.endswith(".md") for path in paths):
        return "Update project documentation"
    if all(path.startswith("tests/") for path in paths):
        return "Update tests"
    if any(path == ".github/workflows/tests.yml" for path in paths):
        return "Update CI workflow"
    if any(path.startswith("scripts/training_studio.py") for path in paths):
        return "Update training studio"
    if any(path.startswith("scripts/") for path in paths):
        return "Update utility scripts"
    if any(path.startswith("reefscape_rl/") for path in paths):
        return "Update simulator implementation"
    if any(path.startswith("robot_code/") for path in paths):
        return "Update robot integration"
    if any(
        path
        in {"pyproject.toml", "requirements.txt", "requirements-dev.txt", "LICENSE", ".gitignore"}
        for path in paths
    ):
        return "Update project metadata"
    return "Update project files"


def staged_files() -> list[str]:
    output = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMRTD"])
    return output.splitlines()


def commit_files(commit: str) -> list[str]:
    output = run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit])
    return output.splitlines()


def clean_message_file(message_file: Path) -> int:
    text = message_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return 0

    subject = lines[0].strip()
    if not is_vague(subject):
        return 0

    replacement = generated_subject(staged_files())
    lines[0] = replacement
    ending = "\n" if text.endswith("\n") else ""
    message_file.write_text("\n".join(lines) + ending, encoding="utf-8")
    print(f"commit-msg: replaced vague subject with: {replacement}", file=sys.stderr)
    return 0


def check_range(commit_range: str) -> int:
    output = run_git(["log", "--format=%H%x00%s", commit_range])
    bad: list[tuple[str, str, str]] = []
    for line in output.splitlines():
        commit, subject = line.split("\x00", 1)
        if is_vague(subject):
            bad.append((commit[:12], subject, generated_subject(commit_files(commit))))

    if not bad:
        return 0

    print("pre-push: vague commit subjects found:", file=sys.stderr)
    for commit, subject, suggestion in bad:
        print(f"  {commit}  {subject!r} -> {suggestion!r}", file=sys.stderr)
    print("Run an interactive rebase or amend these before pushing.", file=sys.stderr)
    return 1


def pre_push(remote_ref: str | None = None) -> int:
    upstream = remote_ref or ""
    if not upstream:
        try:
            upstream = run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
        except subprocess.CalledProcessError:
            return 0

    try:
        merge_base = run_git(["merge-base", "HEAD", upstream])
    except subprocess.CalledProcessError:
        return 0
    return check_range(f"{merge_base}..HEAD")


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "commit-msg":
        return clean_message_file(Path(argv[2]))
    if len(argv) >= 3 and argv[1] == "check-range":
        return check_range(argv[2])
    if len(argv) >= 2 and argv[1] == "pre-push":
        return pre_push(argv[2] if len(argv) > 2 else None)

    print(
        "usage: commit_message_guard.py commit-msg <file> | check-range <range> | pre-push [upstream]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
