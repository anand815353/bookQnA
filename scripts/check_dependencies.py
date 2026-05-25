"""
Lightweight dependency health check for the BookQnA LangChain 0.3 stack.

Run from the bookQnA directory: python scripts/check_dependencies.py
"""

from __future__ import annotations

import ast
import sys
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VERSION_KEYS = [
    ("fastapi", "fastapi"),
    ("langchain", "langchain"),
    ("langchain_core", "langchain-core"),
    ("langchain_community", "langchain-community"),
    ("langchain_text_splitters", "langchain-text-splitters"),
    ("langchain_google_genai", "langchain-google-genai"),
    ("langchain_huggingface", "langchain-huggingface"),
    ("langchain_chroma", "langchain-chroma"),
    ("langsmith", "langsmith"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("pydantic", "pydantic"),
]


def _version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def _major_int(version: str) -> int:
    return int(version.split(".", 1)[0])


def _iter_project_py_files() -> list[Path]:
    out: list[Path] = []
    for p in PROJECT_ROOT.rglob("*.py"):
        if ".venv" in p.parts or "__pycache__" in p.parts:
            continue
        out.append(p)
    return out


def _project_imports_langchain_openai() -> bool:
    for path in _iter_project_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "langchain_openai" or alias.name.startswith(
                        "langchain_openai."
                    ):
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and (
                    node.module == "langchain_openai"
                    or node.module.startswith("langchain_openai.")
                ):
                    return True
    return False


def _project_imports_langgraph() -> bool:
    for path in _iter_project_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "langgraph" or alias.name.startswith("langgraph."):
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and (
                    node.module == "langgraph" or node.module.startswith("langgraph.")
                ):
                    return True
    return False


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    print(sys.executable)
    print()

    errors: list[str] = []

    lc = _version("langchain-core")
    if lc is None:
        errors.append("langchain-core is not installed.")
    elif _major_int(lc) >= 1:
        errors.append(
            f"langchain-core major version is {_major_int(lc)} (expected 0.3.x; use constraints.txt)."
        )

    if _version("langchain-classic") is not None:
        errors.append(
            "langchain-classic is installed but this project targets LangChain 0.3; uninstall it."
        )

    openai_installed = _version("langchain-openai") is not None
    if openai_installed and not _project_imports_langchain_openai():
        errors.append(
            "langchain-openai is installed but the BookQnA codebase does not import it; remove it to avoid core 1.x pulls."
        )

    prebuilt_installed = _version("langgraph-prebuilt") is not None
    if prebuilt_installed and not _project_imports_langgraph():
        errors.append(
            "langgraph-prebuilt is installed but this project does not import langgraph; uninstall langgraph-prebuilt (and related langgraph packages) unless you add LangGraph features."
        )

    print("--- Installed versions (importlib.metadata) ---")
    for label, dist in VERSION_KEYS:
        v = _version(dist)
        line = f"  {label}: {v if v else '(not installed)'}"
        print(line)

    print()
    if errors:
        print("Dependency check FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("Dependency check OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
