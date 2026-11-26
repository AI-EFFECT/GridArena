"""Validates and unpacks an uploaded submission into its ``code/`` directory.

A submission is either a single ``.py`` file, or a ``.zip`` holding a
``submission.py`` plus whatever it needs (helper modules, model weights, ...).
The ``.zip`` may have ``submission.py`` at its root or inside one top-level
folder (what zipping a folder produces).
"""

import ast
import io
import zipfile
from pathlib import Path, PurePosixPath

ENTRYPOINT = "submission.py"

MAX_ZIP_MEMBERS = 2000
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024  # 1 GB


class SubmissionError(ValueError):
    """The uploaded submission is malformed; the message is shown to the user."""


def check_entrypoint(source: str, function: str) -> None:
    """Fail early (before queueing a run) on syntax errors or a missing function."""
    try:
        tree = ast.parse(source, filename=ENTRYPOINT)
    except SyntaxError as e:
        raise SubmissionError(f"{ENTRYPOINT} has a syntax error on line {e.lineno}: {e.msg}")

    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function:
            raise SubmissionError(f"'{function}' must be a regular function, not 'async def'.")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function:
            return
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == function for t in node.targets
        ):
            return
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == function:
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name) == function for alias in node.names
        ):
            return
    raise SubmissionError(f"{ENTRYPOINT} must define a top-level function '{function}'.")


def _zip_root(names: list) -> str:
    """Return the folder prefix holding submission.py ('' for the zip root)."""
    if ENTRYPOINT in names:
        return ""
    tops = {PurePosixPath(n).parts[0] for n in names if n.strip("/")}
    if len(tops) == 1:
        prefix = f"{tops.pop()}/"
        if f"{prefix}{ENTRYPOINT}" in names:
            return prefix
    raise SubmissionError(f"The .zip must contain {ENTRYPOINT} at its root (or inside a single top-level folder).")


def _unpack_zip(data: bytes, dest: Path) -> None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise SubmissionError("The uploaded file is not a valid .zip archive.")

    with zf:
        members = [m for m in zf.infolist() if not m.filename.startswith("__MACOSX/")]
        if len(members) > MAX_ZIP_MEMBERS:
            raise SubmissionError(f"The .zip has more than {MAX_ZIP_MEMBERS} files.")
        if sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
            raise SubmissionError(f"The .zip unpacks to more than {MAX_UNPACKED_BYTES // 1024 ** 2} MB.")

        root = _zip_root([m.filename for m in members])
        for m in members:
            if not m.filename.startswith(root) or m.is_dir():
                continue
            rel = PurePosixPath(m.filename[len(root):])
            if rel.is_absolute() or ".." in rel.parts or "\\" in m.filename or ":" in m.filename:
                raise SubmissionError(f"Unsafe path in .zip: {m.filename}")
            if (m.external_attr >> 16) & 0o170000 == 0o120000:
                raise SubmissionError(f"Symbolic links are not allowed in the .zip: {m.filename}")
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as out:
                out.write(src.read())


def install_submission(filename: str, data: bytes, dest: Path, function: str) -> None:
    """Validate the upload and write it to ``dest`` (created) with the
    entrypoint at ``dest/submission.py``. Raises SubmissionError."""
    suffix = Path(filename or "").suffix.lower()
    dest.mkdir(parents=True, exist_ok=True)

    if suffix == ".py":
        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError:
            raise SubmissionError("The .py file must be UTF-8 encoded.")
        check_entrypoint(source, function)
        (dest / ENTRYPOINT).write_text(source, encoding="utf-8")
    elif suffix == ".zip":
        _unpack_zip(data, dest)
        check_entrypoint((dest / ENTRYPOINT).read_text(encoding="utf-8", errors="replace"), function)
    else:
        raise SubmissionError("Upload a .py file or a .zip containing submission.py.")
