"""PDF text from arXiv, converted locally (build plan doc-5wh, Task 10).

In the pilot, web fetches returned raw PDF bytes, so PDFs are downloaded to a
cache in the repo (`cache/pdf/`, ignored by git) and turned into text with
`pdftotext -layout`. Only arXiv's PDF host may be fetched.

A downloaded PDF is untrusted input to a C parser, so `pdftotext` runs under
bwrap: no network, a read-only view of the system, no access to the user's
files, and a time limit. Without bwrap it does not run.
"""

import hashlib
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import minilab
from minilab.literature.http import (
    HostNotAllowed,
    LiteratureHttp,
    SourceUnavailable,
    check_url,
    default_client,
)

PDF_HOST = "arxiv.org"
MAX_PDF_BYTES = 64 * 1024**2
MAX_TEXT_BYTES = 16 * 1024**2
PDFTOTEXT_TIMEOUT_S = 60.0


class PdfUnavailable(SourceUnavailable):
    """A PDF could not be fetched or converted."""


def default_cache_dir() -> Path:
    """The repo's `cache/pdf/`, found from where the package lives."""
    return Path(minilab.__file__).resolve().parents[2] / "cache" / "pdf"


def _sandboxed(pdf: Path) -> list[str]:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise PdfUnavailable("pdftotext is not installed")
    if shutil.which("bwrap") is None:
        raise PdfUnavailable("bwrap is not installed; pdftotext is never run without it")
    args = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for link in ("/bin", "/lib", "/lib64", "/sbin"):
        if os.path.islink(link):
            args += ["--symlink", os.readlink(link), link]
        elif os.path.isdir(link):
            args += ["--ro-bind", link, link]
    for path in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/fonts"):
        args += ["--ro-bind-try", path, path]
    args += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    args += ["--ro-bind", str(pdf), "/tmp/in.pdf", "--remount-ro", "/", "--chdir", "/tmp", "--"]
    args += [pdftotext, "-layout", "-enc", "UTF-8", "/tmp/in.pdf", "-"]
    return args


def _convert(pdf: Path, timeout_s: float) -> str:
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            _sandboxed(pdf),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise PdfUnavailable(f"pdftotext timed out after {timeout_s} s") from exc
        out.seek(0)
        text = out.read(MAX_TEXT_BYTES).decode("utf-8", "replace")
        if code != 0:
            err.seek(0)
            detail = err.read(2000).decode("utf-8", "replace").strip()
            raise PdfUnavailable(f"pdftotext exited with code {code}: {detail}")
    return text


def _download(url: str, target: Path, client: LiteratureHttp) -> None:
    body = client.get(url, max_bytes=MAX_PDF_BYTES)
    if not body.startswith(b"%PDF-"):
        raise PdfUnavailable(f"{url} did not return a PDF")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=".download.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def pdf_text(
    url: str,
    *,
    client: LiteratureHttp | None = None,
    cache_dir: Path | None = None,
    timeout_s: float = PDFTOTEXT_TIMEOUT_S,
) -> str:
    """The text of the PDF at `url`, an arXiv PDF link, downloading it once into the cache.

    Raise HostNotAllowed for any other host, and PdfUnavailable (a
    SourceUnavailable) if the PDF cannot be fetched or converted.
    """
    if check_url(url).host != PDF_HOST:
        raise HostNotAllowed(f"PDFs come only from {PDF_HOST}, not {url}")
    directory = cache_dir if cache_dir is not None else default_cache_dir()
    target = directory / f"{hashlib.sha256(url.encode()).hexdigest()[:32]}.pdf"
    if not target.is_file():
        _download(url, target, client or default_client())
    return _convert(target, timeout_s)
