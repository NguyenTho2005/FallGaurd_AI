#!/usr/bin/env python
"""Download and organize raw Fall Detection datasets for FallGuard AI.

Supported datasets
------------------
1. **UR Fall Detection Dataset (URFD)** -> ``data/raw/urfd/``
   Parses the official dataset webpage dynamically to discover verified download
   links for fall sequences, ADL sequences, and extracted feature CSVs.
   Source: https://fenix.ur.edu.pl/~mkepski/ds/uf.html

2. **Multiple Cameras Fall Dataset (MCFD)** -> ``data/raw/multiple_cameras_fall/``
   Streams the official ZIP archive (24 scenarios across 8 IP cameras: scenarios
   1-22 contain fall and confounding events; scenarios 23-24 contain confounding
   events only) and extracts it preserving original folder structure and filenames.
   Source: https://www.iro.umontreal.ca/~labimage/Dataset/dataset.zip

3. **Le2i / IMVIA Fall Dataset** -> ``data/raw/imvia_fall/``
   Downloads via the official Kaggle API (``tuyenldvn/falldataset-imvia``) using
   user-configured credentials (``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME`` /
   ``KAGGLE_KEY`` environment variables).

Examples
--------
    python scripts/download_datasets.py --init-only
    python scripts/download_datasets.py --dry-run
    python scripts/download_datasets.py --datasets urfd --urfd-types video csv
    python scripts/download_datasets.py --datasets multiple_cameras_fall --keep-zip
    python scripts/download_datasets.py --datasets imvia_fall
    python scripts/download_datasets.py --setup-kaggle-help
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping, Sequence
from urllib.parse import urljoin, urlparse
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.config import get_settings  # noqa: E402
from ai.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Dataset sources (no hardcoded file lists or credentials)
# --------------------------------------------------------------------------- #
URFD_PAGE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
MCFD_ZIP_URL = "https://www.iro.umontreal.ca/~labimage/Dataset/dataset.zip"
LE2I_KAGGLE_DATASET = "tuyenldvn/falldataset-imvia"
LE2I_KAGGLE_URL = "https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia"

DATASET_CHOICES = ("urfd", "multiple_cameras_fall", "imvia_fall")
URFD_TYPE_CHOICES = ("video", "rgb", "depth", "csv", "all")
DEFAULT_URFD_TYPES = ("video", "csv")

_IGNORED_ENTRIES = {".gitkeep", ".DS_Store", "Thumbs.db"}
_CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass(frozen=True, slots=True)
class URFDLink:
    """A verified dataset file link parsed from the URFD webpage."""

    url: str
    filename: str
    category: str
    sequence_type: str
    section: str = ""


@dataclass(slots=True)
class DownloadSummary:
    """Outcome counters for a single dataset operation."""

    dataset: str
    target_dir: Path
    discovered: int = 0
    downloaded: int = 0
    skipped: int = 0
    extracted: int = 0
    failed: int = 0
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Directory layout helpers
# --------------------------------------------------------------------------- #
def default_raw_dir() -> Path:
    """Return the default ``data/raw`` path from application settings."""
    return Path(get_settings().data_dir) / "raw"


def ensure_directory_structure(raw_root: Path | None = None) -> dict[str, Path]:
    """Ensure raw dataset directories exist without modifying existing files."""
    base = Path(raw_root) if raw_root is not None else default_raw_dir()
    dirs: dict[str, Path] = {
        "raw_root": base,
        "urfd": base / "urfd",
        "multiple_cameras_fall": base / "multiple_cameras_fall",
        "imvia_fall": base / "imvia_fall",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _has_existing_data(directory: Path) -> bool:
    """Return True if ``directory`` already contains non-placeholder files."""
    if not directory.is_dir():
        return False
    for entry in directory.iterdir():
        if entry.name in _IGNORED_ENTRIES or entry.name.endswith(".part"):
            continue
        return True
    return False


# --------------------------------------------------------------------------- #
# HTTP session & streaming helpers
# --------------------------------------------------------------------------- #
def build_http_session(
    *,
    retries: int = 3,
    backoff_factor: float = 0.5,
    user_agent: str = "FallGuard-AI-DatasetDownloader/1.0",
) -> requests.Session:
    """Create a ``requests.Session`` configured with bounded retries."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def download_file_stream(
    url: str,
    destination: Path,
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
    force: bool = False,
    resume: bool = True,
    progress_label: str | None = None,
) -> bool:
    """Stream ``url`` to ``destination`` safely using a ``.part`` temporary file.

    Returns ``True`` if a file was downloaded, or ``False`` if it already existed
    and ``force`` is ``False``.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        return False

    owns_session = session is None
    client = session or build_http_session()
    temp_path = destination.with_name(f"{destination.name}.part")
    if force and temp_path.exists():
        temp_path.unlink(missing_ok=True)

    headers: dict[str, str] = {}
    existing_bytes = temp_path.stat().st_size if (resume and temp_path.is_file()) else 0
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    try:
        with client.get(url, stream=True, timeout=timeout, headers=headers) as response:
            if response.status_code == 416 and existing_bytes > 0:
                # Local .part may already be complete or mismatched; restart cleanly.
                temp_path.unlink(missing_ok=True)
                existing_bytes = 0
                return download_file_stream(
                    url,
                    destination,
                    session=client,
                    timeout=timeout,
                    force=True,
                    resume=False,
                    progress_label=progress_label,
                )

            if response.status_code not in (200, 206):
                raise RuntimeError(
                    f"HTTP {response.status_code} while downloading {url}"
                )

            is_resumed = response.status_code == 206 and existing_bytes > 0
            mode = "ab" if is_resumed else "wb"
            base_written = existing_bytes if is_resumed else 0
            content_length = int(response.headers.get("Content-Length") or 0)
            total_expected = (base_written + content_length) if content_length else 0

            written = base_written
            last_logged_pct = -10
            label = progress_label or destination.name
            if is_resumed:
                logger.info(
                    "Resuming %s from %.1f MiB",
                    label,
                    base_written / (1024 * 1024),
                )

            with temp_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if total_expected > 0:
                        pct = int(100 * written / total_expected)
                        if pct >= last_logged_pct + 10:
                            last_logged_pct = pct
                            logger.info(
                                "Downloading %s: %d%% (%.1f / %.1f MiB)",
                                label,
                                min(pct, 100),
                                written / (1024 * 1024),
                                total_expected / (1024 * 1024),
                            )

        if written == 0:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded empty response from {url}")
        if total_expected > 0 and written < total_expected:
            raise RuntimeError(
                f"Incomplete download for {url}: received {written} of {total_expected} bytes"
            )

        temp_path.replace(destination)
        return True
    except (requests.RequestException, OSError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    finally:
        if owns_session:
            client.close()


def safe_extract_zip(
    zip_path: Path,
    dest_dir: Path,
    *,
    overwrite: bool = False,
) -> int:
    """Extract ``zip_path`` into ``dest_dir`` while preventing Zip Slip traversal.

    Preserves the archive's directory layout and filenames. Existing files are
    never overwritten unless ``overwrite=True``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest_dir.resolve()
    extracted_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in members:
            target = (resolved_dest / member.filename).resolve()
            if not target.is_relative_to(resolved_dest):
                raise RuntimeError(
                    f"Unsafe archive entry outside target directory: {member.filename!r}"
                )

        for member in members:
            target = (resolved_dest / member.filename).resolve()
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target.open("wb") as dst:
                while True:
                    buf = src.read(_CHUNK_SIZE)
                    if not buf:
                        break
                    dst.write(buf)
            extracted_count += 1

    return extracted_count


# --------------------------------------------------------------------------- #
# Dataset 1: UR Fall Detection Dataset (dynamic HTML parsing)
# --------------------------------------------------------------------------- #
class _URFDHTMLParser(HTMLParser):
    """Extract section headings and anchor hrefs from the URFD webpage."""

    def __init__(self) -> None:
        super().__init__()
        self._in_heading = False
        self._heading_buf: list[str] = []
        self._current_section = ""
        self._current_href: str | None = None
        self.anchors: list[tuple[str, str]] = []  # (href, section)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("h1", "h2", "h3"):
            self._in_heading = True
            self._heading_buf.clear()
        elif tag_lower == "a":
            for key, value in attrs:
                if key.lower() == "href" and value:
                    self._current_href = value.strip()
                    self.anchors.append((self._current_href, self._current_section))
                    break

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("h1", "h2", "h3") and self._in_heading:
            self._in_heading = False
            heading = " ".join("".join(self._heading_buf).split())
            if heading:
                self._current_section = heading
        elif tag_lower == "a":
            self._current_href = None

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading_buf.append(data)


def _classify_urfd_link(url: str, section: str) -> URFDLink | None:
    """Classify a candidate link from the URFD page, or return ``None`` if non-data."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    path_lower = parsed.path.lower()
    if "/data/" not in path_lower:
        return None

    filename = Path(parsed.path).name
    if not filename:
        return None
    lower_name = filename.lower()

    if not lower_name.endswith((".zip", ".csv", ".mp4", ".avi")):
        return None

    if re.search(r"-d\.zip$", lower_name):
        category = "depth"
    elif re.search(r"-rgb\.zip$", lower_name):
        category = "rgb"
    elif lower_name.endswith((".mp4", ".avi")):
        category = "video"
    elif lower_name.endswith(".csv"):
        category = "csv"
    else:
        return None

    if lower_name.startswith("fall-"):
        seq_type = "fall"
    elif lower_name.startswith("adl-"):
        seq_type = "adl"
    elif lower_name.startswith("urfall-") or "feature" in section.lower():
        seq_type = "features"
    else:
        seq_type = "other"

    return URFDLink(
        url=url,
        filename=filename,
        category=category,
        sequence_type=seq_type,
        section=section,
    )


def parse_urfd_html(html_text: str, base_url: str = URFD_PAGE_URL) -> list[URFDLink]:
    """Parse ``html_text`` from the URFD webpage and return verified dataset links."""
    parser = _URFDHTMLParser()
    parser.feed(html_text)
    parser.close()

    base_parsed = urlparse(base_url)
    seen_urls: set[str] = set()
    links: list[URFDLink] = []
    for raw_href, section in parser.anchors:
        resolved = urljoin(base_url, raw_href)
        if urlparse(resolved).netloc != base_parsed.netloc:
            continue
        if resolved in seen_urls:
            continue
        item = _classify_urfd_link(resolved, section)
        if item is not None:
            seen_urls.add(resolved)
            links.append(item)
    return links


def fetch_urfd_links(
    page_url: str = URFD_PAGE_URL,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> list[URFDLink]:
    """Fetch the URFD webpage and extract verified dataset download links."""
    owns_session = session is None
    client = session or build_http_session()
    try:
        logger.info("Fetching UR Fall Detection index page: %s", page_url)
        response = client.get(page_url, timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch URFD page {page_url}: HTTP {response.status_code}"
            )
        links = parse_urfd_html(response.text, base_url=response.url or page_url)
        if not links:
            raise RuntimeError(
                f"No dataset links discovered on {page_url}; webpage structure may have changed."
            )
        return links
    except requests.RequestException as exc:
        raise RuntimeError(f"Error fetching URFD page {page_url}: {exc}") from exc
    finally:
        if owns_session:
            client.close()


def filter_urfd_links(
    links: Sequence[URFDLink],
    types: Sequence[str] = DEFAULT_URFD_TYPES,
) -> list[URFDLink]:
    """Filter parsed URFD links by requested modality/file types."""
    requested = {item.lower() for item in types}
    invalid = requested - set(URFD_TYPE_CHOICES)
    if invalid:
        raise ValueError(
            f"Unsupported URFD type(s): {sorted(invalid)}. "
            f"Choose from {URFD_TYPE_CHOICES}."
        )
    if "all" in requested:
        return list(links)
    return [link for link in links if link.category in requested]


def download_urfd(
    target_dir: Path,
    *,
    types: Sequence[str] = DEFAULT_URFD_TYPES,
    extract_zips: bool = False,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = 60.0,
    workers: int = 4,
    page_url: str = URFD_PAGE_URL,
) -> DownloadSummary:
    """Download UR Fall Detection Dataset files discovered from the official webpage."""
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = DownloadSummary(dataset="urfd", target_dir=target_dir)

    with build_http_session() as session:
        all_links = fetch_urfd_links(page_url=page_url, session=session, timeout=timeout)
        selected = filter_urfd_links(all_links, types=types)
        summary.discovered = len(selected)

        counts_by_cat: dict[str, int] = {}
        for item in all_links:
            counts_by_cat[item.category] = counts_by_cat.get(item.category, 0) + 1
        logger.info(
            "URFD webpage parsed: %d total data links (%s); selected %d for types=%s",
            len(all_links),
            ", ".join(f"{k}={v}" for k, v in sorted(counts_by_cat.items())),
            len(selected),
            list(types),
        )

        if dry_run:
            summary.notes.append(
                f"Dry-run: {len(selected)} URFD files matched types={list(types)} "
                f"(total discovered={len(all_links)})"
            )
            return summary

        def _download_one(link: URFDLink) -> tuple[URFDLink, Path, bool]:
            dest = target_dir / link.sequence_type / link.filename
            did_download = download_file_stream(
                link.url,
                dest,
                session=session,
                timeout=timeout,
                force=force,
                resume=True,
                progress_label=link.filename,
            )
            return link, dest, did_download

        max_workers = max(1, min(int(workers), 8))
        downloaded_paths: list[tuple[URFDLink, Path]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_download_one, link): link for link in selected}
            for future in as_completed(futures):
                link = futures[future]
                try:
                    _, dest, did_download = future.result()
                    if did_download:
                        summary.downloaded += 1
                    else:
                        summary.skipped += 1
                    downloaded_paths.append((link, dest))
                except Exception as exc:
                    summary.failed += 1
                    logger.error("URFD download failed for %s: %s", link.url, exc)

        if extract_zips:
            for link, dest in downloaded_paths:
                if dest.suffix.lower() != ".zip" or not dest.is_file():
                    continue
                extract_target = dest.parent / dest.stem
                try:
                    summary.extracted += safe_extract_zip(
                        dest, extract_target, overwrite=force
                    )
                except Exception as exc:
                    summary.failed += 1
                    logger.error("Failed to extract %s: %s", dest.name, exc)

    if summary.failed > 0:
        raise RuntimeError(
            f"URFD finished with {summary.failed} error(s) out of {summary.discovered} files."
        )
    return summary


# --------------------------------------------------------------------------- #
# Dataset 2: Multiple Cameras Fall Dataset
# --------------------------------------------------------------------------- #
def _summarize_mcfd_directory(target_dir: Path) -> str:
    """Summarize extracted scenario directories without modifying files."""
    if not target_dir.is_dir():
        return "0 entries"
    subdirs = sorted(
        entry.name
        for entry in target_dir.iterdir()
        if entry.is_dir() and entry.name not in _IGNORED_ENTRIES
    )
    files = [
        entry.name
        for entry in target_dir.iterdir()
        if entry.is_file() and entry.name not in _IGNORED_ENTRIES
    ]
    return (
        f"{len(subdirs)} top-level folders, {len(files)} top-level files "
        f"(dataset spec: 24 scenarios x 8 IP cameras; "
        f"scenarios 1-22 falls+confounding, scenarios 23-24 confounding only)"
    )


def download_multiple_cameras_fall(
    target_dir: Path,
    *,
    zip_url: str = MCFD_ZIP_URL,
    keep_zip: bool = False,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = 120.0,
) -> DownloadSummary:
    """Download and extract the Multiple Cameras Fall Dataset ZIP archive."""
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = DownloadSummary(dataset="multiple_cameras_fall", target_dir=target_dir)
    summary.discovered = 1

    zip_path = target_dir / "dataset.zip"
    preexisting_zip = zip_path.is_file() and zip_path.stat().st_size > 0
    existing_subdirs = [
        entry
        for entry in target_dir.iterdir()
        if entry.is_dir() and entry.name not in _IGNORED_ENTRIES
    ]

    if existing_subdirs and not force:
        summary.skipped = 1
        msg = (
            f"Multiple Cameras Fall Dataset already present at {target_dir} "
            f"({_summarize_mcfd_directory(target_dir)}); skipping without overwriting."
        )
        logger.info(msg)
        summary.notes.append(msg)
        return summary

    if dry_run:
        with build_http_session() as session:
            resp = session.head(zip_url, allow_redirects=True, timeout=timeout)
            size_bytes = int(resp.headers.get("Content-Length") or 0)
            size_gib = size_bytes / (1024**3) if size_bytes else 0.0
            msg = (
                f"Dry-run: verified {zip_url} (HTTP {resp.status_code}, "
                f"~{size_gib:.2f} GiB) -> {target_dir}"
            )
            logger.info(msg)
            summary.notes.append(msg)
        return summary

    did_download = download_file_stream(
        zip_url,
        zip_path,
        timeout=timeout,
        force=force,
        resume=True,
        progress_label="multiple_cameras_fall/dataset.zip",
    )
    if did_download:
        summary.downloaded = 1
    else:
        summary.skipped = 1

    logger.info("Extracting %s into %s (preserving original structure)...", zip_path, target_dir)
    summary.extracted = safe_extract_zip(zip_path, target_dir, overwrite=force)
    logger.info(
        "Extracted %d file(s). %s",
        summary.extracted,
        _summarize_mcfd_directory(target_dir),
    )

    # Never delete a zip file that existed before this run.
    if did_download and not preexisting_zip and not keep_zip:
        zip_path.unlink(missing_ok=True)
        logger.info("Removed temporary archive %s (pass --keep-zip to retain it)", zip_path)

    return summary


# --------------------------------------------------------------------------- #
# Dataset 3: Le2i Fall Dataset (Kaggle API)
# --------------------------------------------------------------------------- #
def kaggle_setup_instructions() -> str:
    """Return clear setup instructions for Kaggle API authentication."""
    return (
        "Kaggle API credentials are not configured.\n"
        "Configure authentication using ONE of the following methods (never hardcode keys):\n"
        "\n"
        "Method 1 (Recommended) - kaggle.json file:\n"
        "  1. Sign in to Kaggle -> Account -> Settings -> API -> 'Create New Token'.\n"
        "  2. Save the downloaded kaggle.json to:\n"
        "       - Linux/macOS: ~/.kaggle/kaggle.json  (and run: chmod 600 ~/.kaggle/kaggle.json)\n"
        "       - Windows:     %USERPROFILE%\\.kaggle\\kaggle.json\n"
        "     File format: {\"username\": \"<your_kaggle_username>\", \"key\": \"<your_api_key>\"}\n"
        "     (Or set KAGGLE_CONFIG_DIR to the directory containing kaggle.json.)\n"
        "\n"
        "Method 2 - Environment variables:\n"
        "  - Linux/macOS: export KAGGLE_USERNAME=<your_username> KAGGLE_KEY=<your_api_key>\n"
        "  - PowerShell:  $env:KAGGLE_USERNAME=\"<your_username>\"; $env:KAGGLE_KEY=\"<your_api_key>\"\n"
    )


def check_kaggle_credentials(
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> tuple[bool, str]:
    """Verify whether Kaggle API credentials exist without exposing secret values."""
    environ = os.environ if env is None else env
    user_home = Path.home() if home is None else Path(home)

    if environ.get("KAGGLE_USERNAME") and environ.get("KAGGLE_KEY"):
        return True, "Kaggle credentials found via KAGGLE_USERNAME and KAGGLE_KEY environment variables."
    if environ.get("KAGGLE_API_TOKEN"):
        return True, "Kaggle credentials found via KAGGLE_API_TOKEN environment variable."

    config_dir_env = environ.get("KAGGLE_CONFIG_DIR")
    candidates = [Path(config_dir_env) / "kaggle.json"] if config_dir_env else []
    candidates.append(user_home / ".kaggle" / "kaggle.json")

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"Found {candidate} but could not parse valid JSON: {exc}"
        if isinstance(payload, dict) and payload.get("username") and payload.get("key"):
            return True, f"Kaggle credentials found at {candidate}."
        return False, f"Found {candidate} but it is missing 'username' or 'key'."

    return False, kaggle_setup_instructions()


def download_le2i_fall(
    target_dir: Path,
    *,
    dataset_slug: str = LE2I_KAGGLE_DATASET,
    force: bool = False,
    dry_run: bool = False,
) -> DownloadSummary:
    """Download the Le2i / IMVIA Fall Dataset using the official Kaggle API."""
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = DownloadSummary(dataset="imvia_fall", target_dir=target_dir)
    summary.discovered = 1

    if _has_existing_data(target_dir) and not force:
        summary.skipped = 1
        msg = (
            f"Le2i Fall Dataset directory {target_dir} is already populated; "
            "skipping without overwriting."
        )
        logger.info(msg)
        summary.notes.append(msg)
        return summary

    creds_ok, creds_msg = check_kaggle_credentials()
    if dry_run:
        status = "ready" if creds_ok else "credentials missing"
        msg = f"Dry-run: Kaggle dataset {dataset_slug} ({LE2I_KAGGLE_URL}) -> {target_dir} [{status}]"
        logger.info(msg)
        summary.notes.append(msg)
        if not creds_ok:
            summary.notes.append(creds_msg)
        return summary

    if not creds_ok:
        raise RuntimeError(creds_msg)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The official 'kaggle' Python package is required to download the Le2i dataset. "
            "Install it with: pip install kaggle"
        ) from exc

    logger.info("Authenticating with official Kaggle API (%s)...", creds_msg)
    api = KaggleApi()
    api.authenticate()

    logger.info(
        "Downloading Kaggle dataset %s into %s...",
        dataset_slug,
        target_dir,
    )
    api.dataset_download_files(
        dataset_slug,
        path=str(target_dir),
        unzip=True,
        force=force,
        quiet=False,
    )
    summary.downloaded = 1
    logger.info("Le2i Fall Dataset downloaded and extracted to %s", target_dir)
    return summary


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=(*DATASET_CHOICES, "all"),
        default=["all"],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override raw dataset root directory (default: data/raw)",
    )
    parser.add_argument(
        "--urfd-types",
        nargs="+",
        choices=URFD_TYPE_CHOICES,
        default=list(DEFAULT_URFD_TYPES),
        help="URFD file categories to download (default: video csv)",
    )
    parser.add_argument(
        "--extract-urfd",
        action="store_true",
        help="Extract downloaded URFD .zip archives into sequence subdirectories",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep dataset.zip after extracting Multiple Cameras Fall Dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing files in target dataset folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover/verify sources and credentials without downloading files",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create the data/raw directory structure only and exit",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent download workers for URFD files (default: 4)",
    )
    parser.add_argument(
        "--setup-kaggle-help",
        action="store_true",
        help="Print Kaggle API authentication instructions and exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.setup_kaggle_help:
        print(kaggle_setup_instructions())
        return 0

    configure_logging()
    dirs = ensure_directory_structure(args.data_dir)
    logger.info("Raw dataset root ready: %s", dirs["raw_root"])

    if args.init_only:
        return 0

    selected_datasets: Iterable[str]
    if "all" in args.datasets:
        selected_datasets = DATASET_CHOICES
    else:
        selected_datasets = tuple(dict.fromkeys(args.datasets))

    summaries: list[DownloadSummary] = []
    errors: list[str] = []

    for name in selected_datasets:
        try:
            if name == "urfd":
                summaries.append(
                    download_urfd(
                        dirs["urfd"],
                        types=args.urfd_types,
                        extract_zips=args.extract_urfd,
                        force=args.force,
                        dry_run=args.dry_run,
                        timeout=args.timeout,
                        workers=args.workers,
                    )
                )
            elif name == "multiple_cameras_fall":
                summaries.append(
                    download_multiple_cameras_fall(
                        dirs["multiple_cameras_fall"],
                        keep_zip=args.keep_zip,
                        force=args.force,
                        dry_run=args.dry_run,
                        timeout=max(args.timeout, 120.0),
                    )
                )
            elif name == "imvia_fall":
                summaries.append(
                    download_le2i_fall(
                        dirs["imvia_fall"],
                        force=args.force,
                        dry_run=args.dry_run,
                    )
                )
        except Exception as exc:
            logger.error("Dataset '%s' failed: %s", name, exc)
            errors.append(f"{name}: {exc}")

    for summary in summaries:
        logger.info(
            "Summary [%s] -> %s | discovered=%d downloaded=%d skipped=%d extracted=%d failed=%d",
            summary.dataset,
            summary.target_dir,
            summary.discovered,
            summary.downloaded,
            summary.skipped,
            summary.extracted,
            summary.failed,
        )
        for note in summary.notes:
            print(note)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
