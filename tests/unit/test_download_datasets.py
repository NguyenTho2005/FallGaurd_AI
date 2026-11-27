"""Unit tests for scripts/download_datasets.py (network and Kaggle API mocked)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock, patch
import zipfile

import pytest

from scripts import download_datasets as dl

SAMPLE_URFD_HTML = """
<html>
<body>
  <h2>Fall sequences:</h2>
  <table>
    <tr>
      <td>01</td>
      <td><a href=
        "./data/fall-01-cam0-d.zip">fall-01-cam0-d.zip</a></td>
      <td><a href="./data/fall-01-cam1-d.zip">fall-01-cam1-d.zip</a></td>
      <td><a href="./data/fall-01-cam0-rgb.zip">fall-01-cam0-rgb.zip</a></td>
      <td><a href="./data/fall-01-cam1-rgb.zip">fall-01-cam1-rgb.zip</a></td>
      <td><a href="./data/fall-01-data.csv">fall-01-data.csv</a></td>
      <td><a href="./data/fall-01-acc.csv">fall-01-acc.csv</a></td>
      <td><a href="./data/fall-01-cam0.mp4">cam0</a>
          <a href="./data/fall-01-cam1.mp4">cam1</a></td>
    </tr>
  </table>
  <h2>Activities of Daily Living (ADL) sequences:</h2>
  <table>
    <tr>
    <tr>
      <td>01</td>
      <td><a href="./data/adl-01-cam0-d.zip">adl-01-cam0-d.zip</a></td>
      <td><a href="./data/adl-01-cam0-rgb.zip">adl-01-cam0-rgb.zip</a></td>
      <td><a href="./data/adl-01-data.csv">adl-01-data.csv</a></td>
      <td><a href="./data/adl-01-acc.csv">adl-01-acc.csv</a></td>
      <td><a href="./data/adl-01-cam0.mp4">cam0</a></td>
    </tr>
  </table>
  <h2>Extracted features:</h2>
  <table>
    <tr>
      <td><a href="./data/urfall-cam0-falls.csv">urfall-cam0-falls.csv</a></td>
      <td><a href="./data/urfall-cam0-adls.csv">urfall-cam0-adls.csv</a></td>
    </tr>
  </table>
  <a href="mailto:mkepski@ur.edu.pl">contact</a>
  <a href="https://example.org/paper.pdf">external</a>
</body>
</html>
"""


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        chunks: list[bytes] | None = None,
        headers: dict[str, str] | None = None,
        url: str = dl.URFD_PAGE_URL,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._chunks = chunks or []
        self.headers = headers or {}
        self.url = url

    def iter_content(self, chunk_size: int = 1024):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_ensure_directory_structure_creates_expected_folders(tmp_path: Path):
    existing = tmp_path / "existing.txt"
    existing.write_text("keep me", encoding="utf-8")

    dirs = dl.ensure_directory_structure(tmp_path)

    assert dirs["urfd"].is_dir()
    assert dirs["multiple_cameras_fall"].is_dir()
    assert dirs["imvia_fall"].is_dir()
    assert existing.read_text(encoding="utf-8") == "keep me"


def test_parse_urfd_html_discovers_and_classifies_links():
    links = dl.parse_urfd_html(SAMPLE_URFD_HTML, base_url=dl.URFD_PAGE_URL)

    assert len(links) == 15
    assert all(link.url.startswith("https://fenix.ur.edu.pl/~mkepski/ds/data/") for link in links)

    videos = dl.filter_urfd_links(links, ["video"])
    assert {v.filename for v in videos} == {
        "fall-01-cam0.mp4",
        "fall-01-cam1.mp4",
        "adl-01-cam0.mp4",
    }

    csvs = dl.filter_urfd_links(links, ["csv"])
    assert len(csvs) == 6
    assert any(c.sequence_type == "features" and c.filename == "urfall-cam0-falls.csv" for c in csvs)

    rgb = dl.filter_urfd_links(links, ["rgb"])
    assert len(rgb) == 3

    depth = dl.filter_urfd_links(links, ["depth"])
    assert len(depth) == 3

    with pytest.raises(ValueError, match="Unsupported URFD type"):
        dl.filter_urfd_links(links, ["invalid"])


def test_download_file_stream_skips_existing_without_force(tmp_path: Path):
    target = tmp_path / "file.mp4"
    target.write_bytes(b"existing-content")

    mock_session = MagicMock()
    did_download = dl.download_file_stream(
        "https://example.org/file.mp4",
        target,
        session=mock_session,
        force=False,
    )

    assert did_download is False
    mock_session.get.assert_not_called()
    assert target.read_bytes() == b"existing-content"


def test_download_file_stream_resumes_partial_file(tmp_path: Path):
    target = tmp_path / "archive.zip"
    part = tmp_path / "archive.zip.part"
    part.write_bytes(b"hello ")

    mock_session = MagicMock()
    mock_session.get.return_value = _FakeResponse(
        status_code=206,
        chunks=[b"world"],
        headers={"Content-Length": "5"},
    )

    did_download = dl.download_file_stream(
        "https://example.org/archive.zip",
        target,
        session=mock_session,
        force=False,
    )

    assert did_download is True
    assert target.read_bytes() == b"hello world"
    assert not part.exists()
    _, kwargs = mock_session.get.call_args
    assert kwargs["headers"]["Range"] == "bytes=6-"


def test_safe_extract_zip_preserves_structure_and_blocks_zip_slip(tmp_path: Path):
    good_zip = tmp_path / "good.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr("chute01/cam1.avi", b"video1")
        zf.writestr("chute23/cam8.avi", b"video2")

    dest = tmp_path / "out"
    dest.mkdir()
    existing = dest / "chute01" / "cam1.avi"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep-original")

    extracted = dl.safe_extract_zip(good_zip, dest, overwrite=False)
    assert extracted == 1
    assert existing.read_bytes() == b"keep-original"
    assert (dest / "chute23" / "cam8.avi").read_bytes() == b"video2"

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", b"evil")

    with pytest.raises(RuntimeError, match="Unsafe archive entry"):
        dl.safe_extract_zip(bad_zip, dest)


def test_download_multiple_cameras_fall_skips_if_already_extracted(tmp_path: Path):
    mcfd_dir = tmp_path / "multiple_cameras_fall"
    (mcfd_dir / "chute01").mkdir(parents=True)
    (mcfd_dir / "chute01" / "cam1.avi").write_bytes(b"123")

    summary = dl.download_multiple_cameras_fall(mcfd_dir, force=False)
    assert summary.skipped == 1
    assert summary.downloaded == 0
    assert (mcfd_dir / "chute01" / "cam1.avi").read_bytes() == b"123"


def test_download_multiple_cameras_fall_streams_extracts_and_cleans_zip(tmp_path: Path):
    zip_bytes_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_bytes_path, "w") as zf:
        zf.writestr("chute01/cam1.avi", b"cam1-bytes")
        zf.writestr("chute24/cam2.avi", b"cam2-bytes")
    payload = zip_bytes_path.read_bytes()

    mcfd_dir = tmp_path / "multiple_cameras_fall"
    fake_session = MagicMock()
    fake_session.__enter__.return_value = fake_session
    fake_session.__exit__.return_value = False
    fake_session.get.return_value = _FakeResponse(
        status_code=200,
        chunks=[payload],
        headers={"Content-Length": str(len(payload))},
    )

    with patch.object(dl, "build_http_session", return_value=fake_session):
        summary = dl.download_multiple_cameras_fall(mcfd_dir, keep_zip=False)

    assert summary.downloaded == 1
    assert summary.extracted == 2
    assert (mcfd_dir / "chute01" / "cam1.avi").read_bytes() == b"cam1-bytes"
    assert (mcfd_dir / "chute24" / "cam2.avi").read_bytes() == b"cam2-bytes"
    assert not (mcfd_dir / "dataset.zip").exists()


def test_check_kaggle_credentials_supports_env_and_json(tmp_path: Path):
    ok, msg = dl.check_kaggle_credentials(env={}, home=tmp_path)
    assert ok is False
    assert "~/.kaggle/kaggle.json" in msg
    assert "KAGGLE_USERNAME" in msg

    ok_env, msg_env = dl.check_kaggle_credentials(
        env={"KAGGLE_USERNAME": "alice", "KAGGLE_KEY": "super-secret-value"},
        home=tmp_path,
    )
    assert ok_env is True
    assert "super-secret-value" not in msg_env

    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text(
        json.dumps({"username": "alice", "key": "secret-token-xyz"}),
        encoding="utf-8",
    )
    ok_file, msg_file = dl.check_kaggle_credentials(env={}, home=tmp_path)
    assert ok_file is True
    assert "secret-token-xyz" not in msg_file


def test_download_le2i_fall_uses_official_kaggle_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "dummy-key")

    mock_api_instance = MagicMock()
    mock_api_cls = MagicMock(return_value=mock_api_instance)

    fake_kaggle = types.ModuleType("kaggle")
    fake_kaggle_api = types.ModuleType("kaggle.api")
    fake_kaggle_ext = types.ModuleType("kaggle.api.kaggle_api_extended")
    fake_kaggle_ext.KaggleApi = mock_api_cls
    monkeypatch.setitem(sys.modules, "kaggle", fake_kaggle)
    monkeypatch.setitem(sys.modules, "kaggle.api", fake_kaggle_api)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", fake_kaggle_ext)

    dest = tmp_path / "imvia_fall"
    summary = dl.download_le2i_fall(dest)

    assert summary.downloaded == 1
    mock_api_instance.authenticate.assert_called_once()
    mock_api_instance.dataset_download_files.assert_called_once_with(
        dl.LE2I_KAGGLE_DATASET,
        path=str(dest),
        unzip=True,
        force=False,
        quiet=False,
    )


def test_download_urfd_dry_run_and_download(tmp_path: Path):
    fake_session = MagicMock()
    fake_session.__enter__.return_value = fake_session
    fake_session.__exit__.return_value = False

    def _fake_get(url: str, **kwargs):
        if url == dl.URFD_PAGE_URL:
            return _FakeResponse(status_code=200, text=SAMPLE_URFD_HTML)
        return _FakeResponse(
            status_code=200,
            chunks=[b"data-payload"],
            headers={"Content-Length": "12"},
        )

    fake_session.get.side_effect = _fake_get

    urfd_dir = tmp_path / "urfd"
    with patch.object(dl, "build_http_session", return_value=fake_session):
        dry = dl.download_urfd(urfd_dir, types=["video"], dry_run=True)
        assert dry.discovered == 3
        assert dry.downloaded == 0

        res = dl.download_urfd(urfd_dir, types=["video", "csv"], workers=2)
        assert res.discovered == 9
        assert res.downloaded == 9
        assert (urfd_dir / "fall" / "fall-01-cam0.mp4").read_bytes() == b"data-payload"
        assert (urfd_dir / "adl" / "adl-01-cam0.mp4").read_bytes() == b"data-payload"
        assert (urfd_dir / "features" / "urfall-cam0-falls.csv").read_bytes() == b"data-payload"


def test_cli_init_only_and_kaggle_help(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert dl.main(["--setup-kaggle-help"]) == 0
    captured = capsys.readouterr()
    assert "~/.kaggle/kaggle.json" in captured.out

    assert dl.main(["--init-only", "--data-dir", str(tmp_path)]) == 0
    assert (tmp_path / "urfd").is_dir()
    assert (tmp_path / "multiple_cameras_fall").is_dir()
    assert (tmp_path / "imvia_fall").is_dir()
