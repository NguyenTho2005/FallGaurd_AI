# FallGuard AI — Data Directory

This directory holds datasets, processed landmark arrays, synthetic samples, and the runtime SQLite database.

> **Do not commit raw videos or large binary artifacts.** `data/raw/*`, `data/processed/*`, `data/samples/*`, and `*.db` are git-ignored.

---

## Directory Structure

```text
data/
├── raw/
│   ├── urfd/                     # UR Fall Detection Dataset
│   ├── multiple_cameras_fall/    # Multiple Cameras Fall Dataset
│   └── imvia_fall/               # Le2i / IMVIA Fall Dataset (via Kaggle API)
├── processed/                    # Extracted (T, 33, 4) pose landmarks + JSON sidecars
├── samples/                      # Local demo/sample clips
├── fallguard.db                  # Runtime SQLite event database
└── README.md                     # This file
```

---

## Raw Datasets Managed by `scripts/download_datasets.py`

### 1. UR Fall Detection Dataset (`data/raw/urfd/`)

- **Source:** https://fenix.ur.edu.pl/~mkepski/ds/uf.html
- **Contents:** 70 sequences (30 falls recorded with 2 Microsoft Kinect cameras + accelerometer; 40 Activities of Daily Living recorded with 1 Kinect camera + accelerometer) plus extracted feature CSVs.
- **License:** CC BY-NC-SA 4.0 (non-commercial academic use).
- **How it is downloaded:** `scripts/download_datasets.py` dynamically fetches and parses the HTML index page to discover verified links (no hardcoded file URLs). Downloaded files are grouped under `fall/`, `adl/`, and `features/` while preserving original filenames.
- **Selectable file types (`--urfd-types`):**
  - `video` — MP4 video sequences (`*-cam0.mp4`, `*-cam1.mp4`)
  - `csv` — synchronization, accelerometer, and extracted feature CSVs
  - `rgb` — RGB PNG sequence archives (`*-rgb.zip`)
  - `depth` — 16-bit depth PNG sequence archives (`*-d.zip`)
  - `all` — all discovered dataset files

```bash
# Default (videos + CSVs)
python scripts/download_datasets.py --datasets urfd

# Include RGB image zip archives and extract them
python scripts/download_datasets.py --datasets urfd --urfd-types video rgb csv --extract-urfd

# Download every file listed on the URFD page (including depth maps)
python scripts/download_datasets.py --datasets urfd --urfd-types all
```

---

### 2. Multiple Cameras Fall Dataset (`data/raw/multiple_cameras_fall/`)

- **Source:** https://www.iro.umontreal.ca/~labimage/Dataset/dataset.zip (~3.52 GiB)
- **Contents:**
  - **24 scenarios** recorded simultaneously by **8 IP cameras**.
  - **Scenarios 1–22:** contain fall events and confounding events.
  - **Scenarios 23–24:** contain confounding events only (no falls).
- **How it is downloaded:** Streams `dataset.zip` with resume support and extracts it safely into `data/raw/multiple_cameras_fall/`, preserving the original directory structure and filenames.

```bash
python scripts/download_datasets.py --datasets multiple_cameras_fall

# Retain dataset.zip after extraction
python scripts/download_datasets.py --datasets multiple_cameras_fall --keep-zip
```

---

### 3. Le2i / IMVIA Fall Dataset (`data/raw/imvia_fall/`)

- **Source:** https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
- **Kaggle Slug:** `tuyenldvn/falldataset-imvia`
- **How it is downloaded:** Uses the official `kaggle` Python API (`KaggleApi`). Credentials are **never** hardcoded in the repository.

#### Configuring Kaggle API Credentials

1. Install the official Kaggle package if not already installed:

   ```bash
   pip install kaggle
   ```

2. Obtain an API token from Kaggle:
   - Log in to [kaggle.com](https://www.kaggle.com) -> **Settings** -> **API** -> **Create New Token** (downloads `kaggle.json`).

3. Configure credentials using **one** of the following methods:

   - **Option A — `kaggle.json` file (Recommended):**
     - **Linux / macOS:** save to `~/.kaggle/kaggle.json` and restrict permissions:
       ```bash
       mkdir -p ~/.kaggle
       mv /path/to/kaggle.json ~/.kaggle/kaggle.json
       chmod 600 ~/.kaggle/kaggle.json
       ```
     - **Windows:** save to `%USERPROFILE%\.kaggle\kaggle.json`.
     - Format:
       ```json
       {"username": "<your_kaggle_username>", "key": "<your_api_key>"}
       ```
     - Alternatively, set `KAGGLE_CONFIG_DIR` to the folder containing `kaggle.json`.

   - **Option B — Environment variables:**
     - **Linux / macOS:**
       ```bash
       export KAGGLE_USERNAME="<your_kaggle_username>"
       export KAGGLE_KEY="<your_api_key>"
       ```
     - **Windows PowerShell:**
       ```powershell
       $env:KAGGLE_USERNAME="<your_kaggle_username>"
       $env:KAGGLE_KEY="<your_api_key>"
       ```

4. Download the dataset:

   ```bash
   python scripts/download_datasets.py --datasets imvia_fall
   ```

---

## Dry-Run & Verification

To verify the directory layout, parse URFD links, inspect the MCFD archive headers, and validate Kaggle credential configuration without downloading large files:

```bash
python scripts/download_datasets.py --dry-run
```

Existing files in `data/raw/`, `data/processed/`, and `data/samples/` are never overwritten or moved unless `--force` is explicitly provided.
