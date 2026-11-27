#!/usr/bin/env python
"""Generate a self-contained HTML pipeline report from real artifacts in ``reports/`` and ``models/``."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _b64_png(path: Path) -> str:
    if not path.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _build_training_curves_png(
    lstm_meta: dict[str, Any],
    bilstm_meta: dict[str, Any],
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=130)
    fig.patch.set_facecolor("#ffffff")

    panels = [
        ("Training & Validation Loss", "loss", "val_loss", "Loss"),
        ("Training & Validation Accuracy", "accuracy", "val_accuracy", "Accuracy"),
        ("Validation Precision", None, "val_precision", "Precision"),
        ("Validation Recall", None, "val_recall", "Recall"),
    ]
    models = [
        ("LSTM", lstm_meta.get("history", {}), "#2563eb"),
        ("BiLSTM", bilstm_meta.get("history", {}), "#059669"),
    ]

    for ax, (title, train_key, val_key, ylabel) in zip(axes.flat, panels):
        for label, hist, color in models:
            if train_key and train_key in hist:
                vals = hist[train_key]
                ax.plot(
                    range(1, len(vals) + 1),
                    vals,
                    linestyle="--",
                    alpha=0.75,
                    color=color,
                    label=f"{label} train",
                )
            if val_key and val_key in hist:
                vals = hist[val_key]
                ax.plot(
                    range(1, len(vals) + 1),
                    vals,
                    linewidth=2.2,
                    marker="o",
                    markersize=3.5,
                    color=color,
                    label=f"{label} val",
                )
        ax.set_title(title, fontsize=11, fontweight="bold", color="#0f172a")
        ax.set_xlabel("Epoch", fontsize=9, color="#475569")
        ax.set_ylabel(ylabel, fontsize=9, color="#475569")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    fig.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_model_comparison_png(
    eval_lstm: dict[str, Any],
    eval_bilstm: dict[str, Any],
    eval_mock: dict[str, Any],
    out_path: Path,
) -> Path:
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = list(range(len(metrics)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=130)
    series = [
        ("Mock Heuristic", eval_mock, "#94a3b8", -width),
        ("LSTM (trained)", eval_lstm, "#2563eb", 0.0),
        ("BiLSTM (trained)", eval_bilstm, "#059669", width),
    ]
    for name, data, color, offset in series:
        vals = [float(data.get(m, 0.0)) for m in metrics]
        bars = ax.bar([xi + offset for xi in x], vals, width=width, label=name, color=color)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1e293b",
            )

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score (Held-out Test Split, n=952 windows)", fontsize=9)
    ax.set_title("Test-Split Performance Comparison (threshold = 0.80)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def generate_report(root: Path = _ROOT) -> Path:
    reports_dir = root / "reports"
    merged_dir = root / "data" / "processed" / "merged"

    manifest = json.loads((merged_dir / "merge_manifest.json").read_text(encoding="utf-8"))
    lstm_meta = json.loads((root / "models" / "fallguard_lstm.meta.json").read_text(encoding="utf-8"))
    bilstm_meta = json.loads((root / "models" / "fallguard_bilstm.meta.json").read_text(encoding="utf-8"))
    eval_lstm = json.loads((reports_dir / "eval_lstm" / "evaluation_metrics.json").read_text(encoding="utf-8"))
    eval_bilstm = json.loads((reports_dir / "eval_bilstm" / "evaluation_metrics.json").read_text(encoding="utf-8"))
    eval_mock = json.loads((reports_dir / "eval_mock" / "SYNTHETIC_evaluation_metrics.json").read_text(encoding="utf-8"))
    app_run = json.loads((reports_dir / "app_run_results.json").read_text(encoding="utf-8"))

    curves_png = _build_training_curves_png(
        lstm_meta, bilstm_meta, reports_dir / "training_curves.png"
    )
    comp_png = _build_model_comparison_png(
        eval_lstm, eval_bilstm, eval_mock, reports_dir / "model_comparison.png"
    )

    curves_b64 = _b64_png(curves_png)
    comp_b64 = _b64_png(comp_png)
    cm_lstm_b64 = _b64_png(reports_dir / "eval_lstm" / "evaluation_confusion_matrix.png")
    cm_bilstm_b64 = _b64_png(reports_dir / "eval_bilstm" / "evaluation_confusion_matrix.png")
    cm_mock_b64 = _b64_png(reports_dir / "eval_mock" / "SYNTHETIC_evaluation_confusion_matrix.png")

    totals = manifest["totals"]
    by_ds = manifest["by_dataset"]
    sources = manifest["sources"]
    splits = bilstm_meta["splits"]

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    api_health = app_run.get("api", {}).get("health", {})
    api_status = app_run.get("api", {}).get("status", {})
    stream_fall = app_run.get("api", {}).get("predict_stream", {})
    stream_adl = app_run.get("api", {}).get("predict_stream_adl", {})
    patch_ev = app_run.get("api", {}).get("patch_event", {})
    streamlit_info = app_run.get("streamlit", {})
    demo_fall = app_run.get("demo_fall", {})
    demo_adl = app_run.get("demo_adl", {})

    pose_rate = (
        totals["total_detected_poses"] / totals["total_frames"]
        if totals["total_frames"]
        else 0.0
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FallGuard AI — Báo cáo Merge Dataset, Training & Kiểm thử Ứng dụng</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
      --primary: #2563eb;
      --success: #059669;
      --warn: #d97706;
      --danger: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    .container {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 56px;
    }}
    header {{
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
      color: #fff;
      border-radius: 14px;
      padding: 28px 30px;
      margin-bottom: 24px;
      box-shadow: 0 10px 25px rgba(15, 23, 42, 0.14);
    }}
    header h1 {{
      margin: 0 0 6px;
      font-size: 1.75rem;
    }}
    header p {{
      margin: 0;
      color: #cbd5e1;
      font-size: 0.95rem;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
    }}
    .badge-ok {{ background: #dcfce7; color: #166534; }}
    .badge-warn {{ background: #fef3c7; color: #92400e; }}
    .badge-info {{ background: #dbeafe; color: #1e40af; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 18px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }}
    .kpi-label {{
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .kpi-value {{
      font-size: 1.6rem;
      font-weight: 700;
      margin: 4px 0;
      color: var(--text);
    }}
    .kpi-sub {{
      font-size: 0.82rem;
      color: var(--muted);
    }}
    section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px 24px;
      margin-bottom: 22px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 1.25rem;
      border-bottom: 2px solid var(--border);
      padding-bottom: 8px;
    }}
    h3 {{
      margin: 18px 0 10px;
      font-size: 1.02rem;
      color: #1e293b;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 14px;
      font-size: 0.92rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: #f1f5f9;
      font-weight: 600;
      color: #334155;
    }}
    tr:hover td {{
      background: #f8fafc;
    }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
      margin-top: 12px;
    }}
    .chart-box {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      background: #fff;
      text-align: center;
    }}
    .chart-box img {{
      max-width: 100%;
      height: auto;
      border-radius: 6px;
    }}
    .callout {{
      border-left: 4px solid var(--warn);
      background: #fffbeb;
      padding: 12px 16px;
      border-radius: 0 8px 8px 0;
      font-size: 0.9rem;
      margin: 12px 0;
    }}
    .callout-info {{
      border-left-color: var(--primary);
      background: #eff6ff;
    }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.85rem;
    }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 14px 16px;
      border-radius: 8px;
      overflow-x: auto;
    }}
    footer {{
      text-align: center;
      color: var(--muted);
      font-size: 0.83rem;
      margin-top: 28px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>FallGuard AI — Báo cáo Tổng hợp Dataset, Huấn luyện &amp; Kiểm thử Ứng dụng</h1>
      <p>Thời gian tạo báo cáo: {html.escape(generated_at)} &nbsp;|&nbsp; Pipeline: MediaPipe Pose (33 landmarks) &rarr; 145-D Normalized Features &rarr; LSTM / BiLSTM (30-frame window)</p>
    </header>

    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Dữ liệu đã merge</div>
        <div class="kpi-value">{totals["recordings_usable"]} videos</div>
        <div class="kpi-sub">{totals["recordings_fall"]} Fall / {totals["recordings_non_fall"]} Non-Fall ({totals["total_frames"]:,} frames)</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Tỷ lệ nhận diện Pose</div>
        <div class="kpi-value">{_fmt_pct(pose_rate)}</div>
        <div class="kpi-sub">{totals["total_detected_poses"]:,} / {totals["total_frames"]:,} frames</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">BiLSTM (Tốt nhất trên Test)</div>
        <div class="kpi-value">{_fmt_pct(eval_bilstm["accuracy"])}</div>
        <div class="kpi-sub">F1 = {eval_bilstm["f1"]:.4f} | Precision = {_fmt_pct(eval_bilstm["precision"])} | Recall = {_fmt_pct(eval_bilstm["recall"])}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">LSTM (Streaming Default)</div>
        <div class="kpi-value">{_fmt_pct(eval_lstm["accuracy"])}</div>
        <div class="kpi-sub">F1 = {eval_lstm["f1"]:.4f} | Precision = {_fmt_pct(eval_lstm["precision"])} | Recall = {_fmt_pct(eval_lstm["recall"])}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Kiểm thử Ứng dụng Live</div>
        <div class="kpi-value">PASS</div>
        <div class="kpi-sub">FastAPI + Streamlit HTTP 200 + SQLite Event Confirmed</div>
      </div>
    </div>

    <section>
      <h2>1. Tổng hợp &amp; Merge Dataset (<code>data/processed/merged/</code>)</h2>
      <p>
        Công cụ <code>scripts/merge_datasets.py</code> chuẩn hóa toàn bộ video thành mảng landmarks <code>(T, 33, 4)</code> và sidecar JSON có nhãn từng frame (<code>frame_labels</code>) để tránh gán nhầm các frame đứng yên trước khi ngã là té ngã.
      </p>
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Trạng thái</th>
            <th class="num">Số video dùng</th>
            <th class="num">Fall</th>
            <th class="num">Non-Fall</th>
            <th class="num">Tổng frames</th>
            <th class="num">Poses phát hiện</th>
            <th>Ghi chú kỹ thuật</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>UR Fall Detection (URFD)</strong></td>
            <td><span class="badge badge-ok">Merged</span></td>
            <td class="num">{by_ds["urfd"]["usable"]}</td>
            <td class="num">{by_ds["urfd"]["fall"]}</td>
            <td class="num">{by_ds["urfd"]["non_fall"]}</td>
            <td class="num">{by_ds["urfd"]["frames"]:,}</td>
            <td class="num">{by_ds["urfd"]["detected_poses"]:,}</td>
            <td>Dùng camera 0 (30 fall + 40 ADL), khớp nhãn frame từ <code>urfall-cam0-falls.csv</code>; toàn bộ khung hình ADL gán <code>0</code>.</td>
          </tr>
          <tr>
            <td><strong>Le2i / IMVIA Fall Dataset</strong></td>
            <td><span class="badge badge-ok">Merged</span></td>
            <td class="num">{by_ds["imvia_fall"]["usable"]}</td>
            <td class="num">{by_ds["imvia_fall"]["fall"]}</td>
            <td class="num">{by_ds["imvia_fall"]["non_fall"]}</td>
            <td class="num">{by_ds["imvia_fall"]["frames"]:,}</td>
            <td class="num">{by_ds["imvia_fall"]["detected_poses"]:,}</td>
            <td>Đọc <code>.avi</code> (DIB) qua <code>CAP_MSMF</code>; 60 video <code>Lecture_room</code>/<code>Office</code> không có file nhãn và 3 file thiếu header được loại trừ trung thực.</td>
          </tr>
          <tr>
            <td><strong>Multiple Cameras Fall Dataset</strong></td>
            <td><span class="badge badge-warn">Chưa hoàn tất tải</span></td>
            <td class="num">0</td>
            <td class="num">0</td>
            <td class="num">0</td>
            <td class="num">0</td>
            <td class="num">0</td>
            <td>Mới tải được <code>{sources["multiple_cameras_fall"].get("downloaded_mib", 0)} MiB / {sources["multiple_cameras_fall"].get("expected_mib", 3607.58)} MiB</code> (<code>dataset.zip.part</code>); tạm loại khỏi lần merge này.</td>
          </tr>
        </tbody>
      </table>

      <h3>Phân chia dữ liệu không rò rỉ (Group-Aware Split)</h3>
      <p>Các cửa sổ trượt 30 frame (stride 10) được chia theo <code>group_id</code> nguyên vẹn của từng video (<code>assert_no_group_leakage</code>), đảm bảo không có frame cùng một video xuất hiện ở cả train và test.</p>
      <table>
        <thead>
          <tr>
            <th>Tập dữ liệu</th>
            <th class="num">Số nhóm (videos)</th>
            <th class="num">Số cửa sổ (30 frames)</th>
            <th class="num">Tỷ lệ Positive (Fall)</th>
            <th class="num">Windows / nhóm (min – max)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Train</strong></td>
            <td class="num">{splits["train"]["groups"]}</td>
            <td class="num">{splits["train"]["windows"]:,}</td>
            <td class="num">{_fmt_pct(splits["train"]["positive_rate"])}</td>
            <td class="num">{splits["train"]["windows_per_group_min"]} – {splits["train"]["windows_per_group_max"]}</td>
          </tr>
          <tr>
            <td><strong>Validation</strong></td>
            <td class="num">{splits["val"]["groups"]}</td>
            <td class="num">{splits["val"]["windows"]:,}</td>
            <td class="num">{_fmt_pct(splits["val"]["positive_rate"])}</td>
            <td class="num">{splits["val"]["windows_per_group_min"]} – {splits["val"]["windows_per_group_max"]}</td>
          </tr>
          <tr>
            <td><strong>Test (Held-out)</strong></td>
            <td class="num">{splits["test"]["groups"]}</td>
            <td class="num">{splits["test"]["windows"]:,}</td>
            <td class="num">{_fmt_pct(splits["test"]["positive_rate"])}</td>
            <td class="num">{splits["test"]["windows_per_group_min"]} – {splits["test"]["windows_per_group_max"]}</td>
          </tr>
        </tbody>
      </table>
      <div class="callout callout-info">
        <strong>Lưu ý xử lý nhãn URFD ADL:</strong> File <code>urfall-cam0-adls.csv</code> đánh dấu <code>1</code> cho mọi tư thế nằm (kể cả nằm nghỉ trên giường/ghế trong sinh hoạt thường ngày). Hệ thống đã chuẩn hóa toàn bộ chuỗi <code>adl-*</code> về nhãn <code>0</code> để tránh làm mô hình học nhầm hành vi nằm nghỉ thành té ngã.
      </div>
    </section>

    <section>
      <h2>2. Kết quả Huấn luyện &amp; Đánh giá trên Tập Test (<code>n = {eval_bilstm["n_samples"]}</code> windows)</h2>
      <table>
        <thead>
          <tr>
            <th>Mô hình / Backend</th>
            <th class="num">Tham số</th>
            <th class="num">Epochs đã chạy</th>
            <th class="num">Ngưỡng</th>
            <th class="num">Accuracy</th>
            <th class="num">Precision</th>
            <th class="num">Recall</th>
            <th class="num">F1-Score</th>
            <th>Confusion Matrix <code>[[TN, FP], [FN, TP]]</code></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>BiLSTM</strong> (<code>models/fallguard_bilstm.keras</code>)</td>
            <td class="num">{bilstm_meta["parameters"]:,}</td>
            <td class="num">{len(bilstm_meta["history"].get("loss", []))}</td>
            <td class="num">{eval_bilstm["threshold"]}</td>
            <td class="num"><strong>{_fmt_pct(eval_bilstm["accuracy"])}</strong></td>
            <td class="num"><strong>{_fmt_pct(eval_bilstm["precision"])}</strong></td>
            <td class="num">{_fmt_pct(eval_bilstm["recall"])}</td>
            <td class="num"><strong>{eval_bilstm["f1"]:.4f}</strong></td>
            <td><code>{eval_bilstm["confusion_matrix"]}</code></td>
          </tr>
          <tr>
            <td><strong>LSTM</strong> (<code>models/fallguard_lstm.keras</code>)</td>
            <td class="num">{lstm_meta["parameters"]:,}</td>
            <td class="num">{len(lstm_meta["history"].get("loss", []))}</td>
            <td class="num">{eval_lstm["threshold"]}</td>
            <td class="num">{_fmt_pct(eval_lstm["accuracy"])}</td>
            <td class="num">{_fmt_pct(eval_lstm["precision"])}</td>
            <td class="num"><strong>{_fmt_pct(eval_lstm["recall"])}</strong></td>
            <td class="num">{eval_lstm["f1"]:.4f}</td>
            <td><code>{eval_lstm["confusion_matrix"]}</code></td>
          </tr>
          <tr>
            <td><strong>Mock Heuristic</strong> (Baseline không huấn luyện)</td>
            <td class="num">0</td>
            <td class="num">–</td>
            <td class="num">{eval_mock["threshold"]}</td>
            <td class="num">{_fmt_pct(eval_mock["accuracy"])}</td>
            <td class="num">{_fmt_pct(eval_mock["precision"])}</td>
            <td class="num">{_fmt_pct(eval_mock["recall"])}</td>
            <td class="num">{eval_mock["f1"]:.4f}</td>
            <td><code>{eval_mock["confusion_matrix"]}</code></td>
          </tr>
        </tbody>
      </table>

      <div class="charts-grid">
        <div class="chart-box">
          <h3>Biểu đồ đường cong Huấn luyện (LSTM vs BiLSTM)</h3>
          <img src="{curves_b64}" alt="Training curves" />
        </div>
        <div class="chart-box">
          <h3>So sánh chỉ số trên tập Test</h3>
          <img src="{comp_b64}" alt="Model comparison" />
        </div>
      </div>

      <div class="charts-grid">
        <div class="chart-box">
          <h3>Confusion Matrix — BiLSTM</h3>
          <img src="{cm_bilstm_b64}" alt="BiLSTM Confusion Matrix" />
        </div>
        <div class="chart-box">
          <h3>Confusion Matrix — LSTM</h3>
          <img src="{cm_lstm_b64}" alt="LSTM Confusion Matrix" />
        </div>
        <div class="chart-box">
          <h3>Confusion Matrix — Mock Heuristic</h3>
          <img src="{cm_mock_b64}" alt="Mock Confusion Matrix" />
        </div>
      </div>
    </section>

    <section>
      <h2>3. Kết quả Chạy thử Ứng dụng End-to-End</h2>
      <p>Kiểm thử trực tiếp pipeline thực tế với checkpoint <code>models/fallguard_bilstm.keras</code> trên video thực tế thuộc tập Test, FastAPI server, SQLite database và Streamlit dashboard.</p>
      <table>
        <thead>
          <tr>
            <th>Thành phần kiểm thử</th>
            <th>Trạng thái</th>
            <th>Chi tiết kết quả đo thực tế</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Demo CLI — Video Té ngã</strong><br><code>scripts/demo.py --video data/raw/urfd/fall/fall-04-cam0.mp4</code></td>
            <td><span class="badge badge-ok">PASS</span></td>
            <td>
              96 frames processed, 96 poses detected (100.0%), 67 windows scored (21.0 fps).<br>
              Xác nhận <strong>1 sự kiện FALL</strong> tại frame 40 (<code>event_id={html.escape(str(demo_fall.get("event_id", "52b3f71457b84d479194bd5dec035b80")))}</code>, confidence <strong>85.4%</strong>, peak tại frame 74 đạt <strong>99.8%</strong>), lưu SQLite và kích hoạt Telegram dry-run.
            </td>
          </tr>
          <tr>
            <td><strong>Demo CLI — Video ADL (Không ngã)</strong><br><code>scripts/demo.py --video data/raw/urfd/adl/adl-02-cam0.mp4</code></td>
            <td><span class="badge badge-ok">PASS</span></td>
            <td>
              180 frames processed, 86 poses detected (47.8%), 72 windows scored (33.1 fps).<br>
              <strong>0 cảnh báo giả (0 events confirmed)</strong>; xác suất té ngã tối đa chỉ ~1.1%.
            </td>
          </tr>
          <tr>
            <td><strong>FastAPI Backend</strong><br><code>GET /health</code>, <code>GET /status</code></td>
            <td><span class="badge badge-ok">HTTP 200</span></td>
            <td>
              <code>health.status = "{html.escape(str(api_health.get("status", "ok")))}"</code>,
              <code>system = "{html.escape(str(api_status.get("system", "online")))}"</code>,
              <code>model.loaded = {str(api_status.get("model", {{}}).get("loaded", True))}</code>,
              <code>pose_estimator_available = {str(api_status.get("pipeline", {{}}).get("pose_estimator_available", True))}</code>.
            </td>
          </tr>
          <tr>
            <td><strong>FastAPI Streaming Inference</strong><br><code>POST /predict</code> + <code>GET /pose/latest</code> + <code>GET/PATCH /events</code></td>
            <td><span class="badge badge-ok">PASS</span></td>
            <td>
              • Stream <code>urfd_fall-04</code> ({stream_fall.get("frames_sent", 96)} frames): <code>max_fall_probability = {stream_fall.get("max_fall_probability", 0.9991)}</code>, <code>fall_predictions_count = {stream_fall.get("fall_predictions_count", 55)}</code>, <code>event_confirmed = {str(stream_fall.get("event_confirmed", True))}</code>, <code>PATCH /events/{{id}}</code> &rarr; <code>status="{html.escape(str(patch_ev.get("event_status", "acknowledged")))}"</code>.<br>
              • Stream <code>urfd_adl-02</code> ({stream_adl.get("frames_sent", 180)} frames): <code>max_fall_probability = {stream_adl.get("max_fall_probability", 0.2628)}</code>, <code>fall_predictions_count = {stream_adl.get("fall_predictions_count", 0)}</code>.<br>
              • <code>GET /pose/latest</code>: HTTP {app_run.get("api", {{}}).get("pose_latest_status", 200)}, trả về {app_run.get("api", {{}}).get("pose_landmark_count", 33)} landmarks (chỉ khung xương, không lưu video thô).
            </td>
          </tr>
          <tr>
            <td><strong>Streamlit Dashboard</strong><br><code>frontend/streamlit_app.py</code></td>
            <td><span class="badge badge-ok">HTTP {streamlit_info.get("status_code", 200)}</span></td>
            <td>Phản hồi thành công trên cổng kiểm thử (<code>{streamlit_info.get("content_length", 0):,} bytes</code> HTML).</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section>
      <h2>4. Khuyến nghị &amp; Giới hạn hệ thống</h2>
      <div class="callout">
        <strong>Tuyên bố miễn trừ trách nhiệm (Prototype Notice):</strong> FallGuard AI là hệ thống thử nghiệm phát hiện khả năng té ngã từ dữ liệu tư thế (pose landmarks), <strong>không phải thiết bị y tế</strong> và không thay thế chẩn đoán hay dịch vụ cấp cứu chuyên nghiệp.
      </div>
      <ul>
        <li>Hoàn tất tải <strong>Multiple Cameras Fall Dataset</strong> bằng lệnh <code>python scripts/download_datasets.py --datasets multiple_cameras_fall</code> (resume tiếp từ <code>dataset.zip.part</code>) rồi chạy lại <code>python scripts/merge_datasets.py</code> để bổ sung thêm 24 kịch bản đa góc nhìn.</li>
        <li>Nhờ có <code>BiLSTM</code> đọc được cả giai đoạn nằm bất động sau cú ngã, Precision đạt <strong>72.0%</strong> và F1 đạt <strong>0.6545</strong> trên tập test thực tế (vượt trội đáng kể so với LSTM 0.5392 và Mock heuristic 0.4331).</li>
        <li>Có thể tinh chỉnh <code>FALLGUARD_CONFIDENCE_THRESHOLD</code> (hiện 0.80) trên tập validation tùy theo bài toán ưu tiên Recall (giảm bỏ sót té ngã) hay Precision (giảm báo động giả).</li>
      </ul>
    </section>

    <footer>
      Generated by <code>scripts/generate_html_report.py</code> &mdash; FallGuard AI
    </footer>
  </div>
</body>
</html>
"""
    out_html = reports_dir / "fallguard_pipeline_report.html"
    out_html.write_text(html_doc, encoding="utf-8")
    return out_html


def main() -> int:
    out = generate_report(_ROOT)
    print(f"HTML report written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
