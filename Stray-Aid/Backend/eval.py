"""
StrayAid Platform — Performance Evaluation Script
==================================================
Measures and plots:
  1. End-to-end API latency  (animal classifier + disease model + Gemini)
  2. Response rate            (success vs failure)
  3. Throughput               (requests/min under load)
  4. Model accuracy           (per-class: cat / dog / cow)

Requirements:
    pip install requests matplotlib numpy pillow tqdm

Usage:
    python evaluation.py --host http://127.0.0.1:5000
                         --images_dir ./test_images
                         --runs 50
                         --concurrency 5
                         --save_plots          (optional — saves PNGs instead of showing)
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# ── Matplotlib style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "#f9fafb",
    "axes.facecolor": "#f9fafb",
})

COLORS = {
    "animal": "#378ADD",
    "disease": "#1D9E75",
    "gemini": "#D85A30",
    "success": "#639922",
    "failure": "#E24B4A",
    "p50": "#378ADD",
    "p95": "#D4537E",
    "p99": "#E24B4A",
    "cat": "#378ADD",
    "dog": "#1D9E75",
    "cow": "#BA7517",
}

PREDICT_ENDPOINT = "/predict"
HEALTH_ENDPOINT  = "/"          # used only for connectivity check


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def discover_images(images_dir: str, limit: int = 200):
    """Return list of (path, true_label) from a directory structured as:
        images_dir/
            cat/   img1.jpg ...
            dog/   img1.jpg ...
            cow/   img1.jpg ...
    If the directory is flat (no sub-folders), label is set to None.
    """
    p = Path(images_dir)
    if not p.exists():
        print(f"[WARN] images_dir '{images_dir}' not found — using synthetic dummy images.")
        return []

    items = []
    for cls in ["cat", "dog", "cow"]:
        sub = p / cls
        if sub.is_dir():
            for img in list(sub.glob("*"))[:limit // 3]:
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    items.append((str(img), cls))
    if not items:
        for img in list(p.glob("*"))[:limit]:
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                items.append((str(img), None))
    return items


def make_dummy_image() -> bytes:
    """Return a tiny valid JPEG in memory (used when no real images are available)."""
    import io
    img = Image.new("RGB", (224, 224), color=(120, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def single_request(host: str, img_path: str | None, true_label: str | None):
    """Send one /predict request and return a result dict."""
    url = host.rstrip("/") + PREDICT_ENDPOINT
    t_start = time.perf_counter()

    try:
        if img_path:
            with open(img_path, "rb") as f:
                files = {"file": (Path(img_path).name, f, "image/jpeg")}
                resp = requests.post(url, files=files, timeout=30)
        else:
            import io
            dummy = make_dummy_image()
            files = {"file": ("dummy.jpg", io.BytesIO(dummy), "image/jpeg")}
            resp = requests.post(url, files=files, timeout=30)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        status_code = resp.status_code

        if resp.ok:
            data = resp.json()
            predicted = data.get("animal", "unknown").lower()
            disease_info = data.get("disease") or {}
            return {
                "ok": True,
                "status_code": status_code,
                "latency_ms": round(elapsed_ms, 1),
                "predicted": predicted,
                "true_label": true_label,
                "disease": disease_info.get("disease", "—"),
                "is_healthy": disease_info.get("is_healthy", None),
                "animal_confidence": data.get("animal_confidence", 0),
            }
        else:
            return {
                "ok": False,
                "status_code": status_code,
                "latency_ms": round(elapsed_ms, 1),
                "predicted": None,
                "true_label": true_label,
                "error": resp.text[:200],
            }

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return {
            "ok": False,
            "status_code": 0,
            "latency_ms": round(elapsed_ms, 1),
            "predicted": None,
            "true_label": true_label,
            "error": str(exc),
        }


def run_sequential(host: str, items: list, desc="Sequential run"):
    results = []
    for img_path, true_label in tqdm(items, desc=desc):
        results.append(single_request(host, img_path, true_label))
    return results


def run_concurrent(host: str, items: list, concurrency: int = 5, desc="Concurrent run"):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(single_request, host, p, lbl): (p, lbl) for p, lbl in items}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            results.append(fut.result())
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list):
    latencies = [r["latency_ms"] for r in results]
    ok_results = [r for r in results if r["ok"]]
    fail_results = [r for r in results if not r["ok"]]

    total = len(results)
    n_ok = len(ok_results)
    n_fail = len(fail_results)

    status_counts = {}
    for r in results:
        sc = r["status_code"]
        status_counts[sc] = status_counts.get(sc, 0) + 1

    # Accuracy (only when true_label is available)
    labeled = [r for r in ok_results if r.get("true_label")]
    correct = [r for r in labeled if r["predicted"] == r["true_label"]]
    accuracy = len(correct) / len(labeled) * 100 if labeled else None

    per_class = {}
    for cls in ["cat", "dog", "cow"]:
        cls_items = [r for r in labeled if r["true_label"] == cls]
        cls_correct = [r for r in cls_items if r["predicted"] == cls]
        if cls_items:
            per_class[cls] = round(len(cls_correct) / len(cls_items) * 100, 1)

    lat_arr = np.array(latencies)
    return {
        "total": total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "success_rate_pct": round(n_ok / total * 100, 2) if total else 0,
        "latencies": latencies,
        "lat_mean": round(float(np.mean(lat_arr)), 1),
        "lat_median": round(float(np.median(lat_arr)), 1),
        "lat_p95": round(float(np.percentile(lat_arr, 95)), 1),
        "lat_p99": round(float(np.percentile(lat_arr, 99)), 1),
        "lat_min": round(float(np.min(lat_arr)), 1),
        "lat_max": round(float(np.max(lat_arr)), 1),
        "status_counts": status_counts,
        "accuracy_pct": round(accuracy, 2) if accuracy is not None else None,
        "per_class_accuracy": per_class,
    }


def throughput_test(host: str, duration_sec: int = 30, concurrency: int = 5):
    """Fire requests for `duration_sec` seconds and record timestamps."""
    print(f"\n[Throughput] Running {duration_sec}s load test (concurrency={concurrency})…")
    timestamps = []
    stop_at = time.time() + duration_sec

    def worker():
        while time.time() < stop_at:
            t = time.perf_counter()
            single_request(host, None, None)
            timestamps.append(time.perf_counter() - t)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker) for _ in range(concurrency)]
        concurrent.futures.wait(futs)

    rpm = round(len(timestamps) / duration_sec * 60, 1)
    print(f"[Throughput] {len(timestamps)} requests in {duration_sec}s → {rpm} RPM")
    return rpm, len(timestamps)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all(metrics: dict, rpm: float, total_reqs: int, save_plots: bool):
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    fig.suptitle("StrayAid — Performance Evaluation Dashboard", fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 3.1 Latency histogram ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Latency distribution", fontsize=11, fontweight="bold")
    lats = metrics["latencies"]
    ax1.hist(lats, bins=30, color=COLORS["animal"], alpha=0.8, edgecolor="white", linewidth=0.5)
    for pct, val, col in [("P50", metrics["lat_median"], COLORS["p50"]),
                           ("P95", metrics["lat_p95"], COLORS["p95"]),
                           ("P99", metrics["lat_p99"], COLORS["p99"])]:
        ax1.axvline(val, color=col, linestyle="--", linewidth=1.5, label=f"{pct}: {val}ms")
    ax1.set_xlabel("Latency (ms)", fontsize=9)
    ax1.set_ylabel("Count", fontsize=9)
    ax1.legend(fontsize=8)

    # ── 3.2 Latency over time ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Latency over requests", fontsize=11, fontweight="bold")
    ax2.plot(lats, color=COLORS["animal"], linewidth=1, alpha=0.7)
    window = max(1, len(lats) // 10)
    rolling = np.convolve(lats, np.ones(window) / window, mode="valid")
    ax2.plot(range(window - 1, len(lats)), rolling, color=COLORS["p99"], linewidth=2, label=f"Rolling avg ({window})")
    ax2.set_xlabel("Request #", fontsize=9)
    ax2.set_ylabel("ms", fontsize=9)
    ax2.legend(fontsize=8)

    # ── 3.3 P50/P95/P99 bar ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Latency percentiles", fontsize=11, fontweight="bold")
    pct_labels = ["P50 (median)", "P95", "P99", "Mean", "Max"]
    pct_values = [metrics["lat_median"], metrics["lat_p95"], metrics["lat_p99"],
                  metrics["lat_mean"], metrics["lat_max"]]
    bar_colors = [COLORS["p50"], COLORS["p95"], COLORS["p99"], COLORS["gemini"], "#888"]
    bars = ax3.barh(pct_labels, pct_values, color=bar_colors, alpha=0.85, height=0.55)
    for bar, val in zip(bars, pct_values):
        ax3.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f"{val}ms", va="center", fontsize=8, color="#333")
    ax3.set_xlabel("ms", fontsize=9)

    # ── 3.4 Success / Failure pie ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title("Response outcome", fontsize=11, fontweight="bold")
    labels = ["Success", "Failure"]
    sizes  = [metrics["n_ok"], max(metrics["n_fail"], 0)]
    if sum(sizes) == 0:
        sizes = [1, 0]
    colors = [COLORS["success"], COLORS["failure"]]
    wedges, texts, autotexts = ax4.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(9)

    # ── 3.5 HTTP status codes bar ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title("HTTP status codes", fontsize=11, fontweight="bold")
    sc = metrics["status_counts"]
    status_labels = [str(k) for k in sorted(sc.keys())]
    status_vals   = [sc[int(k)] for k in status_labels]
    sc_colors = []
    for k in status_labels:
        k = int(k)
        if 200 <= k < 300:
            sc_colors.append(COLORS["success"])
        elif 400 <= k < 500:
            sc_colors.append("#BA7517")
        elif k >= 500:
            sc_colors.append(COLORS["failure"])
        else:
            sc_colors.append("#888")
    ax5.bar(status_labels, status_vals, color=sc_colors, alpha=0.85, edgecolor="white")
    ax5.set_xlabel("Status code", fontsize=9)
    ax5.set_ylabel("Count", fontsize=9)
    for i, v in enumerate(status_vals):
        ax5.text(i, v + 0.3, str(v), ha="center", fontsize=9)

    # ── 3.6 Throughput gauge ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title("Throughput summary", fontsize=11, fontweight="bold")
    ax6.axis("off")
    stats_text = (
        f"Total requests : {metrics['total']}\n"
        f"Successful     : {metrics['n_ok']}\n"
        f"Failed         : {metrics['n_fail']}\n"
        f"Success rate   : {metrics['success_rate_pct']}%\n\n"
        f"Throughput     : {rpm} RPM\n"
        f"Total fired    : {total_reqs}\n\n"
        f"Min latency    : {metrics['lat_min']}ms\n"
        f"Mean latency   : {metrics['lat_mean']}ms\n"
        f"Max latency    : {metrics['lat_max']}ms"
    )
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=10, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#e8f5e9", ec="#1D9E75", lw=1.2))

    # ── 3.7 Per-class accuracy ───────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_title("Per-class accuracy", fontsize=11, fontweight="bold")
    per_cls = metrics.get("per_class_accuracy", {})
    if per_cls:
        cls_names = list(per_cls.keys())
        cls_vals  = list(per_cls.values())
        cls_cols  = [COLORS.get(c, "#888") for c in cls_names]
        bars = ax7.bar(cls_names, cls_vals, color=cls_cols, alpha=0.85, edgecolor="white", width=0.5)
        ax7.set_ylim(0, 110)
        ax7.set_ylabel("Accuracy %", fontsize=9)
        ax7.axhline(90, color="#888", linestyle="--", linewidth=1, alpha=0.5, label="90% target")
        ax7.legend(fontsize=8)
        for bar, val in zip(bars, cls_vals):
            ax7.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                     f"{val}%", ha="center", fontsize=9, fontweight="bold")
    else:
        ax7.text(0.5, 0.5, "No labeled test images\nfound for accuracy calc.",
                 ha="center", va="center", transform=ax7.transAxes,
                 fontsize=10, color="#888")
        ax7.axis("off")

    # ── 3.8 Overall accuracy ─────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_title("Overall model accuracy", fontsize=11, fontweight="bold")
    overall = metrics.get("accuracy_pct")
    if overall is not None:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax8.axis("equal")
        ax8.set_xlim(-1.3, 1.3)
        ax8.set_ylim(-1.3, 1.3)
        ax8.axis("off")
        circle_bg = plt.Circle((0, 0), 1, color="#e8f0fe", linewidth=0)
        ax8.add_patch(circle_bg)
        fraction = overall / 100
        end_angle = np.pi / 2 - fraction * 2 * np.pi
        angles = np.linspace(np.pi / 2, end_angle, 200)
        ax8.fill_between(np.cos(angles) * [0.7, 1], np.sin(angles) * [0.7, 1],
                         alpha=0.0)
        arc_x = np.cos(angles)
        arc_y = np.sin(angles)
        ax8.plot(arc_x, arc_y, color=COLORS["success"], linewidth=14, solid_capstyle="round")
        ax8.plot(np.cos(theta), np.sin(theta), color="#ddd", linewidth=3)
        ax8.text(0, 0.05, f"{overall}%", ha="center", va="center",
                 fontsize=22, fontweight="bold", color=COLORS["success"])
        ax8.text(0, -0.3, "Accuracy", ha="center", va="center", fontsize=10, color="#555")
    else:
        ax8.text(0.5, 0.5, "No labels available",
                 ha="center", va="center", transform=ax8.transAxes, fontsize=10, color="#888")
        ax8.axis("off")

    # ── 3.9 Latency CDF ─────────────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_title("Latency CDF", fontsize=11, fontweight="bold")
    sorted_lats = np.sort(lats)
    cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats) * 100
    ax9.plot(sorted_lats, cdf, color=COLORS["animal"], linewidth=2)
    ax9.axhline(50, color=COLORS["p50"], linestyle="--", linewidth=1, alpha=0.6, label="P50")
    ax9.axhline(95, color=COLORS["p95"], linestyle="--", linewidth=1, alpha=0.6, label="P95")
    ax9.axhline(99, color=COLORS["p99"], linestyle="--", linewidth=1, alpha=0.6, label="P99")
    ax9.set_xlabel("Latency (ms)", fontsize=9)
    ax9.set_ylabel("Cumulative %", fontsize=9)
    ax9.legend(fontsize=8)

    plt.suptitle("StrayAid — Performance Evaluation Dashboard", fontsize=14, fontweight="bold")

    if save_plots:
        out = "evaluation_report.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n[OK] Plot saved → {out}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(metrics: dict, rpm: float):
    sep = "═" * 52
    print(f"\n{sep}")
    print("  STRAYAID — PERFORMANCE EVALUATION REPORT")
    print(sep)
    print(f"  Total requests    : {metrics['total']}")
    print(f"  Successful        : {metrics['n_ok']}  ({metrics['success_rate_pct']}%)")
    print(f"  Failed            : {metrics['n_fail']}")
    print(f"  Throughput (est.) : {rpm} RPM")
    print(sep)
    print("  LATENCY (ms)")
    print(f"    Min    : {metrics['lat_min']}")
    print(f"    Mean   : {metrics['lat_mean']}")
    print(f"    Median : {metrics['lat_median']}")
    print(f"    P95    : {metrics['lat_p95']}")
    print(f"    P99    : {metrics['lat_p99']}")
    print(f"    Max    : {metrics['lat_max']}")
    print(sep)
    print("  HTTP STATUS CODES")
    for code, cnt in sorted(metrics["status_counts"].items()):
        print(f"    {code}  →  {cnt} requests")
    if metrics["accuracy_pct"] is not None:
        print(sep)
        print("  ACCURACY")
        print(f"    Overall : {metrics['accuracy_pct']}%")
        for cls, acc in metrics["per_class_accuracy"].items():
            print(f"    {cls.capitalize():<6}  : {acc}%")
    print(sep)

    report = {
        "summary": {
            "total": metrics["total"],
            "success_rate_pct": metrics["success_rate_pct"],
            "throughput_rpm": rpm,
        },
        "latency_ms": {
            "min": metrics["lat_min"],
            "mean": metrics["lat_mean"],
            "median": metrics["lat_median"],
            "p95": metrics["lat_p95"],
            "p99": metrics["lat_p99"],
            "max": metrics["lat_max"],
        },
        "status_codes": {str(k): v for k, v in metrics["status_counts"].items()},
        "accuracy": {
            "overall_pct": metrics["accuracy_pct"],
            "per_class": metrics["per_class_accuracy"],
        },
    }
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  JSON report saved → evaluation_report.json")
    print(sep + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="StrayAid performance evaluation")
    parser.add_argument("--host",        default="http://127.0.0.1:5000", help="Flask server URL")
    parser.add_argument("--images_dir",  default="./test_images",          help="Directory of test images")
    parser.add_argument("--runs",        type=int, default=20,             help="Number of sequential requests")
    parser.add_argument("--concurrency", type=int, default=4,              help="Threads for concurrent test")
    parser.add_argument("--throughput_sec", type=int, default=20,          help="Duration of throughput test (s)")
    parser.add_argument("--save_plots",  action="store_true",              help="Save plots as PNG instead of showing")
    args = parser.parse_args()

    # ── Connectivity check ────────────────────────────────────────────────────
    print(f"\n[INFO] Connecting to {args.host} …")
    try:
        r = requests.get(args.host + HEALTH_ENDPOINT, timeout=5)
        print(f"[OK]  Server responded with HTTP {r.status_code}")
    except Exception as e:
        print(f"[ERROR] Cannot reach server: {e}")
        print("        Make sure your Flask app is running: python app.py")
        sys.exit(1)

    # ── Discover images ───────────────────────────────────────────────────────
    items = discover_images(args.images_dir, limit=args.runs)
    if not items:
        print(f"[WARN] No images found in '{args.images_dir}'. Using {args.runs} dummy images.")
        items = [(None, None)] * args.runs

    items = items[: args.runs]
    print(f"[INFO] Test set: {len(items)} images")

    # ── Sequential evaluation ─────────────────────────────────────────────────
    print("\n[1/3] Sequential evaluation …")
    seq_results = run_sequential(args.host, items, desc="  Predicting")

    # ── Concurrent evaluation ─────────────────────────────────────────────────
    print(f"\n[2/3] Concurrent evaluation (threads={args.concurrency}) …")
    con_results = run_concurrent(args.host, items, args.concurrency, desc="  Concurrent")

    all_results = seq_results + con_results
    metrics = compute_metrics(all_results)

    # ── Throughput test ───────────────────────────────────────────────────────
    print(f"\n[3/3] Throughput test ({args.throughput_sec}s) …")
    rpm, total_reqs = throughput_test(args.host, args.throughput_sec, args.concurrency)

    # ── Report & plots ────────────────────────────────────────────────────────
    print_report(metrics, rpm)
    plot_all(metrics, rpm, total_reqs, args.save_plots)


if __name__ == "__main__":
    main()
