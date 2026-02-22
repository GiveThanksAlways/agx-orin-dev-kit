#!/usr/bin/env python3
"""Analyze control-loop benchmark results and generate report + plots.

Reads CSVs from results/ and produces:
  - results/report.md        (markdown summary table + analysis)
  - results/latency_cdf.png  (CDF of cycle times per framework)
  - results/latency_hist.png (histogram overlay)
  - results/jitter_box.png   (box plot of jitter)

Usage (from control-loop nix shell, which has matplotlib):
    python3 analyze.py [--input-dir results] [--output-dir results]
"""
import os, sys, argparse, csv, textwrap
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("WARNING: matplotlib not available, skipping plots.")

# ── CSV loader ───────────────────────────────────────────────────────────────
def load_csv(path):
    """Return dict of column_name -> np.array(float64)."""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        cols = {h: [] for h in headers}
        for row in reader:
            for h, v in zip(headers, row):
                cols[h].append(float(v))
    return {h: np.array(v) for h, v in cols.items()}

def safe_load(base, name):
    p = os.path.join(base, name)
    if os.path.exists(p):
        return load_csv(p)
    return None

# ── Stats ────────────────────────────────────────────────────────────────────
def compute_stats(arr):
    return {
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr),
        'p99': np.percentile(arr, 99),
        'p999': np.percentile(arr, 99.9),
        'max': np.max(arr),
        'min': np.min(arr),
        'max_dev': np.max(np.abs(arr - np.mean(arr))),
        'count': len(arr),
        'hz': 1e6 / np.mean(arr) if np.mean(arr) > 0 else 0,
        'pct_1khz': 100.0 * np.sum(arr < 1000.0) / len(arr),
        'pct_2khz': 100.0 * np.sum(arr < 500.0) / len(arr),
    }

# ── Plotting ─────────────────────────────────────────────────────────────────
COLORS = {
    'tinygrad_nv':     '#1f77b4',
    'pytorch_eager':   '#ff7f0e',
    'pytorch_graph':   '#2ca02c',
}

def plot_cdf(datasets, title, xlabel, outpath):
    """CDF plot for multiple datasets."""
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data in datasets.items():
        s = np.sort(data)
        cdf = np.arange(1, len(s)+1) / len(s)
        ax.plot(s, cdf, label=label, color=COLORS.get(label), linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Zoom to p99.9
    all_vals = np.concatenate(list(datasets.values()))
    ax.set_xlim(0, np.percentile(all_vals, 99.9) * 1.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")

def plot_hist(datasets, title, xlabel, outpath, bins=200):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    all_vals = np.concatenate(list(datasets.values()))
    hi = np.percentile(all_vals, 99.5)
    for label, data in datasets.items():
        ax.hist(data[data < hi], bins=bins, alpha=0.5, label=label,
                color=COLORS.get(label), density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")

def plot_box(datasets, title, ylabel, outpath):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(datasets.keys())
    data = [datasets[l] for l in labels]
    # Clip to p99.9 for readable box plots
    clipped = []
    for d in data:
        hi = np.percentile(d, 99.9)
        clipped.append(d[d < hi])
    bp = ax.boxplot(clipped, labels=labels, patch_artist=True)
    colors = [COLORS.get(l, '#999999') for l in labels]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")

def plot_deadline(datasets, title, outpath):
    """Bar chart: % of iterations meeting 1 kHz and 2 kHz deadlines."""
    if plt is None:
        return
    labels = list(datasets.keys())
    pct_1k = [100.0 * np.sum(datasets[l] < 1000.0) / len(datasets[l]) for l in labels]
    pct_2k = [100.0 * np.sum(datasets[l] < 500.0) / len(datasets[l]) for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, pct_1k, w, label='< 1000 µs (1 kHz)', color='#4c72b0')
    ax.bar(x + w/2, pct_2k, w, label='< 500 µs (2 kHz)', color='#dd8452')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('% iterations meeting deadline')
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")

# ── Report generation ────────────────────────────────────────────────────────
def fmt(v, prec=1):
    return f"{v:.{prec}f}"

def stat_row(label, s):
    return (f"| {label:24s} | {fmt(s['mean']):>10s} | {fmt(s['median']):>10s} | "
            f"{fmt(s['std']):>8s} | {fmt(s['p99']):>10s} | {fmt(s['p999']):>10s} | "
            f"{fmt(s['max']):>10s} | {fmt(s['hz'],0):>10s} | {s['count']:>10d} |")

def generate_report(all_stats, outpath, duration):
    hdr = (f"| {'Framework':24s} | {'Mean µs':>10s} | {'Median µs':>10s} | "
           f"{'Std µs':>8s} | {'P99 µs':>10s} | {'P99.9 µs':>10s} | "
           f"{'Max µs':>10s} | {'Hz':>10s} | {'Iters':>10s} |")
    sep = "|" + "|".join(["-" * 26] + ["-" * 12] * 7) + "|"

    sections = []
    for phase_name, phase_data in all_stats.items():
        lines = [f"### {phase_name}\n", hdr, sep]
        for label, s in phase_data.items():
            lines.append(stat_row(label, s))
        sections.append("\n".join(lines))

    # Compute savings
    savings_text = "\n## Analysis\n\n"
    for phase_name, phase_data in all_stats.items():
        keys = list(phase_data.keys())
        if len(keys) < 2:
            continue
        nv = phase_data.get('tinygrad_nv')
        for other_key in keys:
            if other_key == 'tinygrad_nv':
                continue
            other = phase_data[other_key]
            if nv and other:
                ratio = other['mean'] / nv['mean'] if nv['mean'] > 0 else 0
                save = other['mean'] - nv['mean']
                savings_text += (f"- **{phase_name}**: tinygrad NV=1 is **{ratio:.2f}x faster** "
                                 f"than {other_key} (saves {save:.1f} µs/cycle, "
                                 f"mean {nv['mean']:.1f} vs {other['mean']:.1f} µs)\n")

    # Deadline analysis
    deadline_text = "\n## Real-Time Deadline Compliance\n\n"
    deadline_text += "| Framework | Phase | % < 1000 µs (1 kHz) | % < 500 µs (2 kHz) |\n"
    deadline_text += "|-----------|-------|---------------------:|-------------------:|\n"
    for phase_name, phase_data in all_stats.items():
        for label, s in phase_data.items():
            deadline_text += f"| {label} | {phase_name} | {s['pct_1khz']:.2f}% | {s['pct_2khz']:.2f}% |\n"

    report = textwrap.dedent(f"""\
    # Control-Loop Benchmark Report — Jetson AGX Orin 64GB

    **Date**: {time.strftime('%Y-%m-%d %H:%M')}
    **Duration per test**: {duration}s
    **Hardware**: Jetson AGX Orin 64GB (JetPack 6 / jetpack-nixos)
    **Model**: 2-layer MLP (128 hidden, FP16) — 12→128→128→4 (PID) / 24→128→128→4 (SF)

    ## Frameworks Tested

    | Framework | Description |
    |-----------|-------------|
    | tinygrad NV=1 | Direct GPU command queue via TegraIface (custom Orin port) |
    | pytorch_eager | Standard PyTorch CUDA forward pass |
    | pytorch_graph | PyTorch with CUDA Graph capture + replay |

    ## Results

    """) + "\n\n".join(sections) + savings_text + deadline_text

    # Robotics applicability
    report += textwrap.dedent("""
    ## Robotics / Drone Applicability

    The numbers above directly answer whether each framework can sustain
    high-frequency control loops on Jetson AGX Orin:

    - **1 kHz loops** (standard for robotic arms, legged robots): any framework
      with median cycle time < 1000 µs and p99 < 1000 µs is viable. Check the
      deadline compliance table above.
    - **2 kHz loops** (high-performance drones, fast servo control): requires
      median < 500 µs AND low jitter (std < 50 µs). Only frameworks with
      > 99% compliance at the 500 µs deadline are suitable.
    - **Jitter** (std dev of cycle time) determines control stability. Lower
      is better. Jitter > 100 µs at 1 kHz causes visible oscillation in PID
      controllers.
    - **p99.9 latency** determines worst-case behavior. For safety-critical
      systems, this must stay within the deadline budget.

    The key advantage of tinygrad's NV=1 backend is **direct hardware command
    queue submission** via Tegra ioctls, bypassing the CUDA runtime's kernel
    launch overhead. For tiny models (like the MLP used here), kernel launch
    latency dominates total cycle time, making this optimization critical.
    """)

    with open(outpath, 'w') as f:
        f.write(report)
    print(f"  Saved {outpath}")

# ── Main ─────────────────────────────────────────────────────────────────────
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='results')
    ap.add_argument('--output-dir', default='results')
    ap.add_argument('--duration', type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all available CSVs
    files = {
        # launch latency
        'tinygrad_nv_launch':     'tinygrad_nv_launch.csv',
        'pytorch_eager_launch':   'pytorch_eager_launch.csv',
        'pytorch_graph_launch':   'pytorch_graph_launch.csv',
        # PID loop
        'tinygrad_nv_pid':        'tinygrad_nv_pid.csv',
        'pytorch_eager_pid':      'pytorch_eager_pid.csv',
        'pytorch_graph_pid':      'pytorch_graph_pid.csv',
        # Sensor fusion
        'tinygrad_nv_sf':         'tinygrad_nv_sensor_fusion.csv',
        'pytorch_eager_sf':       'pytorch_eager_sensor_fusion.csv',
        'pytorch_graph_sf':       'pytorch_graph_sensor_fusion.csv',
    }

    data = {}
    for key, fname in files.items():
        d = safe_load(args.input_dir, fname)
        if d is not None:
            data[key] = d
            print(f"  Loaded {fname} ({len(list(d.values())[0])} rows)")

    if not data:
        print("ERROR: No CSV files found in", args.input_dir)
        sys.exit(1)

    # Organize stats by phase
    all_stats = {}

    # Launch latency
    launch_datasets = {}
    launch_phase = {}
    for key, col in [('tinygrad_nv_launch', 'latency_us'),
                      ('pytorch_eager_launch', 'latency_us'),
                      ('pytorch_graph_launch', 'latency_us')]:
        if key in data:
            label = key.replace('_launch', '')
            launch_datasets[label] = data[key][col]
            launch_phase[label] = compute_stats(data[key][col])
    if launch_phase:
        all_stats['Launch Latency'] = launch_phase

    # PID loop — cycle time
    pid_cyc = {}
    pid_inf = {}
    pid_phase = {}
    for key, label in [('tinygrad_nv_pid', 'tinygrad_nv'),
                        ('pytorch_eager_pid', 'pytorch_eager'),
                        ('pytorch_graph_pid', 'pytorch_graph')]:
        if key in data:
            pid_cyc[label] = data[key]['cycle_us']
            pid_inf[label] = data[key]['inference_us']
            pid_phase[label] = compute_stats(data[key]['cycle_us'])
    if pid_phase:
        all_stats['PID Loop — Total Cycle'] = pid_phase
    pid_inf_phase = {l: compute_stats(v) for l, v in pid_inf.items()}
    if pid_inf_phase:
        all_stats['PID Loop — Inference Only'] = pid_inf_phase

    # Sensor fusion loop
    sf_cyc = {}
    sf_inf = {}
    sf_phase = {}
    for key, label in [('tinygrad_nv_sf', 'tinygrad_nv'),
                        ('pytorch_eager_sf', 'pytorch_eager'),
                        ('pytorch_graph_sf', 'pytorch_graph')]:
        if key in data:
            sf_cyc[label] = data[key]['cycle_us']
            sf_inf[label] = data[key]['inference_us']
            sf_phase[label] = compute_stats(data[key]['cycle_us'])
    if sf_phase:
        all_stats['Sensor Fusion — Total Cycle'] = sf_phase
    sf_inf_phase = {l: compute_stats(v) for l, v in sf_inf.items()}
    if sf_inf_phase:
        all_stats['Sensor Fusion — Inference Only'] = sf_inf_phase

    # Generate plots
    print("\nGenerating plots...")
    if launch_datasets:
        plot_cdf(launch_datasets, 'Launch Latency CDF', 'Latency (µs)',
                 os.path.join(args.output_dir, 'launch_cdf.png'))
        plot_hist(launch_datasets, 'Launch Latency Distribution', 'Latency (µs)',
                  os.path.join(args.output_dir, 'launch_hist.png'))

    if pid_cyc:
        plot_cdf(pid_cyc, 'PID Loop — Cycle Time CDF', 'Cycle Time (µs)',
                 os.path.join(args.output_dir, 'pid_cycle_cdf.png'))
        plot_hist(pid_cyc, 'PID Loop — Cycle Time Distribution', 'Cycle Time (µs)',
                  os.path.join(args.output_dir, 'pid_cycle_hist.png'))
        plot_box(pid_cyc, 'PID Loop — Cycle Time Box Plot', 'Cycle Time (µs)',
                 os.path.join(args.output_dir, 'pid_cycle_box.png'))
        plot_deadline(pid_cyc, 'PID Loop — Deadline Compliance',
                      os.path.join(args.output_dir, 'pid_deadline.png'))

    if sf_cyc:
        plot_cdf(sf_cyc, 'Sensor Fusion — Cycle Time CDF', 'Cycle Time (µs)',
                 os.path.join(args.output_dir, 'sf_cycle_cdf.png'))
        plot_hist(sf_cyc, 'Sensor Fusion — Cycle Time Distribution', 'Cycle Time (µs)',
                  os.path.join(args.output_dir, 'sf_cycle_hist.png'))
        plot_box(sf_cyc, 'Sensor Fusion — Cycle Time Box Plot', 'Cycle Time (µs)',
                 os.path.join(args.output_dir, 'sf_cycle_box.png'))
        plot_deadline(sf_cyc, 'Sensor Fusion — Deadline Compliance',
                      os.path.join(args.output_dir, 'sf_deadline.png'))

    # Generate report
    print("\nGenerating report...")
    generate_report(all_stats, os.path.join(args.output_dir, 'report.md'), args.duration)
    print("\nAll done.")

if __name__ == '__main__':
    main()
