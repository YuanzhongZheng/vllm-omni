#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Annotation pipeline benchmark — vllm-omni offline inference.

Three modes (--mode):
  quick  — latency on N files × R runs, then throughput pass (default)
  sweep  — batch-size sweep: compare annotate_batch(B) vs B×annotate(1)
  full   — quick + sweep back to back

Metrics collected:
  files/sec, per-file latency (mean ± std), tokens/sec, parse-error %,
  GPU utilisation %, GPU memory GB (nvidia-smi), peak torch-allocated GB

Results are printed as an ASCII report AND saved to:
  bench_<mode>_<timestamp>.log   — human-readable
  bench_results_<GPU>.json       — machine-readable, one entry per run

Usage (minimal):
    python benchmark.py --audio-dir /data/audio

    python benchmark.py --audio-dir /data/audio --mode sweep
    python benchmark.py --audio-dir /data/audio --mode full --files 20
    python benchmark.py --audio-dir /data/audio --batch-sizes 1 4 8 16 32
    python benchmark.py --audio-dir /data/audio --gpu 1 --model /path/to/model
"""

# Set CUDA_VISIBLE_DEVICES before any CUDA import
import sys
for _i, _a in enumerate(sys.argv[1:], 1):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        import os as _os
        _os.environ.setdefault("CUDA_VISIBLE_DEVICES", sys.argv[_i + 1])
        break

import gc
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# annotate_audio.py lives in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_audio import (
    SUPPORTED_AUDIO_EXTS,
    Omni,
    SamplingParams,
    SEED,
    TALKER_CODEC_EOS_TOKEN_ID,
    _PROMPT_TEMPLATE,
    SAMPLING_RATE,
    annotate_batch,
    _make_error_result,
    _parse_output,
)
from vllm.multimodal.media.audio import load_audio

_W = 84   # report width


# ─── GPU monitor ─────────────────────────────────────────────────────────────

class GPUMonitor:
    """Background thread that polls nvidia-smi every `interval_s` seconds."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.25):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._utils: List[float] = []
        self._mems_mib: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._utils.clear()
        self._mems_mib.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        u = self._utils or [0.0]
        m = self._mems_mib or [0.0]
        return {
            "util_mean_pct": float(np.mean(u)),
            "util_max_pct": float(np.max(u)),
            "mem_mean_gb": float(np.mean(m)) / 1024.0,
            "mem_max_gb": float(np.max(m)) / 1024.0,
        }

    def _poll(self) -> None:
        cmd = [
            "nvidia-smi", f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, timeout=2, text=True,
                                              stderr=subprocess.DEVNULL)
                parts = out.strip().split(",")
                if len(parts) == 2:
                    self._utils.append(float(parts[0].strip()))
                    self._mems_mib.append(float(parts[1].strip()))
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def _get_gpu_name(gpu_index: int = 0) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        names = out.stdout.strip().split("\n")
        return names[min(gpu_index, len(names) - 1)].strip().replace(" ", "_")
    except Exception:
        return "Unknown_GPU"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def collect_files(audio_dir: str, max_files: Optional[int] = None) -> List[Path]:
    files = sorted(
        f for f in Path(audio_dir).rglob("*")
        if f.suffix.lower() in SUPPORTED_AUDIO_EXTS
    )
    if max_files:
        files = files[:max_files]
    return files


def _token_count(raw: str) -> int:
    return len(raw.split()) if raw else 0


def _is_oom(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _banner(title: str) -> str:
    return "\n".join([
        "=" * _W,
        f"{title:^{_W}}",
        "=" * _W,
        f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])


# ─── Single-file annotate (for sequential baseline inside sweep) ──────────────

def _annotate_single(omni: Omni, audio_path: str, max_new_tokens: int) -> dict:
    """Annotate one file using the same Omni engine, bypassing batch overhead."""
    return annotate_batch(omni, [audio_path], max_new_tokens=max_new_tokens)[0]


# ─── Warmup ───────────────────────────────────────────────────────────────────

def run_warmup(omni: Omni, files: List[Path]) -> float:
    print("\n  [warmup] running first inference to load model weights into GPU …")
    t0 = time.perf_counter()
    _annotate_single(omni, str(files[0]), max_new_tokens=64)
    elapsed = time.perf_counter() - t0
    print(f"  [warmup] {elapsed:.1f}s")
    return elapsed


# ─── Mode: quick ─────────────────────────────────────────────────────────────

@dataclass
class QuickResult:
    files: int = 0
    runs: int = 0
    # latency (repeated runs on the same files)
    lat_mean_s: float = 0.0
    lat_std_s: float = 0.0
    lat_min_s: float = 0.0
    lat_max_s: float = 0.0
    lat_files_per_sec: float = 0.0
    lat_tokens_per_sec: float = 0.0
    lat_errors: int = 0
    lat_parse_errors: int = 0
    # throughput (single sequential pass)
    tput_elapsed_s: float = 0.0
    tput_files_per_sec: float = 0.0
    tput_tokens_per_sec: float = 0.0
    tput_errors: int = 0
    tput_parse_errors: int = 0
    # GPU
    util_mean_pct: float = 0.0
    util_max_pct: float = 0.0
    mem_mean_gb: float = 0.0
    mem_max_gb: float = 0.0
    torch_peak_gb: float = 0.0


def run_quick(
    omni: Omni,
    files: List[Path],
    runs: int,
    max_new_tokens: int,
    monitor: GPUMonitor,
) -> QuickResult:
    res = QuickResult(files=len(files), runs=runs)
    paths = [str(f) for f in files]

    # ── latency: run each file `runs` times ──
    print(f"\n  [latency] {len(files)} file(s) × {runs} run(s) ...")
    latencies: List[float] = []
    lat_tokens: List[int] = []

    torch.cuda.reset_peak_memory_stats()
    monitor.start()

    for path in paths:
        for r in range(runs):
            t0 = time.perf_counter()
            try:
                result = _annotate_single(omni, path, max_new_tokens)
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)
                lat_tokens.append(_token_count(result.get("raw_output", "")))
                if result.get("parse_error"):
                    res.lat_parse_errors += 1
                print(f"    {Path(path).name}  run {r+1}: {elapsed:.2f}s"
                      + (" [parse_err]" if result.get("parse_error") else ""))
            except Exception as e:
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)
                lat_tokens.append(0)
                res.lat_errors += 1
                print(f"    {Path(path).name}  run {r+1}: ERROR {e}")

    if latencies:
        res.lat_mean_s = float(np.mean(latencies))
        res.lat_std_s = float(np.std(latencies))
        res.lat_min_s = float(np.min(latencies))
        res.lat_max_s = float(np.max(latencies))
        res.lat_files_per_sec = 1.0 / res.lat_mean_s if res.lat_mean_s > 0 else 0.0
        res.lat_tokens_per_sec = (float(np.mean(lat_tokens)) / res.lat_mean_s
                                  if res.lat_mean_s > 0 else 0.0)

    # ── throughput: single pass over all files ──
    print(f"\n  [throughput] sequential pass over {len(files)} files ...")
    total_tokens = 0
    t0 = time.perf_counter()
    for idx, path in enumerate(paths, 1):
        try:
            result = _annotate_single(omni, path, max_new_tokens)
            total_tokens += _token_count(result.get("raw_output", ""))
            if result.get("parse_error"):
                res.tput_parse_errors += 1
        except Exception as e:
            res.tput_errors += 1
            print(f"    [{idx}] ERROR {e}")
        if idx % max(1, len(paths) // 4) == 0 or idx == len(paths):
            el = time.perf_counter() - t0
            print(f"    [{idx}/{len(paths)}] {el:.0f}s elapsed  "
                  f"{el/idx:.1f}s/file  errors={res.tput_errors}")

    res.tput_elapsed_s = time.perf_counter() - t0
    res.tput_files_per_sec = (len(files) / res.tput_elapsed_s
                               if res.tput_elapsed_s > 0 else 0.0)
    res.tput_tokens_per_sec = (total_tokens / res.tput_elapsed_s
                                if res.tput_elapsed_s > 0 else 0.0)

    gpu_stats = monitor.stop()
    res.torch_peak_gb = torch.cuda.max_memory_allocated() / 1e9
    res.util_mean_pct = gpu_stats["util_mean_pct"]
    res.util_max_pct = gpu_stats["util_max_pct"]
    res.mem_mean_gb = gpu_stats["mem_mean_gb"]
    res.mem_max_gb = gpu_stats["mem_max_gb"]
    return res


def format_quick_report(r: QuickResult, model: str, gpu: str) -> str:
    sep = "─" * _W
    return "\n".join([
        _banner("QUICK BENCHMARK"),
        f"model : {model}",
        f"gpu   : {gpu}",
        f"files : {r.files}  runs_per_file : {r.runs}",
        "",
        "── Latency " + "─" * 70,
        f"  mean ± std   : {r.lat_mean_s:.2f}s ± {r.lat_std_s:.2f}s",
        f"  min / max    : {r.lat_min_s:.2f}s / {r.lat_max_s:.2f}s",
        f"  files/sec    : {r.lat_files_per_sec:.3f}",
        f"  tokens/sec   : {r.lat_tokens_per_sec:.1f}",
        f"  parse errors : {r.lat_parse_errors}  inference errors : {r.lat_errors}",
        "",
        "── Throughput (sequential pass) " + "─" * 49,
        f"  total elapsed : {r.tput_elapsed_s:.1f}s",
        f"  files/sec     : {r.tput_files_per_sec:.3f}",
        f"  tokens/sec    : {r.tput_tokens_per_sec:.1f}",
        f"  parse errors  : {r.tput_parse_errors}  inference errors : {r.tput_errors}",
        "",
        "── GPU " + "─" * 75,
        f"  util (mean/max) : {r.util_mean_pct:.0f}% / {r.util_max_pct:.0f}%",
        f"  mem  (mean/max) : {r.mem_mean_gb:.1f} GB / {r.mem_max_gb:.1f} GB  "
        f"(nvidia-smi)",
        f"  torch peak      : {r.torch_peak_gb:.1f} GB",
        "=" * _W,
    ])


# ─── Mode: sweep ─────────────────────────────────────────────────────────────

@dataclass
class SweepRow:
    batch_size: int = 0
    batch_wall_s: float = 0.0
    batch_wall_std_s: float = 0.0
    batch_lat_per_file_s: float = 0.0
    batch_files_per_sec: float = 0.0
    seq_wall_s: float = 0.0
    seq_wall_std_s: float = 0.0
    seq_lat_per_file_s: float = 0.0
    seq_files_per_sec: float = 0.0
    speedup: float = 0.0
    parse_errors: int = 0
    errors: int = 0
    oom: bool = False


def _bench_one_batch_size(
    omni: Omni,
    files: List[Path],
    batch_size: int,
    runs: int,
    max_new_tokens: int,
) -> SweepRow:
    paths = [str(files[i % len(files)]) for i in range(batch_size)]
    row = SweepRow(batch_size=batch_size)

    # batch runs
    batch_walls: List[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            results = annotate_batch(omni, paths, max_new_tokens=max_new_tokens)
        except Exception as exc:
            if _is_oom(exc):
                torch.cuda.empty_cache(); gc.collect()
                row.oom = True
                return row
            row.errors += 1
            print(f"    [B={batch_size}] batch error: {exc}")
            continue
        torch.cuda.synchronize()
        batch_walls.append(time.perf_counter() - t0)
        row.parse_errors += sum(1 for r in results if r.get("parse_error"))

    if not batch_walls:
        return row

    row.batch_wall_s = float(np.mean(batch_walls))
    row.batch_wall_std_s = float(np.std(batch_walls))
    row.batch_lat_per_file_s = row.batch_wall_s / batch_size
    row.batch_files_per_sec = batch_size / row.batch_wall_s

    # sequential baseline: B × annotate(1)
    seq_walls: List[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok = True
        for path in paths:
            try:
                _annotate_single(omni, path, max_new_tokens)
            except Exception as exc:
                if _is_oom(exc):
                    torch.cuda.empty_cache(); gc.collect()
                row.errors += 1
                print(f"    [B={batch_size}] seq error: {exc}")
                ok = False
                break
        if not ok:
            continue
        torch.cuda.synchronize()
        seq_walls.append(time.perf_counter() - t0)

    if seq_walls:
        row.seq_wall_s = float(np.mean(seq_walls))
        row.seq_wall_std_s = float(np.std(seq_walls))
        row.seq_lat_per_file_s = row.seq_wall_s / batch_size
        row.seq_files_per_sec = batch_size / row.seq_wall_s
        row.speedup = row.seq_wall_s / row.batch_wall_s if row.batch_wall_s > 0 else 0.0

    return row


def run_sweep(
    omni: Omni,
    files: List[Path],
    batch_sizes: List[int],
    runs: int,
    max_new_tokens: int,
    monitor: GPUMonitor,
) -> List[SweepRow]:
    rows: List[SweepRow] = []
    torch.cuda.reset_peak_memory_stats()
    monitor.start()

    for b in batch_sizes:
        print(f"\n  [sweep B={b}] {runs} round(s) ...")
        row = _bench_one_batch_size(omni, files, b, runs, max_new_tokens)
        rows.append(row)
        if row.oom:
            print(f"  [sweep B={b}] OOM — stopping sweep.")
            break
        print(
            f"  [sweep B={b}] "
            f"batch={row.batch_wall_s*1000:.0f}±{row.batch_wall_std_s*1000:.0f}ms  "
            f"seq={row.seq_wall_s*1000:.0f}±{row.seq_wall_std_s*1000:.0f}ms  "
            f"speedup={row.speedup:.2f}×  "
            f"fps_batch={row.batch_files_per_sec:.3f}  fps_seq={row.seq_files_per_sec:.3f}"
            + (f"  [err={row.errors}]" if row.errors else "")
        )

    monitor.stop()
    return rows


def format_sweep_report(rows: List[SweepRow], model: str, gpu: str) -> str:
    hdr = (
        f"{'B':>4} | {'batch ms':>8} | {'±':>5} | "
        f"{'seq ms':>7} | {'±':>5} | "
        f"{'speedup':>7} | {'b fps':>6} | {'s fps':>6} | {'p_err':>5}"
    )
    sep = "─" * _W
    lines = [
        _banner("BATCH-SIZE SWEEP — vllm-omni annotate_batch vs sequential"),
        f"model : {model}",
        f"gpu   : {gpu}",
        "batch = annotate_batch(B)  |  seq = B × annotate(1)",
        "",
        hdr,
        sep,
    ]
    for r in rows:
        if r.oom:
            lines.append(f"{r.batch_size:>4} | {'OOM — stopped':^{_W-6}}")
            continue
        err = f"  [err={r.errors}]" if r.errors else ""
        lines.append(
            f"{r.batch_size:>4} | {r.batch_wall_s*1000:>7.0f}ms | {r.batch_wall_std_s*1000:>4.0f}ms | "
            f"{r.seq_wall_s*1000:>6.0f}ms | {r.seq_wall_std_s*1000:>4.0f}ms | "
            f"{r.speedup:>7.2f}× | {r.batch_files_per_sec:>6.3f} | {r.seq_files_per_sec:>6.3f} | "
            f"{r.parse_errors:>5d}{err}"
        )
    lines += [
        sep,
        "B       = batch size submitted in one annotate_batch() call",
        "speedup = seq_ms / batch_ms  (>1.0 means batching is faster)",
        "b fps   = batch throughput  (B / batch_wall)",
        "s fps   = serial throughput (B / seq_wall)",
        "p_err   = total parse errors across all batch runs",
        "=" * _W,
    ]
    return "\n".join(lines)


# ─── Save results ─────────────────────────────────────────────────────────────

def save_json(
    mode: str,
    gpu: str,
    model: str,
    quick: Optional[QuickResult],
    sweep: Optional[List[SweepRow]],
) -> str:
    fname = f"bench_results_{gpu}.json"
    data: Dict = {}
    if os.path.exists(fname) and os.path.getsize(fname) > 0:
        try:
            with open(fname) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    entry: Dict = {"mode": mode, "model": model, "gpu": gpu}
    if quick:
        entry["quick"] = asdict(quick)
    if sweep:
        entry["sweep"] = [asdict(r) for r in sweep]
    data[ts] = entry

    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    return fname


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Benchmark annotate_audio.py against audio files"
    )

    # Required
    p.add_argument("--audio-dir", required=True, default="/data4/zhengyuanzhong/audios_sample",
                   help="Directory of audio files (scanned recursively)")

    # Mode
    p.add_argument("--mode", choices=["quick", "sweep", "full"], default="quick",
                   help="quick=latency+throughput  sweep=batch-size sweep  "
                        "full=both  (default: quick)")

    # Model
    p.add_argument("--model", default="/data4/zhengyuanzhong/Qwen3-Omni-model/Qwen3-Omni-30B-A3B-Instruct",
                   help="HF model name or local path")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--stage-configs-path", default=None,
                   help="vllm-omni stage config YAML (optional)")

    # quick options
    p.add_argument("--files", type=int, default=50,
                   help="[quick/full] Number of unique audio files (default: 10)")
    p.add_argument("--runs", type=int, default=7,
                   help="[quick/full] Repeated runs per file for latency stats (default: 3)")

    # sweep options
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 24, 32],
                   help="[sweep/full] Batch sizes to try (stops on OOM, default: 1 2 4 8 16 24 32)")
    p.add_argument("--sweep-runs", type=int, default=7,
                   help="[sweep/full] Rounds per batch size (default: 3)")

    # common
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--gpu", type=str, default='1',
                   help="CUDA device(s), e.g. '0' or '0,1'")
    p.add_argument("--log-stats", action="store_true", default=False)

    # engine timeouts (same as annotate_audio.py / end2end.py)
    p.add_argument("--stage-init-timeout", type=int, default=300)
    p.add_argument("--batch-timeout", type=int, default=10)
    p.add_argument("--init-timeout", type=int, default=300)
    p.add_argument("--shm-threshold-bytes", type=int, default=65536)

    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gpu_index = int(args.gpu.split(",")[0]) if args.gpu else 0
    gpu_name = _get_gpu_name(gpu_index)

    # Collect files — cycle if not enough
    files = collect_files(args.audio_dir)
    if not files:
        print(f"No audio files found in {args.audio_dir}")
        sys.exit(1)

    print(f"Found {len(files)} audio file(s)  |  GPU: {gpu_name}  |  mode: {args.mode}")

    # Initialise engine once, shared across all benchmarks
    print(f"\nInitialising Omni engine from {args.model} …")
    omni = Omni(
        model=args.model,
        dtype=args.dtype,
        stage_configs_path=args.stage_configs_path,
        stage_init_timeout=args.stage_init_timeout,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
        log_stats=args.log_stats,
    )
    print(f"Engine ready  |  stages: {omni.num_stages}")

    monitor = GPUMonitor(gpu_index=gpu_index)
    run_warmup(omni, files)

    quick_result: Optional[QuickResult] = None
    sweep_rows: Optional[List[SweepRow]] = None
    reports: List[str] = []

    if args.mode in ("quick", "full"):
        # Cycle files to reach requested count
        quick_files = [files[i % len(files)] for i in range(args.files)]
        print(f"\n{'─'*_W}\n{'QUICK':^{_W}}\n{'─'*_W}")
        quick_result = run_quick(omni, quick_files, args.runs,
                                 args.max_new_tokens, monitor)
        rep = format_quick_report(quick_result, args.model, gpu_name)
        reports.append(rep)
        print("\n" + rep)

    if args.mode in ("sweep", "full"):
        print(f"\n{'─'*_W}\n{'SWEEP':^{_W}}\n{'─'*_W}")
        sweep_rows = run_sweep(omni, files, args.batch_sizes,
                               args.sweep_runs, args.max_new_tokens, monitor)
        rep = format_sweep_report(sweep_rows, args.model, gpu_name)
        reports.append(rep)
        print("\n" + rep)

    omni.close()

    # Save text report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"bench_{args.mode}_{ts}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(reports) + "\n")
    print(f"\nText report  : {log_path}")

    # Save JSON
    json_path = save_json(args.mode, gpu_name, args.model, quick_result, sweep_rows)
    print(f"JSON results : {json_path}")


if __name__ == "__main__":
    main()
