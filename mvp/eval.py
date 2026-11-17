#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""哼唱转 MIDI 离线评测。

直接调用 server.extract_notes，保证评测走的是线上同一条管线。

指标：
  - 音高正确率：与期望音高完全相等的音符 / 期望音符数
  - 八度错误数：与期望相差 12 的整数倍（不含 0）
  - 音符数误差：识别数 - 期望数
  - 起音对齐误差：对齐后 |识别起音 - 期望起音|，ms
  - 量化位移：量化前后同一音符的起音/时值变化，ms

用法：
  .venv/bin/python eval.py
  .venv/bin/python eval.py --wav test_hum.wav --expected eval_cases/test_hum.json
  .venv/bin/python eval.py --wav foo.wav --expected foo.json --json tmp/foo_eval.json

期望文件格式：
  {"name": "...", "bpm": 100,
   "notes": [{"pitch": 60, "start": 0.0, "dur": 0.6}, ...]}
start/dur 可省略：省略 start 时按顺序紧接前一个音；省略 dur 时取一拍。
start 以 server 去头尾静音后的时间为准。
"""
import argparse
import json
import math
import os
import sys

import librosa
import numpy as np
import soundfile as sf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
import server  # noqa: E402


def load_audio(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    if sr != server.SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=server.SR)
        sr = server.SR
    return y, sr


def normalize_case(case):
    bpm = float(case.get("bpm", 100))
    beat = 60.0 / bpm
    notes = []
    cursor = 0.0
    for n in case["notes"]:
        dur = float(n.get("dur", beat))
        start = float(n.get("start", cursor))
        notes.append({"pitch": int(n["pitch"]), "start": start, "dur": dur})
        cursor = start + dur
    return {"name": case.get("name", "case"), "bpm": bpm, "notes": notes}


def run_pipeline(y, sr, bpm):
    """返回 (量化前音符, 量化后音符)。量化调用被包一层以记录输入。"""
    captured = {}
    orig_quantize = server.quantize

    def spy(notes, tempo, grid_per_beat=4):
        captured["raw"] = [dict(n) for n in notes]
        return orig_quantize(notes, tempo, grid_per_beat)

    server.quantize = spy
    try:
        quantized = server.extract_notes(y, sr, bpm)
    finally:
        server.quantize = orig_quantize
    return captured.get("raw", []), quantized


def align(expected, detected):
    """编辑距离对齐，返回 (pairs, missing, extra)；pairs 为 (i, j)。"""
    n, m = len(expected), len(detected)
    dp = np.zeros((n + 1, m + 1), dtype=float)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delta = abs(expected[i - 1]["pitch"] - detected[j - 1]["pitch"])
            if delta == 0:
                sub = 0.0
            elif delta % 12 == 0:
                sub = 0.5  # 八度错仍然算对齐，单独统计
            else:
                sub = 1.0
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub,
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
            )

    pairs, missing, extra = [], [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            delta = abs(expected[i - 1]["pitch"] - detected[j - 1]["pitch"])
            sub = 0.0 if delta == 0 else (0.5 if delta % 12 == 0 else 1.0)
            if abs(dp[i][j] - (dp[i - 1][j - 1] + sub)) < 1e-9:
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + 1.0)) < 1e-9:
            missing.append(i - 1)
            i -= 1
        else:
            extra.append(j - 1)
            j -= 1
    pairs.reverse()
    missing.reverse()
    extra.reverse()
    return pairs, missing, extra


def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def evaluate(case, detected, raw, bpm):
    expected = case["notes"]
    pairs, missing, extra = align(expected, detected)
    step = 60.0 / bpm / 4.0

    exact = 0
    octave = 0
    other = 0
    onset_err = []
    dur_err = []
    wrong = []
    for i, j in pairs:
        e, d = expected[i], detected[j]
        delta = d["pitch"] - e["pitch"]
        if delta == 0:
            exact += 1
        elif delta % 12 == 0:
            octave += 1
        else:
            other += 1
        onset_err.append(abs(d["start"] - e["start"]))
        dur_err.append(abs(d["dur"] - e["dur"]))
        if delta != 0:
            wrong.append({
                "index": i,
                "time": round(e["start"], 3),
                "expected": e["pitch"],
                "detected": d["pitch"],
                "delta": delta,
                "octave": delta % 12 == 0,
            })

    n_exp = len(expected)
    n_det = len(detected)
    quant_shift = []
    quant_dur_shift = []
    if len(raw) == len(detected):
        for r, q in zip(raw, detected):
            quant_shift.append(abs(q["start"] - r["start"]))
            quant_dur_shift.append(abs(q["dur"] - r["dur"]))

    return {
        "name": case["name"],
        "bpm": bpm,
        "grid_sec": round(step, 4),
        "expected_count": n_exp,
        "detected_count": n_det,
        "count_error": n_det - n_exp,
        "exact": exact,
        "pitch_accuracy": round(exact / n_exp, 4) if n_exp else 0.0,
        "pitch_precision": round(exact / n_det, 4) if n_det else 0.0,
        "octave_errors": octave,
        "other_pitch_errors": other,
        "missing": len(missing),
        "extra": len(extra),
        "onset_median_ms": round(1000 * percentile(onset_err, 50), 1) if onset_err else None,
        "onset_p90_ms": round(1000 * percentile(onset_err, 90), 1) if onset_err else None,
        "onset_max_ms": round(1000 * max(onset_err), 1) if onset_err else None,
        "onset_within_half_grid": int(sum(e <= step / 2 + 1e-9 for e in onset_err)),
        "duration_median_ms": round(1000 * percentile(dur_err, 50), 1) if dur_err else None,
        "duration_max_ms": round(1000 * max(dur_err), 1) if dur_err else None,
        "quantize_onset_median_ms": round(1000 * percentile(quant_shift, 50), 1) if quant_shift else None,
        "quantize_onset_max_ms": round(1000 * max(quant_shift), 1) if quant_shift else None,
        "quantize_duration_max_ms": round(1000 * max(quant_dur_shift), 1) if quant_dur_shift else None,
        "wrong_notes": wrong,
        "detected": [
            {"start": round(n["start"], 3), "dur": round(n["dur"], 3), "pitch": n["pitch"]}
            for n in detected
        ],
        "raw": [
            {"start": round(n["start"], 3), "dur": round(n["dur"], 3), "pitch": n["pitch"]}
            for n in raw
        ],
    }


def print_report(metrics):
    print("case: %s  bpm=%g  grid=%.3fs" % (
        metrics["name"], metrics["bpm"], metrics["grid_sec"]))
    print("expected %d | detected %d | count error %+d" % (
        metrics["expected_count"], metrics["detected_count"], metrics["count_error"]))
    print("pitch exact %d/%d = %.1f%% (precision %.1f%%)" % (
        metrics["exact"], metrics["expected_count"],
        100 * metrics["pitch_accuracy"], 100 * metrics["pitch_precision"]))
    print("octave errors %d | other pitch errors %d | missing %d | extra %d" % (
        metrics["octave_errors"], metrics["other_pitch_errors"],
        metrics["missing"], metrics["extra"]))
    if metrics["onset_median_ms"] is not None:
        print("onset error ms: median %.1f | p90 %.1f | max %.1f | within half-grid %d" % (
            metrics["onset_median_ms"], metrics["onset_p90_ms"],
            metrics["onset_max_ms"], metrics["onset_within_half_grid"]))
        print("duration error ms: median %.1f | max %.1f" % (
            metrics["duration_median_ms"], metrics["duration_max_ms"]))
    if metrics["quantize_onset_median_ms"] is not None:
        print("quantize shift ms: onset median %.1f | onset max %.1f | duration max %.1f" % (
            metrics["quantize_onset_median_ms"], metrics["quantize_onset_max_ms"],
            metrics["quantize_duration_max_ms"]))
    if metrics["wrong_notes"]:
        print("wrong notes:")
        for w in metrics["wrong_notes"]:
            kind = "OCTAVE" if w["octave"] else "pitch"
            print("  #%d t=%.2fs expected %d got %d (%+d, %s)" % (
                w["index"], w["time"], w["expected"], w["detected"],
                w["delta"], kind))


def main():
    ap = argparse.ArgumentParser(description="哼唱转 MIDI 评测")
    ap.add_argument("--wav", default=os.path.join(APP_DIR, "test_hum.wav"))
    ap.add_argument("--expected", default=None,
                    help="期望音符 JSON；默认找 eval_cases/<wav名>.json")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    expected_path = args.expected
    if expected_path is None:
        stem = os.path.splitext(os.path.basename(args.wav))[0]
        expected_path = os.path.join(APP_DIR, "eval_cases", stem + ".json")
    if not os.path.exists(expected_path):
        raise SystemExit("找不到期望文件: %s" % expected_path)
    if not os.path.exists(args.wav):
        raise SystemExit("找不到音频: %s" % args.wav)

    with open(expected_path, "r", encoding="utf-8") as fh:
        case = normalize_case(json.load(fh))

    y, sr = load_audio(args.wav)
    raw, detected = run_pipeline(y, sr, case["bpm"])
    metrics = evaluate(case, detected, raw, case["bpm"])
    metrics["wav"] = os.path.abspath(args.wav)
    metrics["expected_path"] = os.path.abspath(expected_path)
    print_report(metrics)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=1)
        print("json -> %s" % args.json_out)


if __name__ == "__main__":
    main()
