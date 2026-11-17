#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对现有管线做输入级压力评测（不改 server.py）。

case:
  chromatic  半音级旋律 C4 C#4 D4 C#4 C4，验证 1.2 半音切分阈值
  shift80    在第一个音后插入 80ms 静音，验证 1/16 量化取整误差
  dropout100 在 E4 中间插入 100ms 静音，验证断音容错/重复音

输出 wav 和 expected json 到 mvp/tmp/eval_stress/。
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
import eval as ev  # noqa: E402
import server  # noqa: E402

OUT = os.path.join(APP_DIR, "tmp", "eval_stress")


def synth_note(freq, dur, sr, amp=0.25):
    t = np.arange(int(round(dur * sr)), dtype=np.float64) / sr
    vib = 2 ** (10.0 / 1200 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(freq * vib) / sr
    y = np.zeros_like(t)
    for k in range(1, 7):
        y += np.sin(k * phase) / k
    y *= amp / max(1e-9, np.abs(y).max())
    edge = max(1, int(0.015 * sr))
    env = np.ones_like(y)
    env[:edge] = np.linspace(0, 1, edge)
    env[-edge:] = np.linspace(1, 0, edge)
    return (y * env).astype(np.float32)


def write_case(name, y, sr, bpm, notes):
    os.makedirs(OUT, exist_ok=True)
    wav_path = os.path.join(OUT, name + ".wav")
    json_path = os.path.join(OUT, name + ".json")
    sf.write(wav_path, y, sr)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"name": name, "bpm": bpm, "notes": notes}, fh, indent=1)
    return wav_path, json_path


def make_chromatic(sr):
    bpm = 120.0
    dur = 0.5
    pitches = [60, 61, 62, 61, 60]
    gap = int(0.02 * sr)
    parts = []
    for p in pitches:
        parts.append(synth_note(440.0 * 2 ** ((p - 69) / 12), dur, sr))
        parts.append(np.zeros(gap, dtype=np.float32))
    y = np.concatenate(parts)
    notes = [{"pitch": p, "start": i * dur, "dur": dur} for i, p in enumerate(pitches)]
    return write_case("chromatic", y, sr, bpm, notes)


def make_inserted(name, at_sec, gap_sec, src_notes, bpm):
    y, sr = ev.load_audio(os.path.join(APP_DIR, "test_hum.wav"))
    pos = int(round(at_sec * sr))
    gap = np.zeros(int(round(gap_sec * sr)), dtype=y.dtype)
    y = np.concatenate([y[:pos], gap, y[pos:]])
    notes = []
    for n in src_notes:
        start = n["start"] + (gap_sec if n["start"] >= at_sec - 1e-9 else 0.0)
        notes.append({"pitch": n["pitch"], "start": start, "dur": n["dur"]})
    return write_case(name, y, sr, bpm, notes)


def main():
    with open(os.path.join(APP_DIR, "eval_cases", "test_hum.json"), encoding="utf-8") as fh:
        base = ev.normalize_case(json.load(fh))

    cases = []
    sr = server.SR
    cases.append(("chromatic",) + make_chromatic(sr))
    cases.append(("shift80",) + make_inserted(
        "shift80", 0.6, 0.08, base["notes"], base["bpm"]))
    cases.append(("dropout100",) + make_inserted(
        "dropout100", 1.5, 0.10, base["notes"], base["bpm"]))

    results = []
    for name, wav_path, json_path in cases:
        with open(json_path, encoding="utf-8") as fh:
            case = ev.normalize_case(json.load(fh))
        y, sr = ev.load_audio(wav_path)
        raw, detected = ev.run_pipeline(y, sr, case["bpm"])
        m = ev.evaluate(case, detected, raw, case["bpm"])
        m["wav"] = wav_path
        results.append(m)
        print("=" * 58)
        ev.print_report(m)
        print("detected:", [(round(n["start"], 2), round(n["dur"], 2), n["pitch"])
                            for n in detected])

    json_path = os.path.join(OUT, "stress_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print("json -> %s" % json_path)


if __name__ == "__main__":
    main()
