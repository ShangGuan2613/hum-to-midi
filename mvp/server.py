#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""哼唱转 MIDI 最小验证服务"""
import os
import sys
import json
import math
import time
import uuid
import tempfile

import numpy as np
import librosa
import soundfile as sf
import pretty_midi
from flask import Flask, request, jsonify, send_file, send_from_directory

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("HUM_OUT") or os.path.join(APP_DIR, "out")
TMP_DIR = os.path.join(APP_DIR, "tmp")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

SR = 22050
HOP = 256

app = Flask(__name__, static_folder=None)


def log(msg):
    print("[hum]", msg, flush=True)


# ---------- 旋律：pyin 提音高 ----------
def extract_notes(y, sr, tempo):
    """单声部哼唱 -> 音符列表"""
    # 去静音头尾
    y, _ = librosa.effects.trim(y, top_db=35)
    if len(y) < sr // 4:
        return []

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=80.0, fmax=1000.0, sr=sr,
        frame_length=2048, hop_length=HOP,
    )

    # 帧时间
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)

    # 能量
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    rms = np.pad(rms, (0, max(0, len(f0) - len(rms))))[:len(f0)]
    rms_max = float(rms.max()) if rms.max() > 0 else 1.0

    # 逐帧整理
    frames = []
    for i, f in enumerate(f0):
        ok = (not math.isnan(f)) and voiced_flag[i] and voiced_prob[i] > 0.30 and f > 0
        if ok:
            midi = 69 + 12 * math.log2(f / 440.0)
            frames.append((float(times[i]), midi, float(rms[i])))
        else:
            frames.append((float(times[i]), None, float(rms[i])))

    # 中值平滑：在连续有声段内平滑音高，避免颤音抖成多个音
    midis = np.array([fr[1] if fr[1] is not None else np.nan for fr in frames], dtype=float)
    smoothed = midis.copy()
    for i in range(len(midis)):
        if not math.isnan(midis[i]):
            lo, hi = max(0, i - 4), min(len(midis), i + 5)
            seg = midis[lo:hi]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                smoothed[i] = float(np.median(seg))

    # 切音符：音高连续稳定 + 有声
    notes = []
    cur = None
    for i, (t, _, r) in enumerate(frames):
        m = smoothed[i]
        voiced = (not math.isnan(m)) and voiced_flag[i] and voiced_prob[i] > 0.30
        if not voiced:
            if cur is not None:
                notes.append(cur)
                cur = None
            continue
        if cur is None:
            cur = {"start": t, "end": t + HOP / sr, "midis": [m], "rms": [r]}
        else:
            # 音高跳变 > 1.2 半音视为新音
            if abs(m - float(np.median(cur["midis"]))) > 1.2:
                notes.append(cur)
                cur = {"start": t, "end": t + HOP / sr, "midis": [m], "rms": [r]}
            else:
                cur["end"] = t + HOP / sr
                cur["midis"].append(m)
                cur["rms"].append(r)
    if cur is not None:
        notes.append(cur)

    out = []
    for n in notes:
        dur = n["end"] - n["start"]
        if dur < 0.09:  # 太短多半是噪声
            continue
        pitch = int(round(float(np.median(n["midis"]))))
        pitch = max(21, min(108, pitch))
        vel = int(40 + 70 * min(1.0, (float(np.mean(n["rms"])) / rms_max) ** 0.5))
        out.append({"start": float(n["start"]), "dur": float(dur),
                    "pitch": pitch, "velocity": max(30, min(120, vel))})

    # 合并相邻同音（被短暂断开）
    merged = []
    for n in out:
        if merged and n["pitch"] == merged[-1]["pitch"] and \
                n["start"] - (merged[-1]["start"] + merged[-1]["dur"]) < 0.06:
            merged[-1]["dur"] = n["start"] + n["dur"] - merged[-1]["start"]
        else:
            merged.append(n)
    return quantize(merged, tempo)


# ---------- 节奏量化 ----------
def quantize(notes, tempo, grid_per_beat=4):
    """按 1/16 网格量化起音和时值"""
    if not notes:
        return []
    step = 60.0 / tempo / grid_per_beat  # 一格的秒数
    q = []
    for n in notes:
        start = round(n["start"] / step) * step
        dur_steps = max(1, round(n["dur"] / step))
        q.append({"start": start, "dur": dur_steps * step,
                  "pitch": n["pitch"], "velocity": n["velocity"]})
    # 防止重叠
    for i in range(len(q) - 1):
        end = q[i]["start"] + q[i]["dur"]
        if end > q[i + 1]["start"]:
            q[i]["dur"] = max(step, q[i + 1]["start"] - q[i]["start"])
    return q


# ---------- 鼓 ----------
def extract_drums(y, sr, tempo):
    """哒哒节奏 -> kick/snare/hihat"""
    y, _ = librosa.effects.trim(y, top_db=35)
    # 前置 20ms 静音，避免开头第一拍被当成帧边界漏掉
    pad = int(sr * 0.02)
    ypad = np.concatenate([np.zeros(pad, dtype=y.dtype), y])
    onsets = librosa.onset.onset_detect(
        y=ypad, sr=sr, hop_length=HOP, units="time", backtrack=True, delta=0.02
    )
    onsets = np.array([max(0.0, float(t) - pad / sr) for t in onsets])
    if len(onsets) == 0:
        return []

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

    def band_ratio(t, lo, hi):
        idx = int(t * sr / HOP)
        seg = S[:, max(0, idx - 1):idx + 2]
        if seg.size == 0:
            return 0.0
        e = seg.mean(axis=1)
        total = e.sum() + 1e-9
        mask = (freqs >= lo) & (freqs < hi)
        return float(e[mask].sum() / total) if mask.any() else 0.0

    hits = []
    for t in onsets:
        low = band_ratio(t, 0, 150)
        mid = band_ratio(t, 150, 2500)
        high = band_ratio(t, 2500, sr / 2)
        if low > 0.35 and low >= mid:
            pitch = 36  # kick
        elif high > 0.55 and mid < 0.30:
            pitch = 42  # hihat
        else:
            pitch = 38  # snare
        hits.append({"start": float(t), "pitch": pitch})

    step = 60.0 / tempo / 4
    out = []
    for h in hits:
        out.append({"start": round(h["start"] / step) * step,
                    "dur": step * 0.9, "pitch": h["pitch"], "velocity": 100})
    return out


# ---------- 写 MIDI ----------
def write_midi(notes, tempo, instrument):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    if instrument == "drum":
        inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        for n in notes:
            inst.notes.append(pretty_midi.Note(
                velocity=int(n["velocity"]), pitch=int(n["pitch"]),
                start=float(n["start"]), end=float(n["start"] + n["dur"])))
    else:
        program = 24 if instrument == "guitar" else 0
        name = "Guitar" if instrument == "guitar" else "Piano"
        inst = pretty_midi.Instrument(program=program, name=name)
        for n in notes:
            inst.notes.append(pretty_midi.Note(
                velocity=int(n["velocity"]), pitch=int(n["pitch"]),
                start=float(n["start"]), end=float(n["start"] + n["dur"])))
    pm.instruments.append(inst)
    return pm


# ---------- 路由 ----------
@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/api/convert", methods=["POST"])
def convert():
    f = request.files.get("audio")
    if f is None:
        return jsonify({"error": "没有收到录音"}), 400
    instrument = request.form.get("instrument", "piano")
    try:
        tempo = float(request.form.get("bpm") or 90)
    except ValueError:
        tempo = 90.0
    tempo = max(40.0, min(240.0, tempo))

    tmp = os.path.join(TMP_DIR, "hum_%s.wav" % uuid.uuid4().hex)
    f.save(tmp)
    try:
        y, sr = sf.read(tmp, dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            sr = SR
        duration = float(len(y)) / SR

        t0 = time.time()
        if instrument == "drum":
            notes = extract_drums(y, sr, tempo)
        else:
            notes = extract_notes(y, sr, tempo)
        elapsed = time.time() - t0

        pm = write_midi(notes, tempo, instrument)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        fname = "%s_%s.mid" % (stamp, instrument)
        path = os.path.join(OUT_DIR, fname)
        pm.write(path)
        # 保留干声，便于复盘算法问题
        sf.write(path[:-4] + ".wav", y, SR)
        with open(path[:-4] + ".json", "w", encoding="utf-8") as fh:
            json.dump({"instrument": instrument, "tempo": tempo,
                       "duration": duration, "notes": notes},
                      fh, ensure_ascii=False, indent=1)

        log("done %s notes=%d %.2fs -> %s" % (instrument, len(notes), elapsed, path))
        return jsonify({
            "ok": True,
            "midi_path": path,
            "filename": fname,
            "instrument": instrument,
            "tempo": tempo,
            "duration": round(duration, 2),
            "note_count": len(notes),
            "elapsed": round(elapsed, 2),
            "notes": notes[:40],
        })
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


@app.route("/api/download/<path:name>")
def download(name):
    safe = os.path.basename(name)
    path = os.path.join(OUT_DIR, safe)
    if not os.path.exists(path):
        return jsonify({"error": "文件不存在"}), 404
    return send_file(path, as_attachment=True, download_name=safe)


@app.route("/api/reveal", methods=["POST"])
def reveal():
    data = request.get_json(silent=True) or {}
    path = data.get("path") or ""
    safe = os.path.basename(path)
    full = os.path.join(OUT_DIR, safe)
    if os.path.exists(full):
        os.system("open -R '%s'" % full.replace("'", "'\\''"))
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 404


if __name__ == "__main__":
    port = int(os.environ.get("HUM_PORT", "8765"))
    log("serving on http://127.0.0.1:%d" % port)
    app.run(host="127.0.0.1", port=port, threaded=True)
