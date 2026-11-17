# HumToTrack · 哼唱转 MIDI

哼一段旋律，直接得到一条能拖进库乐队（GarageBand）用的 MIDI 轨。

## 是什么

一个自用的小工具：浏览器里录音 → 选目标乐器（钢琴 / 吉他 / 鼓）→ 给个 BPM → 后端把哼唱转成音符并量化 → 导出标准 `.mid`。

- 旋律：单声部哼唱 → 单条乐器轨（钢琴 / 吉他单音）
- 鼓：嘴打「哒哒」节奏 → 底鼓 / 军鼓 / 踩镲（GM 鼓位 36 / 38 / 42）
- 音色和后期全部交给库乐队，工具只负责出 MIDI

## 目录

```
docs/                产品需求与实现方案
mvp/
  server.py          Flask 服务：录音上传、音高识别、量化、写 MIDI
  index.html         录音页面
  eval.py            评测脚本（音高正确率、八度错误、音符数误差、起音/时值误差）
  eval_stress.py     输入级压力测试（不改算法，只改输入）
  eval_cases/        评测真值
  test_hum.wav       合成哼唱样例（C4-D4-E4-G4-E4-D4-C4）
  test_drum.wav      合成鼓点样例
```

## 运行

```bash
cd mvp
python3 -m venv .venv
.venv/bin/pip install librosa pretty_midi flask soundfile
.venv/bin/python server.py
```

打开 http://127.0.0.1:8765 ，点录音，哼，停止，转 MIDI。

输出默认落在 `mvp/out/`（`.mid` + 干声 `.wav` + 参数 `.json`），可用环境变量 `HUM_OUT` 覆盖。

## 评测

```bash
cd mvp
.venv/bin/python eval.py
.venv/bin/python eval_stress.py
```

## 现状与已知限制

- 旋律链路已跑通，合成样本 7/7 音全对；真人哼唱的准确率仍在调优，缺真实样本基线
- 切分阈值、最短音时长、量化相位、弱音截断等已知问题见 `docs/实现方案.md`
- 鼓分件为启发式，军鼓/踩镲存在误判
- 仅支持 macOS + 库乐队，BPM 需手动给定

## 许可

仅供个人使用，未做分发。
