# pairwise-audio-voice-stealing

## 领域

游戏、图形与多媒体

## 主题

音频 voice 抢占

## 说明

优先级保留、跨线程原子与时钟冻结。

初始状态下 `media/audio_voice_stealing.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 实现要点

- **优先级与保留通道**：`dialogue` / `critical` 为受保护类别，只能占用保留 voice，
  非保留类别永远不会占用或抢占保留 voice。
- **不向上抢占**：只能抢优先级严格更低的声音；同类别同优先级时只抢更早的，
  已被挂起接管的 voice 不会被二次抢占。环境音再也抢不走正在播放的对白。
- **淡出完成再复用**：被抢占的 voice 先进入 `FADING_OUT`，包络精确归零后才把
  voice 交给新声音并从 0 开始 `FADING_IN`，两段包络永不重叠（消除爆音）。
  大渲染块会在零点处分段，不会越过淡出边界。
- **跨线程原子参数**：`AtomicParamStore` 在游戏线程 `stage`/`commit`，回调线程每块
  `snapshot()` 只取一个不可变快照，不会读到半更新状态。
- **回调时钟随暂停冻结**：`CallbackClock` 暂停时 `step()` 返回 0，包络与起始时间
  一并冻结，恢复后不会跳帧。
- **确定性抢占顺序**：受害者按 `(priority, started_at, seq, index)` 选取，
  请求按 `(seq)` 派发，不依赖字典或线程调度顺序。
- **过载降级**：`overload_degrade` 按占用率分四级，依次 duck 音乐、静音环境音、
  静音音乐并保护对白/保留通道。

行为测试见 `tests/test_audio_voice_stealing_behavior.py`。
