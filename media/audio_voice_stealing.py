"""Priority-aware audio voice allocation with pop-free stealing.

The module models the part of a game audio mixer that decides *which* sound
gets a voice, *when* a voice may be reused, and *how* state crosses the
boundary between the game thread and the real-time audio callback.

Design rules
------------

* **Priority and reservations.**  Categories carry priorities.  A configurable
  number of slots is reserved for protected categories (dialogue / critical
  cues).  Ordinary categories may never occupy or steal a reserved slot, so
  ambience cannot evict a line of dialogue even if the priority table is
  configured badly.
* **Never steal "up".**  A request may only steal a strictly lower-priority
  voice, or -- deterministically -- an older voice of the *same* category and
  priority.  A voice that is already fading out has been claimed and is never
  chosen as a victim twice.
* **Fade out before reuse.**  A stolen voice enters ``FADING_OUT`` and is not
  handed to the requester until its envelope has reached exactly zero.  The
  replacement starts ``FADING_IN`` from zero only after that point, so the two
  envelopes can never overlap on one voice -- the usual source of a click or
  pop.
* **Atomic parameters.**  :class:`AtomicParamStore` stages changes on the game
  thread and publishes one immutable snapshot at a time.  The callback reads a
  single reference per block, so it can never observe half of a multi-parameter
  update.
* **Freeze-able clock.**  :class:`CallbackClock` only advances while audio is
  actually being rendered.  Pausing freezes voice envelopes and start times;
  resuming therefore cannot make the mixer jump forward.
* **Deterministic stealing.**  Victim selection uses the total order
  ``(priority, started_at, seq, index)``.  Requests are serviced in
  ``(-priority, seq, index)`` order.  Nothing depends on dict or thread
  scheduling order.
* **Graceful overload.**  :func:`overload_degrade` maps utilisation to a
  deterministic degradation plan: ambience first, then music, then effects,
  while dialogue and critical cues stay protected.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable as _IterableABC
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

__all__ = [
    "Category",
    "DEFAULT_PRIORITIES",
    "PROTECTED_PRIORITY",
    "VoiceState",
    "Voice",
    "PendingStart",
    "Assignment",
    "CallbackClock",
    "AtomicParamStore",
    "VoicePool",
    "DegradePlan",
    "ValidationReport",
    "PopCheck",
    "DeterminismReport",
    "priority_map",
    "reserved_voice",
    "atomic_param",
    "fade_before_reuse",
    "clock_pause",
    "steal_order",
    "overload_degrade",
    "validation",
    "audit",
    "fallback",
    "pop_guard",
    "metrics",
    "voice_pool",
    "callback_state",
    "determinism",
]


class Category(str, Enum):
    """Logical sound categories understood by the mixer."""

    AMBIENT = "ambient"
    MUSIC = "music"
    SFX = "sfx"
    DIALOGUE = "dialogue"
    CRITICAL = "critical"


DEFAULT_PRIORITIES: Mapping[str, int] = MappingProxyType(
    {
        Category.AMBIENT.value: 0,
        Category.MUSIC.value: 10,
        Category.SFX.value: 20,
        Category.DIALOGUE.value: 80,
        Category.CRITICAL.value: 100,
    }
)

#: Categories at or above this priority are "protected": they may use the
#: reserved slots and can never be evicted by an unprotected request.
PROTECTED_PRIORITY = 80


class VoiceState(str, Enum):
    FREE = "free"
    ACTIVE = "active"
    FADING_OUT = "fading_out"
    FADING_IN = "fading_in"
    STOPPED = "stopped"


@dataclass
class Voice:
    """One mixer voice (a single concurrently playable sound)."""

    index: int
    category: str | None = None
    priority: int = -1
    state: VoiceState = VoiceState.FREE
    gain: float = 0.0
    fade_total: int = 0
    fade_elapsed: int = 0
    started_at: int = 0
    seq: int = 0
    token: str | None = None
    reserved: bool = False
    pending: "PendingStart | None" = None

    @property
    def busy(self) -> bool:
        return self.state is not VoiceState.FREE

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "category": self.category,
            "priority": self.priority,
            "state": self.state.value,
            "gain": round(float(self.gain), 6),
            "started_at": self.started_at,
            "seq": self.seq,
            "token": self.token,
            "reserved": self.reserved,
            "pending": self.pending.token if self.pending else None,
        }


@dataclass(frozen=True)
class PendingStart:
    """A request waiting for its victim's fade-out to reach zero."""

    token: str
    category: str
    priority: int
    started_at: int
    seq: int
    fade_in_frames: int


@dataclass(frozen=True)
class Assignment:
    """Outcome of a voice request."""

    status: str  # started | queued_after_fade | dropped
    token: str
    voice: int | None = None
    victim: int | None = None
    starts_at: int | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "dropped"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "token": self.token,
            "voice": self.voice,
            "victim": self.victim,
            "starts_at": self.starts_at,
            "reason": self.reason,
        }


class CallbackClock:
    """Frame clock owned by the audio callback.

    The clock is frozen while ``paused`` is true.  Pausing and resuming may
    happen on the game thread, rendering on the callback thread, so the frame
    counter is guarded by a small lock.  Because :meth:`advance` is a no-op
    while paused, every consumer that measures time in mixer frames (fades,
    start times, pending starts) automatically stops with playback instead of
    running ahead and jumping on resume.
    """

    __slots__ = ("_frame", "_paused", "_lock")

    def __init__(self, frame: int = 0, paused: bool = False) -> None:
        self._frame = int(frame)
        self._paused = bool(paused)
        self._lock = threading.Lock()

    @property
    def frame(self) -> int:
        with self._lock:
            return self._frame

    @property
    def now(self) -> int:
        return self.frame

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def advance(self, frames: int) -> int:
        """Advance by ``frames`` rendered frames; a no-op while paused."""
        self.step(frames)
        return self.frame

    def step(self, frames: int) -> int:
        """Advance and return the number of frames actually applied.

        Returns ``0`` while paused, which is what the mixer uses to decide how
        far to move envelopes this block.
        """
        frames = int(frames)
        with self._lock:
            if self._paused or frames <= 0:
                return 0
            self._frame += frames
            return frames

    def pause(self) -> "CallbackClock":
        with self._lock:
            self._paused = True
        return self

    def resume(self) -> "CallbackClock":
        with self._lock:
            self._paused = False
        return self

    def set_paused(self, paused: bool) -> "CallbackClock":
        with self._lock:
            self._paused = bool(paused)
        return self

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"frame": self._frame, "paused": self._paused}


class AtomicParamStore:
    """Single-writer / single-reader parameter store.

    The game thread calls :meth:`stage` (possibly many times) and then
    :meth:`commit`; the audio callback calls :meth:`snapshot` once per block
    and uses that one immutable object for the whole block.  A commit builds a
    brand-new mapping and swaps the reference, so a reader either sees the
    complete old parameter set or the complete new one -- never a mixture.

    ``snapshot()`` itself takes no lock: reading an object reference is atomic
    on CPython, and the returned :class:`~types.MappingProxyType` cannot be
    mutated by anyone.  Only writers serialise on the lock.
    """

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._write_lock = threading.Lock()
        self._pending: dict[str, Any] = {}
        self._snapshot: Mapping[str, Any] = MappingProxyType(dict(initial or {}))
        self._version = 0

    # -- game thread -----------------------------------------------------
    def stage(self, changes: Mapping[str, Any] | None = None, **kwargs: Any) -> "AtomicParamStore":
        with self._write_lock:
            if changes:
                self._pending.update(changes)
            self._pending.update(kwargs)
        return self

    def commit(self) -> Mapping[str, Any]:
        """Publish all staged values as one atomic snapshot."""
        with self._write_lock:
            if self._pending:
                new_snapshot = dict(self._snapshot)
                new_snapshot.update(self._pending)
                self._pending = {}
                self._version += 1
                self._snapshot = MappingProxyType(new_snapshot)
            return self._snapshot

    def discard(self) -> "AtomicParamStore":
        with self._write_lock:
            self._pending = {}
        return self

    # -- audio callback --------------------------------------------------
    def snapshot(self) -> Mapping[str, Any]:
        """Return the currently published immutable snapshot (lock-free)."""
        return self._snapshot

    def read(self, key: str, default: Any = None) -> Any:
        return self._snapshot.get(key, default)

    @property
    def version(self) -> int:
        return self._version

    def as_dict(self) -> dict[str, Any]:
        return {"version": self._version, "values": dict(self.snapshot())}


@dataclass(frozen=True)
class DegradePlan:
    """Deterministic plan for a given mixer utilisation."""

    level: int
    utilisation: float
    muted: tuple[str, ...] = ()
    ducked: tuple[str, ...] = ()
    max_voices: int | None = None
    protect_reserved: bool = True
    actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "utilisation": round(self.utilisation, 6),
            "muted": list(self.muted),
            "ducked": list(self.ducked),
            "max_voices": self.max_voices,
            "protect_reserved": self.protect_reserved,
            "actions": list(self.actions),
        }


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PopCheck:
    safe: bool
    max_step: float
    worst_voice: int | None = None
    needed_ramp_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "max_step": round(self.max_step, 6),
            "worst_voice": self.worst_voice,
            "needed_ramp_frames": self.needed_ramp_frames,
        }


@dataclass(frozen=True)
class DeterminismReport:
    equal: bool
    digest: str
    runs: int

    def to_dict(self) -> dict[str, Any]:
        return {"equal": self.equal, "digest": self.digest, "runs": self.runs}


def _norm_category(category: Any) -> str:
    if isinstance(category, Category):
        return category.value
    return str(category)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _default_reserved(total: int) -> int:
    if total < 8:
        return 0
    return min(4, max(1, total // 8))


def _merge_priorities(overrides: Mapping[str, Any] | None) -> dict[str, int]:
    merged = dict(DEFAULT_PRIORITIES)
    if overrides:
        for key, value in overrides.items():
            merged[_norm_category(key)] = int(value)
    return merged


@dataclass(frozen=True)
class _Command:
    seq: int
    kind: str
    category: str
    priority: int
    token: str
    fade_in_frames: int
    fade_out_frames: int


class VoicePool:
    """Deterministic voice allocator.

    Both the game thread and the audio callback mutate the pool only while
    holding ``_lock``.  Game-thread calls enqueue a :class:`_Command` first;
    commands are drained in submission (``seq``) order at the top of every
    block, so request handling itself is deterministic even though threads
    arrive in arbitrary order.

    The callback's per-block entry point is :meth:`advance`, which drains
    commands and then moves every envelope forward by the same number of
    frames.
    """

    def __init__(
        self,
        voices: int = 32,
        priorities: Mapping[str, Any] | None = None,
        reserved: int | None = None,
        sample_rate: int = 48_000,
        fade_ms: float = 8.0,
        fade_out_ms: float | None = None,
        protected_categories: Iterable[str] | None = None,
        clock: CallbackClock | None = None,
    ) -> None:
        self.total = max(1, _coerce_int(voices, 32))
        self.sample_rate = max(1, _coerce_int(sample_rate, 48_000))
        self.priorities = _merge_priorities(priorities)
        if reserved is None:
            reserved = _default_reserved(self.total)
        self.reserved = max(0, min(_coerce_int(reserved), self.total))
        self.protected_categories = {
            _norm_category(c)
            for c in (
                protected_categories
                if protected_categories is not None
                else (Category.DIALOGUE.value, Category.CRITICAL.value)
            )
        }
        self.fade_in_frames = max(1, int(self.sample_rate * float(fade_ms) / 1000.0))
        self.fade_out_frames = max(
            1,
            int(self.sample_rate * float(
                fade_ms if fade_out_ms is None else fade_out_ms
            ) / 1000.0),
        )
        self.clock = clock if clock is not None else CallbackClock()
        self.voices: list[Voice] = [Voice(index=i) for i in range(self.total)]
        self._reserved_indices = tuple(
            range(self.total - self.reserved, self.total)
        ) if self.reserved else ()
        self._reserved_set = set(self._reserved_indices)
        self._lock = threading.RLock()
        self._commands: list[_Command] = []
        self._results: dict[int, Assignment] = {}
        self._next_seq = 0
        self._event_limit = 1024
        self.stolen = 0
        self.dropped = 0
        self.started = 0
        self._events: list[dict[str, Any]] = []
        self._last_gain: dict[int, float] = {v.index: 0.0 for v in self.voices}
        self.max_gain_step = 0.0
        self.pop_violations: list[dict[str, Any]] = []
        self.overload_level = 0

    # -- introspection ---------------------------------------------------
    @property
    def reserved_indices(self) -> tuple[int, ...]:
        return self._reserved_indices

    def is_reserved(self, index: int) -> bool:
        return index in self._reserved_set

    def priority_for(self, category: Any) -> int:
        return int(self.priorities.get(_norm_category(category), 0))

    def is_protected(self, category: Any) -> bool:
        name = _norm_category(category)
        return (
            name in self.protected_categories
            or self.priority_for(name) >= PROTECTED_PRIORITY
        )

    def _voice_allowed(self, index: int, protected: bool) -> bool:
        return protected or index not in self._reserved_set

    # -- command submission (game thread) --------------------------------
    def submit(
        self,
        category: Any = Category.SFX.value,
        token: str | None = None,
        priority: int | None = None,
        fade_in_frames: int | None = None,
        fade_out_frames: int | None = None,
    ) -> int:
        name = _norm_category(category)
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            self._commands.append(
                _Command(
                    seq=seq,
                    kind="start",
                    category=name,
                    priority=self.priority_for(name) if priority is None else int(priority),
                    token=token if token is not None else f"{name}#{seq}",
                    fade_in_frames=(
                        self.fade_in_frames
                        if fade_in_frames is None
                        else max(0, int(fade_in_frames))
                    ),
                    fade_out_frames=(
                        self.fade_out_frames
                        if fade_out_frames is None
                        else max(0, int(fade_out_frames))
                    ),
                )
            )
            return seq

    def request(self, *args: Any, **kwargs: Any) -> Assignment:
        """Submit a request and resolve it immediately (test/game convenience)."""
        seq = self.submit(*args, **kwargs)
        with self._lock:
            self._drain()
            return self._results.pop(seq, None) or Assignment(
                status="dropped", token=str(seq), reason="unresolved"
            )

    def _drain(self) -> list[Assignment]:
        if not self._commands:
            return []
        commands = sorted(self._commands, key=lambda c: c.seq)
        self._commands = []
        assignments = []
        for command in commands:
            result = self._apply(command)
            self._results[command.seq] = result
            assignments.append(result)
        # Results exist for request() callers to collect; do not retain an
        # unbounded history when callers use submit() directly.
        if len(self._results) > self._event_limit:
            for stale in sorted(self._results)[: len(self._results) - self._event_limit]:
                del self._results[stale]
        return assignments

    # -- allocation ------------------------------------------------------
    def _find_free(self, protected: bool) -> int | None:
        allowed = [
            v.index
            for v in self.voices
            if v.state is VoiceState.FREE and self._voice_allowed(v.index, protected)
        ]
        if not allowed:
            return None
        if protected:
            reserved = [i for i in allowed if i in self._reserved_set]
            if reserved:
                return reserved[0]
        return allowed[0]

    def _steal_candidates(self, protected: bool, category: str, priority: int, seq: int) -> list[Voice]:
        candidates: list[Voice] = []
        for voice in self.voices:
            if voice.state not in (VoiceState.ACTIVE, VoiceState.FADING_IN):
                continue
            if voice.pending is not None:
                continue
            if not self._voice_allowed(voice.index, protected):
                continue
            if voice.priority < priority:
                candidates.append(voice)
            elif (
                voice.priority == priority
                and voice.category == category
                and voice.seq < seq
            ):
                candidates.append(voice)
        return candidates

    def _victim_key(self, voice: Voice) -> tuple[int, int, int, int]:
        return (voice.priority, voice.started_at, voice.seq, voice.index)

    def choose_victim(
        self,
        category: Any = Category.SFX.value,
        priority: int | None = None,
        seq: int | None = None,
        protected: bool | None = None,
    ) -> Voice | None:
        name = _norm_category(category)
        prio = self.priority_for(name) if priority is None else int(priority)
        if protected is None:
            protected = self.is_protected(name)
        if seq is None:
            seq = self._next_seq
        candidates = self._steal_candidates(protected, name, prio, seq)
        if not candidates:
            return None
        return min(candidates, key=self._victim_key)

    def _apply(self, cmd: _Command) -> Assignment:
        protected = self.is_protected(cmd.category)
        free = self._find_free(protected)
        if free is not None:
            self._start(free, cmd, pending=None)
            return Assignment(
                status="started",
                token=cmd.token,
                voice=free,
                starts_at=self.clock.frame,
            )

        victim = self.choose_victim(
            cmd.category, priority=cmd.priority, seq=cmd.seq, protected=protected
        )
        if victim is not None:
            victim.state = VoiceState.FADING_OUT
            victim.fade_total = cmd.fade_out_frames
            victim.fade_elapsed = 0
            victim.pending = PendingStart(
                token=cmd.token,
                category=cmd.category,
                priority=cmd.priority,
                started_at=self.clock.frame + cmd.fade_out_frames,
                seq=cmd.seq,
                fade_in_frames=cmd.fade_in_frames,
            )
            self.stolen += 1
            self._record(
                stage="steal",
                seq=cmd.seq,
                token=cmd.token,
                voice=victim.index,
                victim=victim.token,
                fading_out_for=cmd.fade_out_frames,
                reason="priority" if victim.priority != cmd.priority else "older_same_category",
            )
            return Assignment(
                status="queued_after_fade",
                token=cmd.token,
                voice=victim.index,
                victim=victim.index,
                starts_at=victim.pending.started_at,
                reason="fade_out_before_reuse",
            )

        self.dropped += 1
        reason = (
            "all_free_voices_reserved"
            if protected is False and self._all_free_are_reserved()
            else "no_stealable_voice"
        )
        self._record(stage="drop", seq=cmd.seq, token=cmd.token, reason=reason)
        return Assignment(status="dropped", token=cmd.token, reason=reason)

    def _all_free_are_reserved(self) -> bool:
        free = [v for v in self.voices if v.state is VoiceState.FREE]
        return bool(free) and all(v.index in self._reserved_set for v in free)

    def _start(self, index: int, cmd: _Command, pending: PendingStart | None) -> None:
        voice = self.voices[index]
        after_fade = pending is not None
        if voice.gain != 0.0:
            self.pop_violations.append(
                {
                    "index": index,
                    "reason": "reuse_without_fade_complete",
                    "gain": voice.gain,
                }
            )
        voice.category = cmd.category
        voice.priority = cmd.priority
        voice.token = cmd.token
        voice.state = VoiceState.FADING_IN
        voice.gain = 0.0
        voice.fade_total = cmd.fade_in_frames
        voice.fade_elapsed = 0
        voice.started_at = self.clock.frame
        voice.seq = cmd.seq
        voice.reserved = index in self._reserved_set
        voice.pending = None
        self.started += 1
        self._last_gain[index] = 0.0
        self._record(
            stage="started",
            seq=cmd.seq,
            token=cmd.token,
            voice=index,
            reason="fade_complete" if after_fade else "free_voice",
        )

    # -- rendering (audio callback) --------------------------------------
    def advance(self, frames: int) -> int:
        """Render ``frames`` frames worth of envelope movement.

        Draining pending game-thread commands happens first, at the block
        boundary, so the callback never applies a half-updated request while
        it is iterating voices.

        The block is split at every fade-out completion boundary.  A large
        render block therefore cannot overshoot a fade: the old sound reaches
        exactly zero, the replacement is started at that exact frame, and the
        remainder of the block is spent on the new sound's fade-in.  This also
        keeps ``started_at`` exact, which matters for deterministic ordering.
        """
        frames = max(0, _coerce_int(frames))
        with self._lock:
            self._drain()
            if self.clock.paused:
                return 0
            done = 0
            remaining = frames
            guard = 0
            while remaining > 0:
                guard += 1
                if guard > len(self.voices) + 4:
                    break
                segment = remaining
                immediate = False
                for voice in self.voices:
                    if voice.state is not VoiceState.FADING_OUT:
                        continue
                    left = voice.fade_total - voice.fade_elapsed
                    if left <= 0:
                        immediate = True
                    elif left < segment:
                        segment = left
                if immediate:
                    for voice in self.voices:
                        if (
                            voice.state is VoiceState.FADING_OUT
                            and voice.fade_elapsed >= voice.fade_total
                        ):
                            voice.gain = 0.0
                            self._finish_fade_out(voice)
                    continue
                if segment <= 0:
                    break
                applied = self.clock.step(segment)
                if applied <= 0:
                    break
                for voice in self.voices:
                    self._step_voice(voice, applied)
                done += applied
                remaining -= applied
            return done

    def _step_voice(self, voice: Voice, frames: int) -> None:
        state = voice.state
        if state is VoiceState.FADING_IN:
            voice.fade_elapsed += frames
            if voice.fade_total <= 0 or voice.fade_elapsed >= voice.fade_total:
                voice.gain = 1.0
                voice.state = VoiceState.ACTIVE
            else:
                voice.gain = voice.fade_elapsed / voice.fade_total
        elif state is VoiceState.FADING_OUT:
            voice.fade_elapsed += frames
            if voice.fade_total <= 0 or voice.fade_elapsed >= voice.fade_total:
                voice.gain = 0.0
                self._finish_fade_out(voice)
            else:
                voice.gain = 1.0 - (voice.fade_elapsed / voice.fade_total)
        self._track_gain(voice)

    def _finish_fade_out(self, voice: Voice) -> None:
        pending = voice.pending
        if pending is not None:
            # The old sound is exactly silent here; only now may the new one
            # begin.  Envelopes therefore never overlap on this voice.
            voice.pending = None
            voice.state = VoiceState.FREE
            voice.token = None
            cmd = _Command(
                seq=pending.seq,
                kind="start",
                category=pending.category,
                priority=pending.priority,
                token=pending.token,
                fade_in_frames=pending.fade_in_frames,
                fade_out_frames=self.fade_out_frames,
            )
            self._start(voice.index, cmd, pending=pending)
            return
        voice.state = VoiceState.FREE
        voice.category = None
        voice.priority = -1
        voice.token = None
        voice.gain = 0.0
        voice.fade_total = 0
        voice.fade_elapsed = 0
        voice.seq = 0
        voice.reserved = False

    def _track_gain(self, voice: Voice) -> None:
        prev = self._last_gain.get(voice.index, 0.0)
        step = abs(voice.gain - prev)
        if step > self.max_gain_step:
            self.max_gain_step = step
        self._last_gain[voice.index] = voice.gain

    def stop(self, index: int, fade_out_frames: int | None = None) -> None:
        """Fade a voice out and release it (without a replacement sound)."""
        with self._lock:
            voice = self.voices[index]
            if not voice.busy:
                return
            total = self.fade_out_frames if fade_out_frames is None else max(0, int(fade_out_frames))
            voice.state = VoiceState.FADING_OUT
            voice.fade_total = total
            voice.fade_elapsed = 0
            if voice.pending is not None:
                # A replacement was queued behind the old fade, but the caller
                # is now stopping the voice outright.  Report the cancellation
                # instead of silently dropping the request.
                self.dropped += 1
                self._record(
                    stage="drop",
                    token=voice.pending.token,
                    voice=index,
                    reason="cancelled_by_stop",
                )
                voice.pending = None

    # -- reporting -------------------------------------------------------
    def _record(self, **event: Any) -> None:
        event["frame"] = self.clock.frame
        self._events.append(event)
        # The callback runs for the lifetime of the session; keep the history
        # bounded so auditing never leaks memory.
        if len(self._events) > self._event_limit:
            del self._events[: len(self._events) - self._event_limit]

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def utilisation(self) -> float:
        busy = sum(1 for v in self.voices if v.busy)
        return busy / self.total

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": self.total,
                "reserved": self.reserved,
                "reserved_indices": list(self._reserved_indices),
                "priorities": dict(self.priorities),
                "clock": self.clock.as_dict(),
                "voices": [v.as_dict() for v in self.voices],
                "stolen": self.stolen,
                "dropped": self.dropped,
                "started": self.started,
                "utilisation": round(self.utilisation(), 6),
                "pop_violations": list(self.pop_violations),
            }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        busy = sum(1 for v in self.voices if v.busy)
        return f"VoicePool(total={self.total}, busy={busy}, reserved={self.reserved})"


# ---------------------------------------------------------------------------
# Public contract helpers
# ---------------------------------------------------------------------------


def priority_map(*args: Any, **kwargs: Any) -> dict[str, int]:
    """Return a category -> priority mapping.

    Accepts an optional override mapping (``priority_map({"sfx": 30})``) or a
    single ``priority_map("sfx", 30)`` pair.  Unrecognised positional values
    are ignored so the helper is safe to introspect.
    """
    overrides: dict[str, Any] = {}
    if args:
        first = args[0]
        if isinstance(first, _MappingABC):
            overrides.update(first)
        elif len(args) >= 2 and isinstance(args[1], (int, float)):
            overrides[_norm_category(first)] = args[1]
    overrides.update(kwargs)
    return _merge_priorities(overrides)


def reserved_voice(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Describe the reserved-channel split for a pool or a voice count."""
    if args and isinstance(args[0], VoicePool):
        pool = args[0]
        return {
            "total": pool.total,
            "reserved": pool.reserved,
            "usable": pool.total - pool.reserved,
            "reserved_indices": list(pool.reserved_indices),
            "protected_categories": sorted(pool.protected_categories),
            "invariant": "unprotected categories never occupy or steal reserved voices",
        }
    total = _coerce_int(args[0], 32) if args else 32
    total = max(1, total)
    explicit = args[1] if len(args) >= 2 else kwargs.get("reserved")
    reserved = _default_reserved(total) if explicit is None else _coerce_int(explicit)
    reserved = max(0, min(reserved, total))
    indices = list(range(total - reserved, total)) if reserved else []
    return {
        "total": total,
        "reserved": reserved,
        "usable": total - reserved,
        "reserved_indices": indices,
        "protected_categories": [Category.DIALOGUE.value, Category.CRITICAL.value],
        "invariant": "unprotected categories never occupy or steal reserved voices",
    }


def atomic_param(*args: Any, **kwargs: Any) -> AtomicParamStore:
    """Create (or return) an atomic parameter store.

    ``atomic_param({"volume": 0.8})`` seeds the snapshot.  Passing an existing
    store is a no-op.  The returned object exposes ``stage``/``commit`` on the
    writer side and a lock-free ``snapshot`` on the reader side.
    """
    if args and isinstance(args[0], AtomicParamStore):
        return args[0]
    initial: dict[str, Any] = {}
    if args and isinstance(args[0], _MappingABC):
        initial.update(args[0])
    initial.update(kwargs)
    return AtomicParamStore(initial)


def fade_before_reuse(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Report pending reuses and the fade ordering invariant."""
    if args and isinstance(args[0], VoicePool):
        pool = args[0]
        pending = [
            {
                "voice": v.index,
                "previous": v.token,
                "next": v.pending.token,
                "next_starts_at": v.pending.started_at,
                "fade_out_frames": v.fade_total,
                "fade_elapsed": v.fade_elapsed,
            }
            for v in pool.voices
            if v.state is VoiceState.FADING_OUT and v.pending is not None
        ]
        return {
            "pending": pending,
            "fade_out_frames": pool.fade_out_frames,
            "fade_in_frames": pool.fade_in_frames,
            "invariant": "reuse only after envelope reaches exactly 0",
        }
    fade_in = _coerce_int(args[0], 384) if args else 384
    fade_out = _coerce_int(args[1], fade_in) if len(args) >= 2 else fade_in
    return {
        "pending": [],
        "fade_out_frames": max(1, fade_out),
        "fade_in_frames": max(1, fade_in),
        "invariant": "reuse only after envelope reaches exactly 0",
    }


def clock_pause(*args: Any, **kwargs: Any) -> CallbackClock:
    """Return a callback clock, optionally paused.

    ``clock_pause(clock)`` pauses an existing clock; ``clock_pause(True)``
    returns a new paused clock.  Because envelopes advance on this clock, a
    paused clock freezes them and resuming cannot skip frames.
    """
    if args and isinstance(args[0], CallbackClock):
        clock = args[0]
        paused = args[1] if len(args) >= 2 else kwargs.get("paused", True)
        clock.set_paused(bool(paused))
        return clock
    paused = False
    frame = 0
    if args:
        first = args[0]
        if isinstance(first, bool):
            paused = first
        else:
            frame = _coerce_int(first, 0)
    return CallbackClock(frame=frame, paused=paused)


def _voice_sort_key(item: Any) -> tuple[int, int, int, int]:
    if isinstance(item, Voice):
        return (item.priority, item.started_at, item.seq, item.index)
    if isinstance(item, _MappingABC):
        return (
            _coerce_int(item.get("priority"), -1),
            _coerce_int(item.get("started_at"), 0),
            _coerce_int(item.get("seq"), 0),
            _coerce_int(item.get("index"), 0),
        )
    return (0, 0, 0, 0)


def steal_order(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Return the deterministic victim order (lowest priority, oldest first)."""
    if args and isinstance(args[0], VoicePool):
        pool = args[0]
        victims = [
            v
            for v in pool.voices
            if v.state in (VoiceState.ACTIVE, VoiceState.FADING_IN) and v.pending is None
        ]
        return [v.as_dict() for v in sorted(victims, key=_voice_sort_key)]
    if args and isinstance(args[0], _IterableABC) and not isinstance(args[0], (str, bytes)):
        items = list(args[0])
        return [
            item.as_dict() if isinstance(item, Voice) else dict(item)
            for item in sorted(items, key=_voice_sort_key)
        ]
    # No concrete voices: describe the default category order in which the
    # mixer would sacrifice them.
    return [
        {"category": category, "priority": priority}
        for category, priority in sorted(DEFAULT_PRIORITIES.items(), key=lambda kv: kv[1])
    ]


def overload_degrade(*args: Any, **kwargs: Any) -> DegradePlan:
    """Map utilisation to a deterministic degradation plan."""
    if args and isinstance(args[0], VoicePool):
        utilisation = args[0].utilisation()
        max_voices = args[0].total
    else:
        raw = args[0] if args else kwargs.get("utilisation", 0.0)
        try:
            utilisation = float(raw)
        except (TypeError, ValueError):
            utilisation = 0.0
        max_voices = _coerce_int(kwargs.get("max_voices"), 32)
    utilisation = max(0.0, utilisation)

    if utilisation < 0.75:
        return DegradePlan(level=0, utilisation=utilisation, actions=("none",))
    if utilisation < 0.90:
        return DegradePlan(
            level=1,
            utilisation=utilisation,
            ducked=(Category.MUSIC.value,),
            actions=("duck_music",),
        )
    if utilisation < 1.0:
        return DegradePlan(
            level=2,
            utilisation=utilisation,
            muted=(Category.AMBIENT.value,),
            ducked=(Category.MUSIC.value,),
            actions=("mute_ambient", "duck_music"),
        )
    return DegradePlan(
        level=3,
        utilisation=utilisation,
        muted=(Category.AMBIENT.value, Category.MUSIC.value),
        ducked=(Category.SFX.value,),
        max_voices=max_voices,
        protect_reserved=True,
        actions=("mute_ambient", "mute_music", "duck_sfx", "protect_dialogue"),
    )


def validation(*args: Any, **kwargs: Any) -> ValidationReport:
    """Check a pool or a plain configuration for impossible settings."""
    errors: list[str] = []
    warnings: list[str] = []
    if args and isinstance(args[0], VoicePool):
        pool = args[0]
        total, reserved = pool.total, pool.reserved
        priorities = pool.priorities
        fade_in, fade_out = pool.fade_in_frames, pool.fade_out_frames
        protected = pool.protected_categories
    else:
        config = args[0] if args and isinstance(args[0], _MappingABC) else {}
        config = {**config, **kwargs}
        total = max(1, _coerce_int(config.get("total", config.get("voices", 32)), 32))
        reserved = _coerce_int(
            config.get("reserved", _default_reserved(total)), _default_reserved(total)
        )
        priorities = _merge_priorities(config.get("priorities"))
        fade_in = _coerce_int(config.get("fade_in_frames"), 384)
        fade_out = _coerce_int(config.get("fade_out_frames"), fade_in)
        protected = {Category.DIALOGUE.value, Category.CRITICAL.value}

    if reserved < 0:
        errors.append("reserved voices must be >= 0")
    if reserved >= total:
        errors.append("all voices reserved: unprotected categories can never play")
    if total <= 0:
        errors.append("total voices must be positive")
    if fade_in <= 0 or fade_out <= 0:
        errors.append("fade lengths must be positive to avoid clicks")
    for category, priority in priorities.items():
        if priority < 0:
            errors.append(f"negative priority for {category!r}")
    for category in protected:
        if priorities.get(category, 0) < PROTECTED_PRIORITY:
            warnings.append(
                f"protected category {category!r} has priority below "
                f"{PROTECTED_PRIORITY}; protection is enforced by category anyway"
            )
    if reserved == 0:
        warnings.append("no voices reserved; dialogue may compete with ambience")
    return ValidationReport(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Summarise allocation history and invariants for a pool."""
    if args and isinstance(args[0], VoicePool):
        pool = args[0]
        with pool._lock:
            voice_states: dict[str, int] = {}
            for voice in pool.voices:
                voice_states[voice.state.value] = voice_states.get(voice.state.value, 0) + 1
            steals = [e for e in pool._events if e.get("stage") == "steal"]
            drops = [e for e in pool._events if e.get("stage") == "drop"]
            return {
                "voices": pool.total,
                "reserved": pool.reserved,
                "states": voice_states,
                "started": pool.started,
                "stolen": pool.stolen,
                "dropped": pool.dropped,
                "steal_events": steals[-16:],
                "drop_events": drops[-16:],
                "clock": pool.clock.as_dict(),
                "pop_violations": len(pool.pop_violations),
                "deterministic_order": "(priority, started_at, seq, index)",
            }
    return {
        "voices": 0,
        "reserved": 0,
        "states": {},
        "started": 0,
        "stolen": 0,
        "dropped": 0,
        "steal_events": [],
        "drop_events": [],
        "clock": {"frame": 0, "paused": False},
        "pop_violations": 0,
        "deterministic_order": "(priority, started_at, seq, index)",
    }


def fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Give a deterministic fallback when no voice can be granted."""
    category = _norm_category(args[0]) if args else Category.SFX.value
    reason = kwargs.get("reason")
    if reason is None and len(args) >= 2:
        reason = args[1]
    reason = str(reason) if reason is not None else "no_stealable_voice"
    return {
        "category": category,
        "reason": reason,
        "action": "drop_newest",
        "protect_dialogue": True,
        "alternatives": ["reuse_lower_fidelity", "duck_ambient", "queue_until_free"],
        "note": "critical and dialogue categories are never sacrificed",
    }


def pop_guard(*args: Any, **kwargs: Any) -> PopCheck:
    """Check that a voice is never reused before its envelope hits zero."""
    if args and isinstance(args[0], VoicePool):
        pool = args[0]
        with pool._lock:
            violations = list(pool.pop_violations)
            return PopCheck(
                safe=not violations,
                max_step=pool.max_gain_step,
                worst_voice=violations[0]["index"] if violations else None,
                needed_ramp_frames=max(pool.fade_out_frames, pool.fade_in_frames),
            )
    max_step = 0.0
    worst: int | None = None
    if args and isinstance(args[0], _IterableABC) and not isinstance(args[0], (str, bytes)):
        previous: float | None = None
        for position, sample in enumerate(args[0]):
            try:
                value = float(sample)
            except (TypeError, ValueError):
                previous = None
                continue
            if previous is not None and abs(value - previous) > max_step:
                max_step = abs(value - previous)
                worst = position
            previous = value
    threshold = float(kwargs.get("max_step", 0.25))
    return PopCheck(
        safe=max_step <= threshold,
        max_step=max_step,
        worst_voice=worst,
        needed_ramp_frames=_coerce_int(kwargs.get("fade_frames"), 384),
    )


def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return mixer occupancy counters."""
    if args and isinstance(args[0], VoicePool):
        pool = args[0]
        with pool._lock:
            states: dict[str, int] = {s.value: 0 for s in VoiceState}
            for voice in pool.voices:
                states[voice.state.value] += 1
            return {
                "total": pool.total,
                "reserved": pool.reserved,
                "free": states[VoiceState.FREE.value],
                "active": states[VoiceState.ACTIVE.value],
                "fading_in": states[VoiceState.FADING_IN.value],
                "fading_out": states[VoiceState.FADING_OUT.value],
                "states": states,
                "utilisation": round(pool.utilisation(), 6),
                "stolen": pool.stolen,
                "dropped": pool.dropped,
                "started": pool.started,
                "frame": pool.clock.frame,
                "paused": pool.clock.paused,
            }
    return {
        "total": 0,
        "reserved": 0,
        "free": 0,
        "active": 0,
        "fading_in": 0,
        "fading_out": 0,
        "states": {},
        "utilisation": 0.0,
        "stolen": 0,
        "dropped": 0,
        "started": 0,
        "frame": 0,
        "paused": False,
    }


def voice_pool(*args: Any, **kwargs: Any) -> VoicePool:
    """Build a :class:`VoicePool` from positional or keyword configuration."""
    if args and isinstance(args[0], VoicePool):
        return args[0]
    total = 32
    priorities = None
    reserved = None
    remaining = list(args)
    if remaining and isinstance(remaining[0], _MappingABC):
        config = dict(remaining.pop(0))
        priorities = config.get("priorities")
        reserved = config.get("reserved", reserved)
        total = _coerce_int(config.get("voices", config.get("total", total)), total)
    if remaining:
        total = _coerce_int(remaining.pop(0), total)
    if remaining:
        reserved = _coerce_int(remaining.pop(0), 0)
    total = max(1, total)
    if "priorities" in kwargs:
        priorities = kwargs["priorities"]
    if "reserved" in kwargs:
        reserved = kwargs["reserved"]
    return VoicePool(
        voices=total,
        priorities=priorities,
        reserved=None if reserved is None else _coerce_int(reserved, 0),
        sample_rate=_coerce_int(kwargs.get("sample_rate"), 48_000),
        fade_ms=float(kwargs.get("fade_ms", 8.0)),
        fade_out_ms=kwargs.get("fade_out_ms"),
        protected_categories=kwargs.get("protected_categories"),
        clock=kwargs.get("clock"),
    )


def callback_state(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return the callback-visible state (clock, version, voices)."""
    if args and isinstance(args[0], VoicePool):
        pool = args[0]
        with pool._lock:
            return {
                "clock": pool.clock.as_dict(),
                "paused": pool.clock.paused,
                "voices": [v.as_dict() for v in pool.voices],
                "commands_pending": len(pool._commands),
                "utilisation": round(pool.utilisation(), 6),
            }
    if args and isinstance(args[0], CallbackClock):
        return args[0].as_dict()
    return {
        "clock": {"frame": 0, "paused": False},
        "paused": False,
        "voices": [],
        "commands_pending": 0,
        "utilisation": 0.0,
    }


def _default_scenario() -> dict[str, Any]:
    pool = VoicePool(voices=8, reserved=2)
    pool.request(Category.AMBIENT.value, token="amb-1")
    pool.request(Category.MUSIC.value, token="mus-1")
    for i in range(4):
        pool.request(Category.SFX.value, token=f"sfx-{i}")
    pool.request(Category.DIALOGUE.value, token="dlg-1")
    pool.advance(pool.fade_in_frames)
    for i in range(6):
        pool.request(Category.SFX.value, token=f"sfx-late-{i}")
    pool.advance(pool.fade_out_frames)
    pool.request(Category.CRITICAL.value, token="crit-1")
    pool.advance(128)
    return pool.snapshot()


def _digest(payload: Any) -> str:
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        encoded = repr(payload)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def determinism(*args: Any, **kwargs: Any) -> DeterminismReport:
    """Run the same scenario repeatedly and compare final-state digests."""
    scenario: Callable[[], Any] | None = None
    runs = 2
    if args:
        if callable(args[0]):
            scenario = args[0]
            if len(args) >= 2:
                runs = max(1, _coerce_int(args[1], 2))
        else:
            runs = max(1, _coerce_int(args[0], 2))
    runs = max(1, _coerce_int(kwargs.get("runs"), runs))

    digests: list[str] = []
    for _ in range(runs):
        payload = scenario() if scenario is not None else _default_scenario()
        if isinstance(payload, VoicePool):
            payload = payload.snapshot()
        digests.append(_digest(payload))
    unique = set(digests)
    return DeterminismReport(
        equal=len(unique) == 1,
        digest=digests[0] if digests else _digest({}),
        runs=runs,
    )

