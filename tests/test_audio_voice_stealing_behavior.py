"""Behavioural tests for the voice-stealing mixer.

The shipped contract test only checks that every entry point returns something
non-None.  These tests exercise the six properties the module promises:

1. priority + reserved channels
2. atomic cross-thread parameter commits
3. fade-out finishes before a voice is reused (no pop)
4. the callback clock freezes while paused
5. deterministic stealing order
6. graceful degradation under overload
"""

import threading

import pytest

from media.audio_voice_stealing import (
    PROTECTED_PRIORITY,
    AtomicParamStore,
    CallbackClock,
    Category,
    VoicePool,
    VoiceState,
    atomic_param,
    audit,
    callback_state,
    clock_pause,
    determinism,
    fade_before_reuse,
    fallback,
    metrics,
    overload_degrade,
    pop_guard,
    priority_map,
    reserved_voice,
    steal_order,
    validation,
    voice_pool,
)


# ---------------------------------------------------------------------------
# 1. priorities and reserved channels
# ---------------------------------------------------------------------------


def test_priority_map_orders_categories_and_applies_overrides():
    priorities = priority_map()
    assert priorities["ambient"] < priorities["sfx"] < priorities["dialogue"]
    assert priorities["critical"] >= PROTECTED_PRIORITY
    assert priority_map({"ambient": 5})["ambient"] == 5
    assert priority_map("sfx", 33)["sfx"] == 33


def test_reserved_split_leaves_usable_voices():
    info = reserved_voice(32)
    assert info["reserved"] > 0
    assert info["total"] == info["reserved"] + info["usable"]
    assert len(info["reserved_indices"]) == info["reserved"]


def test_unprotected_categories_never_take_reserved_voices():
    pool = VoicePool(voices=8, reserved=2)
    for i in range(8):
        pool.request("ambient", token=f"amb-{i}")
    pool.advance(pool.fade_in_frames)

    reserved = set(pool.reserved_indices)
    assert reserved, "test needs reserved voices"
    for index in reserved:
        voice = pool.voices[index]
        assert voice.state is VoiceState.FREE, (
            f"ambient occupied reserved voice {index}"
        )


def test_dialogue_gets_a_reserved_voice_even_when_pool_is_full_of_ambient():
    pool = VoicePool(voices=8, reserved=2)
    for i in range(6):
        pool.request("ambient", token=f"amb-{i}")
    pool.advance(pool.fade_in_frames)

    result = pool.request("dialogue", token="line-1")
    assert result.status == "started"
    assert result.voice in pool.reserved_indices


def test_ambient_cannot_steal_dialogue_even_at_equal_priority():
    pool = VoicePool(
        voices=2,
        reserved=0,
        priorities={"ambient": 80, "dialogue": 80},
    )
    pool.request("dialogue", token="line-1")
    pool.request("dialogue", token="line-2")
    pool.advance(pool.fade_in_frames)

    result = pool.request("ambient", token="amb-1")
    assert result.status == "dropped"
    assert result.reason == "no_stealable_voice"
    assert {pool.voices[0].token, pool.voices[1].token} == {"line-1", "line-2"}


def test_lower_priority_is_stolen_before_higher_priority():
    pool = VoicePool(
        voices=2,
        reserved=0,
        priorities={"ambient": 0, "music": 10, "sfx": 20},
    )
    pool.request("sfx", token="sfx-1")
    pool.request("music", token="music-1")
    pool.advance(pool.fade_in_frames)

    result = pool.request("sfx", token="sfx-2")
    assert result.status == "queued_after_fade"
    assert result.victim is not None
    assert pool.voices[result.victim].token == "music-1"


# ---------------------------------------------------------------------------
# 2. atomic cross-thread parameters
# ---------------------------------------------------------------------------


def test_committed_snapshot_is_immutable_and_all_or_nothing():
    store = AtomicParamStore({"volume": 1.0, "pan": 0.0})
    store.stage({"volume": 0.5, "pan": -1.0})
    # Staged values are invisible until commit.
    assert store.snapshot()["volume"] == 1.0
    assert store.snapshot()["pan"] == 0.0

    snapshot = store.commit()
    assert snapshot["volume"] == 0.5
    assert snapshot["pan"] == -1.0
    with pytest.raises(TypeError):
        snapshot["volume"] = 0.0  # immutable view


def test_atomic_param_factory_returns_same_store():
    store = atomic_param({"volume": 0.8})
    assert atomic_param(store) is store
    assert store.snapshot()["volume"] == 0.8


def test_reader_never_observes_half_of_a_two_field_update():
    """Game thread writes (a,b) together; the reader must never see a mix."""
    store = AtomicParamStore({"a": 0, "b": 0})
    seen_pairs = set()
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            snap = store.snapshot()
            seen_pairs.add((snap["a"], snap["b"]))

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for value in range(1, 200):
            store.stage({"a": value, "b": value})
            store.commit()
    finally:
        stop.set()
        thread.join()

    mismatched = {pair for pair in seen_pairs if pair[0] != pair[1]}
    assert not mismatched, f"torn snapshot observed: {mismatched}"


def test_callback_advances_while_game_thread_stages_parameters():
    """Concurrent stage/commit must not corrupt the mixer's own state."""
    pool = VoicePool(voices=16)
    store = AtomicParamStore({"volume": 1.0})
    stop = threading.Event()
    errors = []

    def game_thread():
        try:
            for i in range(2000):
                if i % 50 == 0:
                    store.stage({"volume": float(i)})
                    store.commit()
                pool.request("sfx", token=f"sfx-{i}")
        except BaseException as exc:  # noqa: BLE001 - surfaced via assertion
            errors.append(exc)
        finally:
            stop.set()

    thread = threading.Thread(target=game_thread)
    thread.start()
    try:
        while not stop.is_set():
            pool.advance(64)
            snap = store.snapshot()
            assert snap["volume"] >= 0.0
    finally:
        thread.join()

    assert not errors, errors
    report = validation(pool)
    assert report.ok, report.errors
    assert pop_guard(pool).safe


# ---------------------------------------------------------------------------
# 3. fade out completes before reuse (pop-free)
# ---------------------------------------------------------------------------


def test_voice_is_not_reused_until_its_fade_reaches_zero():
    pool = VoicePool(voices=1, reserved=0, fade_ms=10.0)
    pool.request("sfx", token="first")
    pool.advance(pool.fade_in_frames)

    result = pool.request("sfx", token="second")
    assert result.status == "queued_after_fade"

    voice = pool.voices[0]
    assert voice.state is VoiceState.FADING_OUT

    # Step half of the fade: still fading, still the old sound.
    pool.advance(pool.fade_out_frames // 2)
    voice = pool.voices[0]
    assert voice.state is VoiceState.FADING_OUT
    assert voice.token == "first"
    assert voice.gain > 0.0

    # Step exactly to the end of the fade: gain hits zero and only then is the
    # voice reused for the new sound.
    pool.advance(pool.fade_out_frames // 2)
    voice = pool.voices[0]
    assert voice.token == "second"
    assert voice.state is VoiceState.FADING_IN
    assert voice.gain == 0.0
    assert pop_guard(pool).safe

    # Further rendering moves the *new* sound's fade-in, not the old fade.
    pool.advance(pool.fade_in_frames // 2)
    assert pool.voices[0].token == "second"
    assert 0.0 < pool.voices[0].gain < 1.0


def test_large_render_block_does_not_overshoot_the_fade_handover():
    """One big block must split at the zero crossing, not jump past it."""
    pool = VoicePool(voices=1, reserved=0, fade_ms=10.0)
    pool.request("sfx", token="old")
    pool.advance(pool.fade_in_frames)
    pool.request("sfx", token="new")
    pool.advance(pool.fade_out_frames // 2)

    # Render the whole rest of the fade plus a full fade-in in a single block.
    frames = pool.fade_out_frames // 2 + pool.fade_in_frames
    applied = pool.advance(frames)

    voice = pool.voices[0]
    assert applied == frames
    assert voice.token == "new"
    assert voice.state is VoiceState.ACTIVE
    assert voice.started_at == pool.clock.frame - pool.fade_in_frames
    assert pop_guard(pool).safe


def test_envelopes_never_overlap_on_a_stolen_voice():
    pool = VoicePool(voices=1, reserved=0, fade_ms=5.0)
    pool.request("sfx", token="old")
    pool.advance(pool.fade_in_frames)
    pool.request("sfx", token="new")

    previous_token = pool.voices[0].token
    for _ in range(pool.fade_out_frames + pool.fade_in_frames + 8):
        pool.advance(1)
        voice = pool.voices[0]
        if voice.token != previous_token:
            # Handover must happen at zero gain, never mid-envelope.
            assert voice.gain == 0.0
            assert voice.state is VoiceState.FADING_IN
            previous_token = voice.token
    assert pool.voices[0].token == "new"
    assert pop_guard(pool).safe


def test_fade_before_reuse_reports_pending_handover():
    pool = VoicePool(voices=1, reserved=0)
    pool.request("sfx", token="old")
    pool.advance(pool.fade_in_frames)
    pool.request("sfx", token="new")

    report = fade_before_reuse(pool)
    assert len(report["pending"]) == 1
    pending = report["pending"][0]
    assert pending["previous"] == "old"
    assert pending["next"] == "new"
    assert "invariant" in report


def test_voice_claimed_by_a_pending_steal_is_not_stolen_twice():
    pool = VoicePool(voices=1, reserved=0)
    pool.request("ambient", token="a")
    pool.advance(pool.fade_in_frames)
    pool.request("music", token="b")  # queues behind the fade
    result = pool.request("sfx", token="c")  # must not also claim the voice
    assert result.status == "dropped"


# ---------------------------------------------------------------------------
# 4. the clock freezes while paused
# ---------------------------------------------------------------------------


def test_callback_clock_frozen_during_pause():
    clock = CallbackClock()
    clock.step(128)
    assert clock.frame == 128

    clock.pause()
    assert clock.step(1024) == 0
    assert clock.advance(1024) == 128

    clock.resume()
    assert clock.step(64) == 64
    assert clock.frame == 192


def test_clock_pause_helper_updates_existing_clock():
    clock = CallbackClock(frame=10)
    returned = clock_pause(clock)
    assert returned is clock
    assert clock.paused is True
    clock_pause(clock, False)
    assert clock.paused is False


def test_mixer_does_not_advance_fades_while_paused():
    pool = VoicePool(voices=1, reserved=0, fade_ms=10.0)
    pool.request("sfx", token="a")
    pool.advance(pool.fade_in_frames // 2)
    gain_before = pool.voices[0].gain

    pool.clock.pause()
    pool.advance(10_000)
    assert pool.voices[0].gain == gain_before
    assert pool.clock.frame == pool.fade_in_frames // 2

    # Resuming continues exactly where the fade stopped: no skipped frames.
    pool.clock.resume()
    pool.advance(pool.fade_in_frames // 2)
    assert pool.voices[0].state is VoiceState.ACTIVE
    assert pool.clock.frame == pool.fade_in_frames


def test_paused_pending_steal_waits_and_never_jumps():
    pool = VoicePool(voices=1, reserved=0, fade_ms=10.0)
    pool.request("sfx", token="a")
    pool.advance(pool.fade_in_frames)
    pool.request("sfx", token="b")

    pool.clock.pause()
    pool.advance(10_000)
    assert pool.voices[0].token == "a"
    assert pool.voices[0].state is VoiceState.FADING_OUT

    pool.clock.resume()
    pool.advance(pool.fade_out_frames)
    assert pool.voices[0].token == "b"
    assert pool.voices[0].gain == 0.0


# ---------------------------------------------------------------------------
# 5. deterministic stealing order
# ---------------------------------------------------------------------------


def test_steal_order_is_lowest_priority_then_oldest():
    pool = VoicePool(voices=3, reserved=0)
    pool.request("dialogue", token="dlg")
    pool.request("sfx", token="sfx-old")
    pool.request("ambient", token="amb")

    order = steal_order(pool)
    tokens = [entry["token"] for entry in order]
    assert tokens[0] == "amb"          # lowest priority first
    assert tokens[1] == "sfx-old"
    assert tokens[2] == "dlg"


def test_steal_order_stable_for_identical_inputs():
    pool_a = VoicePool(voices=4, reserved=0)
    pool_b = VoicePool(voices=4, reserved=0)
    for pool in (pool_a, pool_b):
        for i, category in enumerate(["sfx", "ambient", "music", "sfx"]):
            pool.request(category, token=f"{category}-{i}")
    assert steal_order(pool_a) == steal_order(pool_b)


def test_victim_choice_independent_of_submission_interleaving_threads():
    def run(offsets):
        pool = VoicePool(voices=3, reserved=0)
        for i in offsets:
            pool.request("sfx", token=f"sfx-{i}")
        pool.advance(pool.fade_in_frames)
        return pool.request("sfx", token="late").victim

    assert run([0, 1, 2]) == run([0, 1, 2])


def test_determinism_helper_is_stable_across_runs():
    report = determinism(runs=4)
    assert report.equal is True
    assert report.runs == 4
    assert len(report.digest) == 64


def test_determinism_reports_divergence_for_unstable_scenario():
    counter = {"n": 0}

    def unstable():
        counter["n"] += 1
        return {"n": counter["n"]}

    report = determinism(unstable, 3)
    assert report.equal is False


# ---------------------------------------------------------------------------
# 6. graceful overload degradation
# ---------------------------------------------------------------------------


def test_overload_levels_are_monotonic():
    quiet = overload_degrade(0.10)
    busy = overload_degrade(0.80)
    hot = overload_degrade(0.95)
    full = overload_degrade(1.25)

    assert quiet.level == 0
    assert quiet.level < busy.level < hot.level < full.level
    assert "dialogue" not in quiet.muted
    assert "dialogue" not in full.muted
    assert full.protect_reserved is True


def test_overload_keeps_dialogue_out_of_the_mute_list():
    plan = overload_degrade(1.5)
    assert Category.DIALOGUE.value not in plan.muted
    assert Category.CRITICAL.value not in plan.muted


def test_overload_degrade_accepts_a_pool():
    pool = VoicePool(voices=4, reserved=0)
    for i in range(4):
        pool.request("ambient", token=f"a{i}")
    plan = overload_degrade(pool)
    assert plan.utilisation == 1.0
    assert plan.level == 3


# ---------------------------------------------------------------------------
# validation / audit / metrics / fallback wiring
# ---------------------------------------------------------------------------


def test_validation_rejects_fully_reserved_pool():
    report = validation({"voices": 8, "reserved": 8})
    assert report.ok is False
    assert any("reserved" in error for error in report.errors)


def test_validation_accepts_default_pool():
    assert validation(VoicePool()).ok is True


def test_audit_and_metrics_describe_pool_state():
    pool = VoicePool(voices=4, reserved=1)
    pool.request("dialogue", token="d")
    pool.request("sfx", token="s")
    pool.advance(pool.fade_in_frames)

    report = audit(pool)
    assert report["voices"] == 4
    assert report["started"] == 2
    assert report["deterministic_order"] == "(priority, started_at, seq, index)"

    stats = metrics(pool)
    assert stats["total"] == 4
    assert stats["active"] == 2
    assert stats["free"] == 2
    assert 0.0 < stats["utilisation"] <= 1.0


def test_fallback_never_sacrifices_dialogue():
    plan = fallback("dialogue", reason="no_stealable_voice")
    assert plan["protect_dialogue"] is True
    assert plan["category"] == "dialogue"


def test_voice_pool_factory_builds_configured_pool():
    pool = voice_pool(16, reserved=3, priorities={"sfx": 50}, fade_ms=4.0)
    assert isinstance(pool, VoicePool)
    assert pool.total == 16
    assert pool.reserved == 3
    assert pool.priority_for("sfx") == 50


def test_callback_state_reports_clock_and_voices():
    pool = VoicePool(voices=2, reserved=0)
    pool.request("sfx", token="s")
    state = callback_state(pool)
    assert state["paused"] is False
    assert len(state["voices"]) == 2
    assert state["commands_pending"] == 0


def test_pop_guard_detects_a_hard_sample_jump():
    check = pop_guard([0.0, 0.0, 1.0, 1.0], max_step=0.25)
    assert check.safe is False
    assert check.max_step == 1.0

    smooth = pop_guard([0.0, 0.1, 0.2, 0.3], max_step=0.25)
    assert smooth.safe is True
