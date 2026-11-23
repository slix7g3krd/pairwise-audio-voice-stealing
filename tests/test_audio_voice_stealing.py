import pytest
import media.audio_voice_stealing as mod

CASES = [
    ("priority_map", (1)),
    ("reserved_voice", (1)),
    ("atomic_param", (1)),
    ("fade_before_reuse", (1)),
    ("clock_pause", (1)),
    ("steal_order", (1)),
    ("overload_degrade", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("pop_guard", (1)),
    ("metrics", (1)),
    ("voice_pool", (1)),
    ("callback_state", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
