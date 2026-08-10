from linguaeval.adapters.dataset.registry import get_adapter, list_adapters


def test_builtin_adapters_registered():
    names = list_adapters()
    assert "jsonl" in names
    assert "n2s_dialogue_prediction" in names
    assert get_adapter("jsonl") is not None
    assert get_adapter("n2s_dialogue_prediction") is not None


def test_unknown_adapter_raises():
    try:
        get_adapter("not_a_real_adapter")
        assert False, "expected KeyError"
    except KeyError as e:
        assert "not_a_real_adapter" in str(e)
