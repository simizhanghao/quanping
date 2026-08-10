from linguaeval.core.schema import FormatStatus, OutputSpec, PredictionRecord
from linguaeval.parse.pipeline import apply_output_spec


def test_from_raw_json_ok_and_fail():
    output = OutputSpec.from_dict(
        {
            "parser": "json",
            "schema": {"required": ["amount", "time"], "types": {"amount": "number"}},
            "constraints": {"no_markdown": True},
        }
    )
    preds = [
        PredictionRecord(
            "a",
            "m",
            raw_output='{"amount": 500000, "time": "kemarin"}',
            parsed={},
            format=FormatStatus(),
        ),
        PredictionRecord(
            "b",
            "m",
            raw_output="{not json",
            parsed={},
            format=FormatStatus(),
        ),
        PredictionRecord(
            "c",
            "m",
            raw_output='{"amount": "500000", "time": "kemarin"}',
            parsed={},
            format=FormatStatus(),
        ),
    ]
    out = apply_output_spec(preds, output, mode="from_raw")
    assert out[0].format.parse_ok and out[0].format.schema_ok
    assert out[0].parsed["amount"] == 500000
    assert not out[1].format.parse_ok
    assert out[2].format.parse_ok and not out[2].format.schema_ok


def test_from_parsed_keeps_legacy_without_raw():
    output = OutputSpec.from_dict(
        {"parser": "json", "schema": {"required": ["n2s"], "types": {"n2s": "boolean"}}}
    )
    preds = [
        PredictionRecord(
            "x",
            "m",
            raw_output=None,
            parsed={"n2s": True, "routing_skill": "banking", "primary_intent": ""},
            format=FormatStatus(True, True),
        )
    ]
    out = apply_output_spec(preds, output, mode="from_parsed")
    assert out[0].format.parse_ok and out[0].format.schema_ok
    assert out[0].parsed["n2s"] is True
