"""Apply OutputSpec to PredictionRecords in from_raw / from_parsed modes."""

from __future__ import annotations

from typing import List, Literal

from linguaeval.core.schema import FormatStatus, OutputSpec, PredictionRecord
from linguaeval.parse.parsers import parse_raw
from linguaeval.parse.validate import validate_parsed

ParseMode = Literal["from_raw", "from_parsed"]


def apply_output_spec(
    preds: List[PredictionRecord],
    output: OutputSpec,
    *,
    mode: ParseMode = "from_parsed",
) -> List[PredictionRecord]:
    """Mutate/return predictions with parse_ok/schema_ok from OutputSpec.

    - from_raw: require raw_output → Parser → Validator
    - from_parsed: use existing parsed dict → Validator only (legacy replay safe)
    """
    out: List[PredictionRecord] = []
    for p in preds:
        details = {"mode": mode, "parser": output.parser}
        parsed = dict(p.parsed or {})
        parse_ok = True
        parse_err = ""

        if mode == "from_raw":
            raw = p.raw_output
            if raw is None or str(raw).strip() == "":
                parse_ok = False
                parse_err = "missing_raw_output"
                parsed = {}
            else:
                parsed_obj, parse_ok, parse_err = parse_raw(str(raw), output.parser)
                parsed = dict(parsed_obj or {})
        else:
            # from_parsed: keep existing parsed; if empty and raw exists, optionally leave as-is
            if not parsed and p.raw_output:
                # do not auto-switch modes; leave empty → schema fail
                parse_ok = True
                parse_err = "from_parsed_empty_parsed"
            else:
                parse_ok = True

        schema_ok = False
        schema_details = {}
        if parse_ok and parsed:
            schema_ok, schema_details = validate_parsed(parsed, output)
        elif not parse_ok:
            schema_ok = False
            schema_details = {"errors": [parse_err or "parse_failed"]}
        else:
            schema_ok, schema_details = validate_parsed(parsed if parsed else None, output)

        details.update(schema_details)
        if parse_err:
            details["parse_error"] = parse_err

        out.append(
            PredictionRecord(
                sample_id=p.sample_id,
                model_id=p.model_id,
                raw_output=p.raw_output,
                parsed=parsed,
                scores=dict(p.scores or {}),
                format=FormatStatus(
                    parse_ok=parse_ok,
                    schema_ok=schema_ok,
                    details=details,
                ),
                usage=dict(p.usage or {}),
                timing=dict(p.timing or {}),
                error=p.error if parse_ok else (parse_err or p.error),
                meta={**(p.meta or {}), "parse_mode": mode},
            )
        )
    return out
