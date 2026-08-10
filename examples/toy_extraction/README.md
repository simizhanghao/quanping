# Example-03: Toy JSON Extraction

Proves Kernel handles multi-field structured output without N2S field names.

- Targets: `amount` (number), `time` (string)
- Mode: `from_raw` — OutputSpec Parser + Validator actually run
- 32 smoke samples; a few intentional field errors + 1 invalid JSON
