# Toy paired compare (P1-A acceptance A/B)

Known transition counts on `intent_class` (32 applicable):

| Transition | Count | sample_id |
|------------|------:|-----------|
| stable_correct | 20 | t01–t20 |
| gain | 6 | t21–t26 |
| regression | 2 | t27–t28 |
| both_wrong | 4 | t29–t32 |

Target name is `intent_class` (not `label`) — Kernel must not care.
