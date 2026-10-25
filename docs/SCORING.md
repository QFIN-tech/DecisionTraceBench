# Scoring protocol

## Full-trajectory match

A correct trajectory covers every minimum gold event in the same cycle and order, matching action, speaker, scope, complete value, applicable time evidence, and settlement result. Event ID spelling is immaterial, but internal predecessor links must be valid. Cycles are identified by array position and order; `cycle_id` is unnecessary and ignored if retained in legacy files. A missing required event cannot be recovered by exempting other nodes. Cycle boundaries are not merged.

The implementation aligns ordered events and performs maximum one-to-one matching between gold and predicted trajectories. One prediction cannot recover two gold trajectories, nor can two predictions generate two true positives for one gold trajectory. When maximum-match solutions tie, the deterministic matching prefers solutions that do not unnecessarily reinstate a previously exempt new trajectory.

## Local equivalence

JSON object-key order does not matter. Array order is preserved by default; no set conversion or general free-text similarity threshold is applied. Same-currency money expressed as structured number/unit pairs is converted exactly (for example, 50000 yuan = 5 ten-thousand yuan). No exchange-rate conversion, guessed colloquial magnitude, or blanket term/date normalization is used.

An approved semantic correspondence can cover field reference, scope, or value representation at a specific occurrence. It must preserve all stated information. It cannot relax actions, speaker labels, settlement results, numerical facts, cycles, predecessor structure, or time. Cross-field scope/value candidates cannot bypass a rejected or missing field correspondence. The reviewer returns categorical decisions, never an F1 or a subjective numeric grade; code computes the score.

## Extra predictions

Adjudicate unmatched nodes against the task instructions, actual input and relevant reference evidence using these labels:

- `exempt_extra`: supported and query-compliant refinement; neutral for the required-chain match.
- `evidence_uncertain`: actually reviewed but unresolved; neutral under this protocol, with the uncertainty retained.
- `invalid_extra`: unsupported or explicitly query-violating content; blocks that full trajectory.
- `replacement_error`: a wrong attempted required event, not a harmless addition; blocks recovery.

Only the scoring view contracts legitimate exempt nodes and their predecessor edges. Raw predictions are not edited. An unresolved review is different from an unperformed review. Unreviewed nodes are not granted an exemption.

A wholly new gold-absent trajectory can be excluded from the prediction denominator only after all its nodes and structure are reviewed as supported or eligible unresolved content. It creates no true positive. If it subsequently matches a gold trajectory through an approved field correspondence, it re-enters the denominator. Known wrong nodes prevent whole-track exemption.

## Time

For each required event, let P be the predicted interval and G the gold trigger interval:

`overlap(P, G) / duration(P) >= 0.80`.

This is not intersection-over-union. Intervals must have positive duration and nonnegative integer endpoints. The implementation uses an exact integer inequality to avoid boundary rounding. Use temporal matching for audio conditions and disable it only for a declared condition without temporal localization, such as reference-script evaluation. Business dates and durations inside `value` remain evaluated in both settings.

## Aggregation

For recording i, let G_i be the gold trajectory count, P_i the raw prediction count, L_i the count of wholly exempt predictions that remain unmatched, and TP_i the complete one-to-one matches:

`F1_i = 2 TP_i / (G_i + P_i - L_i)`.

The main **DCIST is recording-macro F1**: `100 * mean(F1_i)` on a declared cohort. Global micro-F1 pools the numerator and denominator before division and is a separate statistic. Evaluation requires completed answers by default. For a fixed cohort, `python -m decisiontracebench score` also supports an explicit no-answer-zero policy, with those cases separately labeled and counted; see [input and output reference](FORMAT.md). Completed malformed answers are scored as format errors. Missing adjudications leave the reviewed aggregate unset rather than changing the cohort.

The extra-node ratio is the number of distinct exempt node coordinates divided by the number of predicted nodes. Nodes selected as required matches are removed from the exemption union. New-track nodes are counted once, not once as a track and again as a node. This measures output granularity, not hallucination rate. It never compensates for a missing required trajectory. The scorer reports no extra-node ratio for a malformed answer.

The constructed example has G=1, P=2, L=1, TP=1 after review. Thus F1=1 and its extra-node ratio is 1/3. Its single required trajectory does not gain two true positives merely because another supported field was predicted.

## Adjudication records

Adjudication records supply the semantic and evidence judgments used in reviewed mode. The scorer validates their binding to specific input values and coordinates, required evidence fields and protected matching constraints. Assessing the evidence itself, field eligibility and the meaning of a factual claim is the reviewer's responsibility under the task protocol. Missing required decisions leave the reviewed score unset. Gold schema, cycle-structure or applicable time-validation errors stop evaluation.
