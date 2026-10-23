# Input and output reference

## Model answer

The answer is a JSON object containing `tracks`. Each track has `entity_id`, `field_id`, and ordered `cycles`. A cycle contains `cycle_id` and ordered `events`.

An event contains:

| Field | Meaning / evaluation |
|---|---|
| `event_id` | Unique within an answer; literal ID spelling need not equal gold. |
| `action` | `start`, `advance`, `hold`, or `settle`; exact match. |
| `speaker_id` | `S1`, `S2`, ...; exact match, not role-based relabeling. |
| `value` | Supported content; typed JSON, amount equivalence, and approved occurrence-specific semantic mappings. |
| `source_event_id` | Null for the first event of a cycle; otherwise the preceding event ID in that cycle. |
| `scope_id` | Business effect and authority scope; exact or explicitly approved local equivalence. |
| `start_ms`, `end_ms` | Integer times from the beginning of the recording; required when time evaluation is enabled. |
| `result` | Only on `settle`: `accepted`, `rejected`, `withdrawn`, or `revoked`. |

Define the field and scope vocabulary in the task protocol. The scorer compares the supplied gold and prediction using those labels and any approved correspondences. Schema validation checks the structure of an answer; semantic and extra-node adjudications assess its content against the task rules and evidence.

Cycle labels and event labels are identifiers, not semantic answers. Cycle number/order, event order, and predecessor relations remain constrained. `settle` must end a cycle. An initial `hold` or `settle` is permitted when supported; an invented `start` is not required. `reason` and `derived_final_state` are auxiliary fields and do not enter the main trajectory match.

## Combined case file

`example/case.json` is a complete executable template. Required top-level keys:

- `inference_status`: `completed` for an available answer. Explicit no-answer handling is described under [separate-file and batch evaluation](#separate-file-and-batch-evaluation).
- `time_enabled`: explicit Boolean. Use `true` for audio/transcript experiments and `false` only for the reference-script condition without temporal localization.
- `gold`: the model-answer shape above, with the required minimum trajectories.
- `prediction`: the parsed model-answer object or the complete raw answer string.

Optional arrays contain externally supplied, evidence-backed decisions:

- `semantic_decisions`: `candidate_id`, `dimension`, gold/prediction track indices, exact compared `gold` and `prediction` representations, `verdict`, `reason`, `evidence_quote`. For `value` or `scope`, also give `cycle_index`, `gold_event_index`, and `prediction_event_index`. For `track_key`, the compared representations are `[entity_id, field_id]`.
- `node_decisions`: `prediction_track_index`, `cycle_index`, `event_index`, `label`, `reason`, `rule_quote`, and `evidence_quote`.
- `track_decisions`: `prediction_track_index`, `label`, and `reason`, only for a field absent from raw gold. Whole-track exemption additionally requires explicit review of every node.

Semantic verdicts are `equivalent`, `not_equivalent`, or `uncertain`. Node labels are `exempt_extra`, `evidence_uncertain`, `invalid_extra`, or `replacement_error`. Track labels are `exempt_new_track`, `evidence_uncertain`, or `wrong_field_or_invalid_track`. See [scoring rules](SCORING.md) for their effects.

All indices are **zero based**. Semantic approvals bind to the actual values and event coordinates, not a global string-replacement dictionary. Cross-field event mappings require an approved field correspondence. Numerical representation changes may require an explicit quantity certificate; executable examples appear in `tests/test_trial_score.py` (`quantity_equivalence_verified`, `quantity_equivalence_reason`, and, where needed, `quantity_alignment`). The scorer rejects changes that violate its protected-quantity constraints.

When importing adjudications, join decisions to candidate coordinates by `candidate_id` or `review_id`, rather than array order or text similarity. Internal track review IDs use `case/T0`, `case/T1`, etc. Input records specify `prediction_track_index` and, for node decisions, the cycle and event indices. Evidence and decisions must be supplied in the review records.

`context_for_review_only` and `context_note` are optional documentation for reviewers and are not interpreted by the scorer. The example contains fictional content and constructed time coordinates.

## Output

`score.py` returns `summary` and `samples`. Each sample contains its selected `score`, `f1`, `dcist_percent`, status and matching diagnostics in `details`.

Within `details`, `dcist` is the reviewed single-recording F1 as a fraction, not a percentage. `raw_score` has neither semantic nor extra-node approvals but retains exact monetary equivalence. `reviewed_score` uses supplied valid approvals and is null when necessary adjudications are missing. Score objects contain `tp`, `gold`, `pred`, `precision`, `recall`, and `f1`. Completed malformed answers have zero credit in `raw_score` and `dcist`, with no `reviewed_score` field. Matches identify the input coordinates; input files remain intact.

In reviewed mode, `review_required` and a null score mean necessary unmatched positions have not been adjudicated; they represent neither a scored zero nor an evidence-uncertain exemption. Completed malformed output is a format error with zero matching credit. Failed or truncated inference is an input error unless the caller explicitly applies the no-answer policy below. `example/expected.json` lists the principal diagnostics for the runnable example.

## Separate-file and batch evaluation

`score.py --gold gold.json --prediction answer.json --time on --mode raw` reads the two documents separately. Prediction input may be a JSON document or a complete raw answer string stored as text. It must contain the answer, not an API response envelope. Extracting an envelope's content is the caller's responsibility; never rewrite its meaning.

With `--mode reviewed`, `--reviews reviews.json` accepts only the three optional arrays listed above. It cannot replace gold, prediction, time mode or inference status. See `example/reviews.json`.

`--manifest manifest.json` instead accepts:

```json
{
  "samples": [
    {
      "sample_id": "unique_recording_id",
      "gold": "gold.json",
      "prediction": "answer.json",
      "reviews": "reviews.json",
      "time_enabled": true,
      "inference_status": "completed"
    }
  ]
}
```

Paths are relative to the manifest directory, or absolute paths explicitly supplied by the caller. IDs must be unique; the batch must use one declared time condition and one scoring mode. Output contains per-sample results and a macro summary. Existing output files are not overwritten. Return code 2 reports errors or missing review; no partial-cohort final score is produced.

For a fixed-cohort **no-answer-counted-zero** protocol, explicitly set `inference_status` to `no_answer`, omit `prediction` (or set it to null in a combined case), and pass `--no-answer-policy zero`. `score.py` validates the gold, emits a separate `no_answer_counted_zero` outcome and includes zero in the recording-macro mean. The default policy is `error`. Pending runs, absent files and existing malformed answers are not automatically reclassified as no-answer. The lower-level `code/evaluate.py` interface accepts only completed answers; use `score.py` for no-answer handling and batch aggregation.

`summary.dcist_percent` uses a 0–100 scale; `summary.recording_macro_f1` and per-sample `f1` use 0–1. In raw mode these refer to the raw baseline; only reviewed mode applies supplied exemptions and semantic correspondences. The nested `details` object may report pending reviews that do not affect raw-mode scoring.
