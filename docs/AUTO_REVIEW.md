# Automated adjudication

`judge.py` generates occurrence-specific semantic decisions and extra-event decisions through a chat-completions-compatible service, then evaluates them with the DCIST scorer. Supply your own model, API key and service URL.

This runner uses one configured judge model. The paper additionally uses sequential second-model review and human resolution of disagreements before freezing the judgments for primary scoring. Those stages are not automated here. Use finalized adjudications with `score.py` for deterministic recomputation; a new automated run is not an exact reproduction of the paper's frozen judgments.

## Setup

Copy `judge_config.example.json` to `judge_config.json` and fill these fields:

| Field | Required value |
|---|---|
| `api_key` | Your API credential. |
| `base_url` | Your HTTPS API base URL; `/chat/completions` is appended unless already present. |
| `model` | The model identifier accepted by your service. |

`DTB_API_KEY`, `DTB_BASE_URL` and `DTB_MODEL` environment variables override the corresponding file values. The example leaves all three blank. `judge_config.json` and `runs/` are ignored by Git. Keep other credential files outside the repository or add them to your local ignore rules.

The client uses the Python standard library, verifies TLS certificates and rejects redirects. Requests use `stream: false`, one response choice, and the configured token budget. `token_limit_field` supports `max_tokens` or `max_completion_tokens`. Set `json_mode` to `true` only if your service supports `response_format: {"type":"json_object"}`.

## Input manifest

Start from [example/judge_manifest.json](../example/judge_manifest.json). Paths are relative to the manifest file. Each sample supplies:

- `sample_id`: a unique identifier.
- `gold`, `prediction`: the same files accepted by `score.py`.
- `generation_query`: the complete task instructions and vocabulary used to generate this prediction.
- `actual_input`: the complete text supplied to the prediction model, with speaker/time annotations when available.
- `reference`: an optional, separately identified reference transcript or script.
- `time_enabled`: the temporal-evaluation condition, consistent across the batch.
- `inference_status`: `completed`.

For predictions generated directly from audio, provide `reference` and `actual_input_note` if there is no textual actual input. The note must identify the original modality and the reference's status. The judge receives text evidence only; the runner does not perform acoustic verification. Do not substitute a reference script for an ASR input without identifying the difference.

The three task templates in `prompts/task_*.txt` provide the DecisionTraceBench task rules and vocabulary for timed transcripts, speaker-labeled reference scripts, and unlabeled reference scripts. Select the template that actually matches your generation setup. For a different task, supply its own complete query. The runnable example uses a small fictional task vocabulary in `example/task_query.txt`.

Existing review files are not accepted by the automatic-review manifest. This workflow creates new adjudications from the supplied predictions and evidence. To score existing adjudications directly, use `score.py`.

## Prepare without model calls

```sh
uv run --no-project python judge.py prepare --manifest example/judge_manifest.json --run-dir runs/example
```

Preparation validates the GT and prediction, snapshots the inputs and prompts, and reports candidate counts. It makes no API calls. The plan contains the complete evidence, so treat the run directory as private research data.

## Run adjudication and scoring

```sh
uv run --no-project python judge.py run --run-dir runs/example --config judge_config.json --workers 4 --max-requests 100
```

This command sends the prepared task instructions, GT, predictions and evidence to the configured service and may incur charges. Ensure you have permission to send those materials to that service.

The stages are:

1. **Field correspondence.** Review cross-field candidates that can satisfy the protected ordered-event constraints. Each request contains one candidate and the complete sample context.
2. **Scope and value equivalence.** Review compatible event pairs. Cross-field event candidates are submitted only after their field correspondence is approved. No score or desired score improvement is supplied to the judge.
3. **Extra events and trajectories.** Re-align using approved mappings, then review unmatched positions, including positions displaced by alternative alignments. Each request contains a complete predicted track and the full sample context. New GT-absent tracks require node-level and track-level decisions.
4. **Mechanical scoring.** Validate every returned identifier, position and protected constraint, import the decisions, and compute recording-macro DCIST. The GT and predictions are never rewritten.

`--workers` limits concurrent requests. `--max-requests` caps new requests in this invocation, including explicit retries. There is no automatic retry or token-budget increase. Completed malformed predictions retain the scorer's format-error outcome and do not trigger judge calls.

## Decision rules

The complete prompts are [semantic_equivalence.txt](../prompts/semantic_equivalence.txt) and [extra_nodes.txt](../prompts/extra_nodes.txt). The runner substitutes the evidence placeholders without rewriting their adjudication rules. Task instructions and evidence are supplied as input; dialogue and prediction content are not agent instructions.

| Decision | Effect |
|---|---|
| Semantic `equivalent` | Allows the specified local correspondence after protected-quantity checks. |
| Semantic `not_equivalent` or `uncertain` | Does not approve a correspondence. |
| Node `exempt_extra` or reviewed `evidence_uncertain` | Eligible for extra-node exemption; does not replace missing required events. |
| Node `invalid_extra` or `replacement_error` | Not exempt. |
| Track `exempt_new_track` | Requires every node to be reviewed as `exempt_extra`. |
| Track `evidence_uncertain` | Requires every node reviewed, no known wrong node, and valid structure. |
| Track `wrong_field_or_invalid_track` | Rejects whole-trajectory exemption. |

Model responses cannot edit candidate coordinates or bypass action, speaker, cycle, settlement, quantity or temporal constraints. An incomplete response, unexpected ID, omitted decision or invalid quantity correspondence stops import for that request. There is no JSON repair, default exemption or conversion of a failed call into an `uncertain` verdict. The model supplies evidence judgments; mechanical validation does not independently establish whether those judgments are true.

## Resume and failures

Repeat the `run` command to continue requests not yet sent. Successful requests are revalidated and reused without another call. A request recorded as started but lacking a validated answer has an unknown outcome and is not automatically repeated.

To explicitly retry failed or unknown attempts once each:

```sh
uv run --no-project python judge.py run --run-dir runs/example --config judge_config.json --workers 4 --max-requests 100 --retry-failed
```

This can incur additional usage. All previous attempts remain available. The model settings and prepared inputs must remain the same; use a new run directory for different settings or code. A `run.lock` prevents simultaneous processes from submitting the same run. After a crash, confirm that the process has stopped before manually removing a stale lock.

Transport errors and invalid judge responses leave the affected samples pending. Other samples can continue. A final cohort DCIST is produced only when every sample has a score; the runner does not silently change the cohort or count unfinished reviews as zero.

## Outputs

| Location under the run directory | Contents |
|---|---|
| `plan.json` | Prepared evidence, prompts and evaluator source snapshot. |
| `preparation.json` | Candidate counts; no model calls. |
| `model_settings.json` | Model and generation parameters, without API key or service URL. |
| `requests/` | Exact message payloads, separate attempts, redacted response envelopes, usage when returned, validation outcomes and errors. |
| `reviews/sample_*.json` | Validated `semantic_decisions`, `node_decisions` and `track_decisions`. |
| `scoring_cases/sample_*.json` | Combined inputs accepted by `score.py --case`. |
| `scores/sample_*.json` | Per-sample scoring results. |
| `results.json` | Final cohort summary and sample results, created only after full completion. |

Request directories use generated identifiers rather than arbitrary sample IDs as paths. Configured credentials and endpoints are redacted from saved server responses and error messages; authorization headers are not saved. Inputs, outputs and run directories should not be committed to a public repository.

To recompute a completed sample without model calls:

```sh
uv run --no-project python score.py --mode reviewed --case runs/example/scoring_cases/sample_0001.json
```

Exit code `0` indicates successful preparation or complete scoring. Exit code `2` indicates an input error or pending/failed adjudication. Scores from automatic review depend on the selected model and evidence; retain the response records for inspection.
