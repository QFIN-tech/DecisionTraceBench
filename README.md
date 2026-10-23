# DecisionTraceBench

Evaluation code for **DecisionTraceBench: Tracking Decision-Critical Information in Multi-Party Spoken Conversations**.

[English](README.md) | [简体中文](README.zh-CN.md)

Decision-Critical Information State Tracking (DCIST) evaluates the recovery of complete decision-state trajectories from multi-party conversations. A trajectory match requires the correct event sequence, actions, speaker attribution, scope, values and temporal evidence.

## Features

- Ordered event alignment and one-to-one trajectory matching.
- Cycle, predecessor and settlement constraints.
- Exact monetary equivalence for structured amounts.
- Configurable temporal matching.
- Support for semantic-equivalence mappings and extra-event adjudications.
- Automatic adjudication through a configurable chat-completions service.
- Single-sample evaluation and recording-macro aggregation.

## Requirements

Python **3.10 or newer**. The scorer uses the Python standard library.

Run the tests from the repository root:

```sh
python run_tests.py
```

Run Python without `-O` so that input-validation assertions remain enabled.

## Quick start

### Evaluate a prediction

Use `raw` mode for exact matching with monetary normalization:

```sh
python score.py --mode raw --gold example/gold.json --prediction example/prediction.json --time on
```

Use `reviewed` mode to apply semantic mappings and extra-event decisions:

```sh
python score.py --mode reviewed --gold example/gold.json --prediction example/prediction.json --reviews example/reviews.json --time on
```

The example illustrates how approved semantic equivalence and an exempt additional trajectory affect scoring: raw F1 is `0`, reviewed F1 is `1`, and the exempt-extra-node ratio is `1/3`.

### Evaluate a batch

```sh
python score.py --mode reviewed --manifest example/manifest.json --output results/example.json
```

Input paths in a manifest are resolved relative to the manifest directory. Each sample must have a unique ID, and all samples in a batch must use the same temporal-evaluation setting.

### Use a combined input file

A case file can contain the gold, prediction, settings and adjudications together:

```sh
python score.py --mode reviewed --case example/case.json
```

## Automatic semantic and extra-event review

To generate adjudications rather than supply an existing review file, prepare a review manifest with the prediction, GT, task query and source evidence:

```sh
python judge.py prepare --manifest example/judge_manifest.json --run-dir runs/example
```

This command makes no API calls. Copy `judge_config.example.json` to `judge_config.json` and fill your `api_key`, `base_url` and `model`. Then run:

```sh
python judge.py run --run-dir runs/example --config judge_config.json --workers 4 --max-requests 100
```

The run command sends the prepared evidence to your configured service and may incur charges. It reviews field correspondences, scope/value equivalence, and extra events in order, then computes DCIST. Successful requests are reused on resume; failed or incomplete requests require explicit retry. Missing decisions are not automatically exempted.

The complete prompts, input requirements, configuration and resume options are described in [automated adjudication](docs/AUTO_REVIEW.md).

## Input format

Gold annotations and predictions use the same JSON structure:

```text
tracks
  entity_id, field_id
  cycles
    cycle_id
    events
      event_id, source_event_id
      action, speaker_id, scope_id, value
      start_ms, end_ms
      result (for settle events)
```

Actions are `start`, `advance`, `hold` and `settle`. Event indices in adjudication records are zero-based. `--prediction` accepts a JSON answer or a complete raw answer saved as text.

A batch manifest has the following form:

```json
{
  "samples": [
    {
      "sample_id": "sample_001",
      "gold": "gold.json",
      "prediction": "prediction.json",
      "reviews": "reviews.json",
      "time_enabled": true,
      "inference_status": "completed"
    }
  ]
}
```

The `reviews` file is optional in `reviewed` mode and omitted in `raw` mode. See [input and review formats](docs/FORMAT.md) for field definitions, decision labels and examples.

## Scoring

For each recording, let `G` be the gold trajectory count, `P` the predicted trajectory count, `L` the count of wholly exempt unmatched trajectories, and `TP` the complete one-to-one matches:

```text
F1 = 2 × TP / (G + P − L)
DCIST (%) = 100 × mean(recording-level F1)
```

DCIST uses a recording-macro average. Each required trajectory must be recovered completely; matching only its final value is insufficient.

| Setting | Behavior |
|---|---|
| `--mode raw` | Exact typed-JSON matching with monetary equivalence; no semantic mappings or extra-event exemptions. |
| `--mode reviewed` | Applies supplied occurrence-specific semantic mappings and node/trajectory decisions. |
| `--time on` | Requires temporal overlap divided by predicted interval length to be at least 80%. |
| `--time off` | Evaluates a condition without temporal localization; other matching requirements remain active. |

Monetary normalization treats structured values such as `{"number":"50000","unit":"元"}` and `{"number":"5","unit":"万元"}` as equivalent. Semantic mappings can cover field references, scope and value expressions. They are supplied as adjudication records and bound to specific events; the scorer validates and applies them rather than generating judgments.

For the complete protocol, see [scoring rules](docs/SCORING.md).

## Output

The output JSON contains `summary` and `samples`:

| Field | Meaning |
|---|---|
| `summary.dcist_percent` | Recording-macro score on a 0–100 scale. |
| `summary.recording_macro_f1` | The same score on a 0–1 scale. |
| `summary.tp` | Total number of completely matched trajectories. |
| `samples[].score` | Per-sample counts, precision, recall and F1. |
| `samples[].details` | Matching and review diagnostics. |

In `reviewed` mode, missing required adjudications produce `review_required` and a `null` final score. Completed malformed answers receive zero credit and are counted separately. Explicit no-answer handling is described in [the format guide](docs/FORMAT.md).

Exit code `0` indicates completed scoring; `2` indicates an input error or pending review. The scorer preserves input files and refuses to overwrite an existing output.

## Repository structure

```text
score.py          Single-sample and batch CLI
judge.py          Automatic adjudication and scoring CLI
code/             Matching, validation, monetary and temporal scoring
docs/             Scoring protocol and file formats
prompts/          Task and adjudication queries
example/          Runnable examples
tests/            Regression tests
run_tests.py      Test runner
```

## Citation

If you use this code, please cite **DecisionTraceBench: Tracking Decision-Critical Information in Multi-Party Spoken Conversations**.

## License

This project is licensed under the [MIT License](LICENSE).
