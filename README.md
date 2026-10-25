# DecisionTraceBench

![DecisionTraceBench — Right answer. Wrong trace.](assets/decisiontracebench-banner.png)

DecisionTraceBench is a speech benchmark for tracking decision-critical information in multi-party conversations. Grounded in Chinese financial dialogues, it evaluates how information is introduced, revised, suspended and settled, together with speaker attribution, business scope and temporal evidence. It aims to support collaboration between academia and industry on traceable speech understanding through structured trajectory annotations, task prompts and evaluation scripts.

🤗 [Hugging Face](https://huggingface.co/datasets/QFIN/DecisionTraceBench)   |   🌐 [简体中文](README.zh-CN.md)

Decision-Critical Information State Tracking (DCIST) evaluates the recovery of complete decision-state trajectories from multi-party conversations. A trajectory match requires the correct event sequence, actions, speaker attribution, scope, values and temporal evidence.

Both dataset subsets are available on [Hugging Face](https://huggingface.co/datasets/QFIN/DecisionTraceBench):

| Subset | Recordings | Hours | Trajectories | Events |
|---|---:|---:|---:|---:|
| Human-performed | 26 | 8.61 | 308 | 562 |
| Synthetic | 2,246 | 100.39 | 5,319 | 18,227 |

Both are fixed research evaluation cohorts without training/development splits. Transcripts and construction scripts are not distributed. The paper's main human-audio comparison uses a common cohort of 25 recordings, distinct from the complete 26-recording human release.

For paper-comparable prediction, both subsets use the same global vocabulary: **193 entity–field pairs and 64 scopes**. Use the complete task templates below or the [shared vocabulary](https://huggingface.co/datasets/QFIN/DecisionTraceBench/blob/main/assets/label_vocabulary.json). Subset coverage counts are not alternative inference vocabularies.

## Features

- Ordered event alignment and one-to-one trajectory matching.
- Cycle, predecessor and settlement constraints.
- Exact monetary equivalence for structured amounts.
- Configurable temporal matching.
- Support for semantic-equivalence mappings and extra-event adjudications.
- Automatic adjudication through a configurable chat-completions service.
- Single-sample evaluation and recording-macro aggregation.

## Requirements

Python **3.10 or newer**. The scorer uses the Python standard library. The commands below use [uv](https://docs.astral.sh/uv/).

Run the tests from the repository root:

```sh
uv run --no-project python -B -m unittest discover -s tests -v
```

Run Python without `-O` so that input-validation assertions remain enabled.

## Quick start

### Evaluate a prediction

Use `raw` mode for exact matching with monetary normalization:

```sh
uv run --no-project python -m decisiontracebench score --mode raw --gold assets/examples/gold.json --prediction assets/examples/prediction.json --time on
```

Use `reviewed` mode to apply semantic mappings and extra-event decisions:

```sh
uv run --no-project python -m decisiontracebench score --mode reviewed --gold assets/examples/gold.json --prediction assets/examples/prediction.json --reviews assets/examples/reviews.json --time on
```

The example illustrates how approved semantic equivalence and an exempt additional trajectory affect scoring: raw F1 is `0`, reviewed F1 is `1`, and the exempt-extra-node ratio is `1/3`.

### Evaluate a batch

```sh
uv run --no-project python -m decisiontracebench score --mode reviewed --manifest assets/examples/manifest.json --output results/example.json
```

Input paths in a manifest are resolved relative to the manifest directory. Each sample must have a unique ID, and all samples in a batch must use the same temporal-evaluation setting.

### Use a combined input file

A case file can contain the gold, prediction, settings and adjudications together:

```sh
uv run --no-project python -m decisiontracebench score --mode reviewed --case assets/examples/case.json
```

## Automatic semantic and extra-event review

To generate adjudications rather than supply an existing review file, prepare a review manifest with the prediction, GT, task query and source evidence:

```sh
uv run --no-project python -m decisiontracebench judge prepare --manifest assets/examples/judge_manifest.json --run-dir runs/example
```

This command makes no API calls. Copy `assets/configs/judge_config.example.json` to `assets/configs/judge_config.json` and fill your `api_key`, `base_url` and `model`. Then run:

```sh
uv run --no-project python -m decisiontracebench judge run --run-dir runs/example --config assets/configs/judge_config.json --workers 4 --max-requests 100
```

The run command sends the prepared evidence to your configured service and may incur charges. It reviews field correspondences, scope/value equivalence, and extra events in order, then computes DCIST. Successful requests are reused on resume; failed or incomplete requests require explicit retry. Missing decisions are not automatically exempted.

The complete prompts, input requirements, configuration and resume options are described in [automated adjudication](docs/AUTO_REVIEW.md).

The automatic runner uses one configured judge model. It does not automate the paper's sequential second-model review or human resolution of disagreements. Reproducing the reported scores requires the corresponding fixed cohort, predictions and finalized adjudications; a new judge run produces new judgments.

## Input format

Gold annotations and predictions use the same JSON structure:

```text
tracks
  entity_id, field_id
  cycles
    events
      event_id, source_event_id
      action, speaker_id, scope_id, value
      start_ms, end_ms
      result (for settle events)
```

Actions are `start`, `advance`, `hold` and `settle`. Event indices in adjudication records are zero-based. `--prediction` accepts a JSON answer or a complete raw answer saved as text.

Cycles are identified by array position and order; predictions do not need `cycle_id`. The scorer still accepts this ignored label in original gold files and older predictions. Cycle count, boundaries and order remain constrained.

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
decisiontracebench/  Python package and module CLI
assets/
  configs/          Configuration templates
  examples/         Runnable fictional examples
  prompts/          Task and adjudication prompts
docs/               Scoring protocol and file formats
tests/              Offline regression tests
```

## Python library

Use `python -m decisiontracebench score` or `python -m decisiontracebench judge` as the CLI entry points. Python callers can use `decisiontracebench.scoring.score_case` and `aggregate` directly. The package separates `evaluation` (review validation), `matching` (ordered alignment and trajectory matching), `schema`/`structure`, `money`/`quantities`/`timing`, and `judge_client`/`judge_pipeline`. `historical_reviews` reads archived candidate catalogs and is separate from the current CLI workflow.

## Citation

If you use this code, please cite **DecisionTraceBench: Tracking Decision-Critical Information in Multi-Party Spoken Conversations**.

## License

The code, prompts and documentation in this repository are licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE).

The separately distributed human-recording and synthetic subsets also use CC BY-NC 4.0. Third-party voice resources remain subject to their source licenses.
