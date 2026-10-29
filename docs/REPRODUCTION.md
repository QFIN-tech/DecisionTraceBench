# Paper reproduction materials

## Data-side indexes

The dataset is not yet publicly accessible. The following sidecars will be available on [Hugging Face](https://huggingface.co/datasets/QFIN/DecisionTraceBench) when the dataset is released:

| Material | File |
|---|---|
| Historical-to-public human sample IDs | `metadata/human_sample_mapping.jsonl` |
| Historical-to-public synthetic sample IDs | `metadata/synthetic_sample_mapping.jsonl` |
| Reconstructed main human comparison cohort | `metadata/human_paper_main25.json` |
| Synthetic construction reference scripts | `synthetic/reference_scripts.jsonl` |

The main cohort contains 25 recordings and 293 gold trajectories, excluding `R0024` / `fsb_0015_b`. The selection was reconstructed from archived mappings and scoring code; it is not a frozen prediction/result artifact. The full 26-recording human subset will be available as `human/test`. Historical human ASR-input hashes refer to FLAC copies, not the public MP3 bytes.

Synthetic references contain `sample_id`, `family_id` and ordered `turns`; each turn has `turn_id`, `speaker_id`, `text`, `start_ms` and `end_ms`. They are construction scripts aligned to assembled speech, not independently transcribed recordings. They are reference material, not inputs to the raw-audio or ASR-transcript evaluation conditions. Human reference reading scripts are not supplied by these sidecars.

## Frozen direct-audio query

[paper_audio_direct_20261021.txt](../assets/prompts/paper_audio_direct_20261021.txt) is the byte-identical direct-audio query referenced by the archived Gemini inference scripts.

```text
SHA-256: 6e0c208129c235c1c4de483b9894062e62798534261b3ab8f4e696bf0add8181
```

This historical snapshot retains the original `cycle_id` wording. Current prediction templates omit that unnecessary label; the scorer accepts either form. Keep this file unchanged when checking historical query hashes.

## Remaining experiment artifacts

The planned dataset sidecars will provide data lookup and cohort selection; the frozen query is included in this GitHub repository. Reproducing the paper's result tables still requires the exact frozen predictions and finalized semantic/extra-node adjudications. The actual Gemini request/conversion receipts and human reference scripts are not included. The synthetic shared-2,224-recording cohort cannot be reconstructed solely from ASR availability; it depends on the downstream completed outputs.

Use [the scorer](SCORING.md) with finalized judgments for deterministic recomputation. Running a new model or judge creates new outputs and is not a reconstruction of the frozen paper results.
