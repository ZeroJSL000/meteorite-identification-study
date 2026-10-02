# STA326 Meteorite Identification

This is a curated code release of a four-person **Data Science Practice (STA326)** course project at the Southern University of Science and Technology. The task was to classify photographs as meteorites or non-meteorites under a substantial shift between the training images and the later evaluation images.

The final pipeline trains five stratified folds of a 384-pixel SwinV2 Base classifier, averages their test probabilities, and labels the 86 highest-ranked images as meteorites for the 194-image second-stage submission. The count rule was a competition-specific calibration choice. It is not a general decision threshold or a substitute for independent test labels.

## Reported results

| Experiment | Out-of-fold F1 | Second-stage Kaggle F1 |
| --- | ---: | ---: |
| Earlier strong-augmentation baseline | 0.9924 | 0.5638 |
| Final SwinV2 pipeline with count calibration | 0.93808 | 0.83720 |

The final validation score measures out-of-fold predictions on labeled training images. The second-stage score is the team's reported Kaggle result for its 194-image submission, including the Top-86 rule. These are different evaluation populations; the table is evidence of a validation-to-evaluation gap, not a controlled estimate of the effect of one change. The leaderboard value comes from the submitted final course report; an independent leaderboard receipt is not bundled here. See [Method and results](docs/method-and-results.md) for the methods, provenance, other ablations, and limitations.

## Team and scope

The final report lists **Yu Yue, Yu Zhengye, Sun Shengkai, and Jiang Shanglin** as the project team. The available report does not assign individual parts of the pipeline to specific members. This repository preserves the team's architecture and identifies it as team work.

The source tree contains training, data handling, model construction, checkpoint loading, cross-validation metrics, and ensemble inference. The historical configurations are retained so the experiment structure is visible. The included CSWin implementation is third-party Microsoft code and retains its original MIT notice; see [Third-party notices](THIRD_PARTY_NOTICES.md).

## Repository layout

```text
configs/                 Experiment configurations, including the final SwinV2 run
docs/method-and-results.md
scripts/run_experiment.sh
scripts/run_prediction.sh
src/train.py             Stratified five-fold training and real-image OOF evaluation
src/predict.py           Checkpoint ensemble and optional count calibration
src/models/             timm model factory and historical CSWin model
src/utils/              Configuration, transforms, training loop, and metrics
```

## Requirements and data layout

The original locked environment uses **Python 3.14 or newer** and [`uv`](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --frozen
```

Place course-authorized data and artifacts locally. They are intentionally excluded from Git:

```text
data/
  train_labels.csv                 columns: id,label
  sample_submission.csv           columns: id,label; 194 rows for the final stage
  train_images/train_images/      labeled training images
  test_images/test_images/        images matching sample_submission.csv
  pseudo_labels/api_prediction.csv  optional external predictions; columns: id,label
weights/
  swinv2_base_window12to24_192to384.ms_in22k_ft_in1k.pth
  checkpoints/swinv2_base_384_minimal_pseudo/
    fold_0_best.pth ... fold_4_best.pth
    metadata.json
```

The image CSV IDs must match the filenames. The final configuration requires exactly 194 submission rows so its Top-86 rule is not accidentally applied to a different stage. The final training configuration uses externally generated pseudo-labels with sample weight `0.40`; those predictions are never part of the real-image validation folds. The external CSV, course images, labels, pretrained weights, and trained checkpoints are not included. Consequently, the reported final result cannot be reproduced from this source alone.

## Run inference

With the five final fold checkpoints and their `metadata.json` in place:

```bash
bash scripts/run_prediction.sh
```

This runs `src/predict.py` with `configs/swinv2_base_384_minimal_pseudo.yaml` and writes under `outputs/swinv2_base_384_minimal_pseudo/`:

- `submission_probabilities.csv`: mean of the five fold probabilities.
- `submission_model_only.csv`: labels from the stored OOF threshold.
- `submission_count_calibrated.csv` and `submission.csv`: top-ranked labels using the configured count of 86.

The source's archived final run used a slightly different experiment directory name. If your checkpoints are in a different directory, pass `--checkpoint-dir` to `src/predict.py`. The metadata's `checkpoint_files` and model name must agree with the supplied files. The Top-86 count is specific to the 194-image final submission; change or remove both `inference.target_positive_count` and `inference.expected_rows` for another dataset.

## Run training

To retrain the reported pseudo-label configuration, supply the external pseudo-label CSV and pretrained state dict, then run:

```bash
bash scripts/run_experiment.sh configs/swinv2_base_384_minimal_pseudo.yaml
```

For a run without the external predictions, start from `configs/swinv2_base_384_minimal_external.yaml` and review its settings before comparing results. Training uses aspect-preserving resize and padding, stratified folds, BCE with label smoothing, separate head/backbone learning rates, gradient accumulation, warmup and cosine scheduling, EMA checkpoints, and threshold selection on real-image out-of-fold predictions. `--no-pretrained` is available for a smoke run but does not reproduce the reported experiment.

## Release boundaries

No course images, labels, test-derived pseudo-labels, submission files, or model weights are tracked here. Do not treat the recorded leaderboard score as an independently verified deployment metric. The team's original private repository had no project-wide software license; this curated copy does not add one on the team's behalf.
