# Method and results

This project was completed by Yu Zhengye, Yu Yue, Sun Shengkai, and Jiang Shanglin for STA326 Data Science Practice at Southern University of Science and Technology in June 2026. It classifies rock images as non-meteorite (`0`) or meteorite (`1`). The results below are **team results**; the final course report does not assign individual contributions to specific members.

## Task and data

The course provided 5,098 labeled training images and a 511-image first-stage test set. The training labels comprised 2,413 non-meteorites and 2,685 meteorites. The course brief reported a substantially different first-stage test balance, approximately 4.06 non-meteorites per meteorite. The final second-stage submission contained **194 images**, so the 511-image and 194-image figures describe different evaluation stages.

The images also varied in lighting, background, scale, and shooting angle. In several experiments, very high out-of-fold (OOF) scores did not transfer to the second-stage leaderboard. We therefore treated local validation and online results as different evidence, rather than interchangeable estimates of performance.

## Final pipeline

1. **Preserve shape.** Convert images to RGB, resize the longest side to 384 pixels with `LongestMaxSize`, pad to 384 × 384 with `PadIfNeeded`, then normalize and convert to tensors. The final run used minimal augmentation; earlier, stronger augmentation settings were retained as ablations.
2. **Fine-tune a model.** Use the pretrained `timm` SwinV2 Base backbone `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` with a one-logit binary classification head. The team fine-tuned the model on the course task; the pretrained backbone was initialization, not the final classifier by itself.
3. **Stabilize training.** Use five stratified folds (`seed=2026`), eight epochs, training batch size 4, validation batch size 8, gradient accumulation over eight steps, AdamW with weight decay 0.05, head learning rate `5e-4`, backbone learning rate `5e-5`, one warmup epoch followed by cosine decay to `2e-7`, automatic mixed precision, gradient clipping at 1.0, exponential moving average with decay 0.999, and binary cross-entropy with label smoothing 0.05. The final loss did not use `pos_weight`.
4. **Use lower-weight pseudo labels.** An external, API-generated prediction CSV for second-stage test images was added to each training fold with sample weight 0.40, compared with 1.00 for labeled training images. The pseudo-labeled test images were excluded from validation folds. This step is part of the reported final method and depends on an external artifact.
5. **Validate and submit.** Each fold saved its best validation-F1 checkpoint. The team's OOF probability search selected **0.44** as the validation threshold. For test inference, probabilities from the five checkpoints were averaged. The **final submission used probability ranking and a fixed count of 86 positive labels**, leaving 108 negative labels. It did **not** label the 194 test images by applying the OOF threshold of 0.44.

## Reported results

| Measure | Reported value | Scope |
| --- | ---: | --- |
| OOF F1 | 0.93808 | Five-fold hold-out predictions on labeled training images |
| OOF precision | 0.9337 | Same OOF predictions |
| OOF recall | 0.9425 | Same OOF predictions |
| OOF threshold | 0.44 | F1-optimal threshold on OOF probabilities |
| Kaggle F1 | 0.83720 | Team-reported second-stage leaderboard score for the final submission |
| Final submission | 194 images | 86 predicted meteorites and 108 predicted non-meteorites |

The course report also records these comparisons. They are experimental observations under different configurations, not estimates that isolate a single causal effect in every row.

| Experiment | OOF F1 | Second-stage leaderboard F1 |
| --- | ---: | ---: |
| Swin with strong augmentation | 0.9924 | 0.5638 |
| ConvNeXtV2 with aspect-safe preprocessing | 0.9913 | 0.7363 |
| Focal-loss variant | 0.9929 | approximately 0.60 |
| GeM-pooling variant | 0.9909 | 0.6707 |
| Final SwinV2 with minimal preprocessing and pseudo labels | 0.9381 | 0.83720 |

The main empirical lesson is that the configuration with the highest local OOF F1 was not the best reported online result. The final choice combined morphology-preserving preprocessing, stable fine-tuning, fold-based validation, and a calibrated decision rule.

## Interpretation and reproduction limits

- **The two F1 scores measure different data.** OOF F1 is calculated from held-out training folds; Kaggle F1 was returned for a separate second-stage test set. The OOF-to-leaderboard gap is consistent with distribution shift, but does not by itself identify one cause.
- **The OOF figure includes model selection.** Checkpoints were selected using fold validation F1, and the final probability threshold was selected on pooled OOF predictions. The reported OOF F1 is therefore a tuned validation figure, not an untouched estimate from a separate holdout set.
- **The final result is transductive.** Training used pseudo labels for the second-stage test images, and final labels were count-calibrated to 86 positives. Neither step should be omitted when describing the reported 0.83720 result. The pseudo labels may contain mistakes, and the chosen count may not transfer to a new population.
- **The published score is reported evidence.** The value 0.83720 appears in the team's final course report and presentation; this repository does not independently verify the Kaggle leaderboard receipt. Model checkpoints, the exact external pseudo-label CSV, and the official competition images are required to reproduce the original submission. The repository's code and configurations explain the method, but cannot recreate that scored submission from code alone if those artifacts are unavailable.
- **Data rights are not documented here.** The course directed students to download images from the [Kaggle competition](https://www.kaggle.com/competitions/meteorite-identification/data). This repository should not redistribute the course images or test-set prediction CSV without confirmed permission. Prepare the data locally according to the repository README.
- **Further validation is needed.** A new split by image source or shooting conditions, confidence checks on pseudo labels, and false-positive/false-negative review would give a better account of deployment robustness.

Source of the method and numbers: the team's final STA326 course report dated June 7, 2026, the course project brief and scoring rubric, and the saved June 2, 2026 SwinV2 run metadata. Earlier presentation notes contained an older score and are not used for the results above.
