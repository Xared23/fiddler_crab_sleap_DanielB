# Female held-out validation

Frames: [481, 1252]; 184 human instances; 142 predictions.

| Radius (px) | Precision | Recall | F1 | Median error (px) |
|---|---|---|---|---|
| 5 | 0.7746478873239436 | 0.5978260869565217 | 0.6748466257668712 | 2.338573215898971 |
| 10 | 0.823943661971831 | 0.6358695652173914 | 0.7177914110429447 | 2.4914400553784075 |
| 20 | 0.823943661971831 | 0.6358695652173914 | 0.7177914110429447 | 2.4914400553784075 |
| 50 | 0.8309859154929577 | 0.6413043478260869 | 0.7239263803680982 | 2.491690837442552 |

- Only two held-out frames; same clip as training, possible temporal correlation.
- Validation frames also select best checkpoints/early stopping; not an independent final test set.
- Precision is relative to provided labels; unlabeled animals can be counted as false positives.
- FBody localization/detection only; no limb pose or identity accuracy measured.
