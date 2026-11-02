# 1000-frame continuation/grace tests

Run `grace_261015_2124_15646126`; original Cam 2_2 frames 0-999.
One shared inference pass at centroid 0.05; two independent trackers. Frozen models; no training.
Landmarks 0.2; Shell identity gate 0.5; variable counts; no interpolation or count cap.
SLPs keep high-threshold poses (including unassigned) plus successful low-threshold continuations.
All raw centroid slots and pose measurements are retained in raw_candidates.npz; no chunk SLPs.
Gaps CSVs explicitly list missing frames. Pending identities are memory, not observations.

| Birth threshold | Mean observed poses | Mean assigned | Track IDs | Low-score continuations | Gap links |
|---|---:|---:|---:|---:|---:|
| 0.2 | 134.929 | 134.583 | 1239 | 6937 | 1839 |
| 0.1 | 137.037 | 136.554 | 1605 | 3311 | 2181 |

Controls use the same candidate pool, filtered at each birth threshold, with the unchanged adjacent-frame tracker.
Control counts are not a new independent detector run. Count fluctuations and longer tracks do not establish identity accuracy.
No visual ground truth has been collected; false links, false positives and missed males still require review.
Relink SLPs only to the matching full-resolution original MP4, not a resized or temporally cropped video.
