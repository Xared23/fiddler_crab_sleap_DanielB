# Adjacent-frame detection-count fluctuations

Source: Cam 2_2 frames 0-199, 199 consecutive-frame pairs per threshold.
Frame interval: 1001/30000 s (~33.37 ms). No temporal binning or smoothing.
Primary N is the count of all pose predictions, consistent with the threshold comparison.

Definitions: delta = N(t+1)-N(t); magnitude = abs(delta).
Variances below use divisor 199 (descriptive population variance, ddof=0).
Sample variances (ddof=1), finite-in-image counts and ID-assigned counts are also saved.

| Threshold | Mean abs change | Var(abs change) | Mean signed change | Var(signed change) |
|---|---:|---:|---:|---:|
| 0.2 | 1.738693 | 2.524684 | 0.050251 | 5.545214 |
| 0.1 | 1.964824 | 2.737456 | 0.065327 | 6.593722 |
| 0.05 | 2.281407 | 3.197192 | 0.100503 | 8.391909 |

These are NET count changes: one detection disappearing and another appearing can cancel.
They do not measure pose-coordinate jitter, individual appearance/disappearance rates or ID switches.
The 199 increments are not assumed independent. No inference about true recall/precision is made.
