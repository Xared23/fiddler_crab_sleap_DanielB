# 200-frame centroid threshold comparison

Run: `threshold_261015_2000_15641165`; video Crab_Cam_2_2, frames 0-199.
Same frozen model pair; no training. Landmark threshold remains 0.2. No detection/track count cap.
One final SLP per condition, with all detections and optional body-first IDs; no frame chunks.

| Centroid threshold | Mean raw/frame | Mean assigned/frame | Tracklets |
|---|---:|---:|---:|
| 0.2 | 128.23 | 128.09 | 818 |
| 0.1 | 135.06 | 134.77 | 895 |
| 0.05 | 140.10 | 139.51 | 905 |

Threshold 0.2 is extracted from existing predictions, not rerun. Raw counts are not validated animal counts.
Tracking retains the existing Shell score >=0.5 and duplicate/ambiguity rules; added poses may remain unassigned.
Tracking cannot create a missed detection or maintain identity through an unobserved burrow/occlusion interval.
Spatial matching is one-to-one within 5 px; unmatched detections are candidates, not proven recovered males.
Review PNGs deliberately focus on high-score added candidates and do not estimate precision or recall.
All SLPs reference the original full video and original frame numbers. Do not relink them to a resized/cropped clip.
The primary review plots are threshold_comparison.png and review_frame_*.png.
