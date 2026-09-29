# Evaluation — room08 (camera view only)

Generated 2026-09-25 by `reconstruction_GT/evaluate_gt.py` from `outputs/room08/prediction`, against the ground truth in `captures/room08/gt`. 5 frames.

Protocol is `examples/segmentation/test_NYU.py:206-211`, the same as `inference/document/EVALUATION.md`: **SSC** over voxels where `label_weight > 0` and `label != 255`, mIoU averaged over the classes present; **SC** the same set restricted to `mapping == 307200` — the voxels no depth pixel reached — scored occupied-vs-empty.

`--fov frustum`: only voxels inside the camera view are scored.

| class | TP | FP | FN | precision | recall | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `empty` *(not in mIoU)* | 3226 | 3510 | 118 | 47.9% | 96.5% | 47.1% |
| `ceiling` | 323 | 126 | 33 | 71.9% | 90.7% | 67.0% |
| `floor` | 275 | 429 | 0 | 39.1% | 100.0% | 39.1% |
| `wall` | 2239 | 596 | 1480 | 79.0% | 60.2% | 51.9% |
| `window` | 749 | 639 | 1030 | 54.0% | 42.1% | 31.0% |
| `bed` | 2115 | 410 | 2265 | 83.8% | 48.3% | 44.2% |
| `sofa` | 0 | 382 | 0 | 0.0% | 0.0% | 0.0% |
| `table` | 0 | 43 | 0 | 0.0% | 0.0% | 0.0% |
| `tvs` | 0 | 0 | 230 | 0.0% | 0.0% | 0.0% |
| `furn` | 92 | 78 | 635 | 54.1% | 12.7% | 11.4% |
| `objs` | 321 | 290 | 712 | 52.5% | 31.1% | 24.3% |

**SSC mIoU (8 classes present): 33.6**
  Absent from this ground truth, so not averaged: `chair`, `sofa`, `table`.

**SC IoU: 59.0**  |  precision 97.8  recall 59.8  |  12067 voxels, 72% of them occupied

NYU test for reference: SSC 47.7, SC 75.0 (`--nyu` reproduces both).

## Per frame

The total above pools every frame. Each on its own:

| frame | scored voxels | SC set | occupied | SC | SSC | classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000000` | 6746 | 5707 | 82% | 58.6 | 33.3 | 6 |
| `live_000180` | 3124 | 1951 | 56% | 82.3 | 37.4 | 6 |
| `live_000235` | 1426 | 1118 | 96% | 46.9 | 25.6 | 6 |
| `live_000240` | 1393 | 1076 | 99% | 43.4 | 30.4 | 6 |
| `live_000580` | 3154 | 2215 | 38% | 65.1 | 26.1 | 6 |

## Across the sweep

`reconstruction_GT/sweep_eval.py` scored **210 frames** of this room against the same annotation (`outputs/room08/sweep_eval.csv`). One frame gives one number, and that number says as much about the viewpoint as about the model.

| | SC | SSC |
| --- | ---: | ---: |
| best | 100.0 | 59.9 |
| median | 69.2 | 33.1 |
| worst | 37.0 | 19.7 |

`margin` is SC minus what "predict occupied everywhere" would score on that frame, which is its `occupied` column. **Only 28 of 210 frames beat that baseline**, and 98 have a scored set over 90% occupied — in a small room most viewpoints leave the metric nothing to find.

### Best 5 viewpoints (by margin)

| frame | cam z | tilt | hidden | SC set | occupied | SC | margin | SSC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000580` | 1.40 m | 16.0° | 0.39 m | 2216 | 37.6% | 65.2 | +27.6 | 26.0 |
| `live_000180` | 1.17 m | 5.8° | 0.27 m | 1940 | 55.8% | 82.2 | +26.3 | 37.8 |
| `live_000555` | 1.20 m | 9.1° | 0.44 m | 2090 | 34.8% | 58.8 | +24.0 | 21.8 |
| `live_000565` | 1.31 m | 7.2° | 0.42 m | 2346 | 39.2% | 62.5 | +23.4 | 27.5 |
| `live_000575` | 1.39 m | 14.2° | 0.41 m | 2239 | 40.2% | 63.2 | +23.1 | 19.7 |

### Worst 5 viewpoints

| frame | cam z | tilt | hidden | SC set | occupied | SC | margin | SSC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000835` | 1.18 m | 22.4° | 0.15 m | 998 | 99.5% | 53.5 | -46.0 | 41.2 |
| `live_000255` | 1.02 m | 32.6° | 0.28 m | 1431 | 99.0% | 52.7 | -46.2 | 35.3 |
| `live_000830` | 1.17 m | 20.6° | 0.14 m | 971 | 98.6% | 52.1 | -46.4 | 27.8 |
| `live_000235` | 1.06 m | 29.5° | 0.22 m | 1116 | 96.0% | 46.8 | -49.1 | 26.4 |
| `live_000240` | 1.05 m | 27.8° | 0.21 m | 1079 | 98.5% | 43.6 | -54.9 | 31.6 |

> The highest **raw** SC in the sweep is 100.0, on frame live_000645 — whose scored set is 100% occupied, i.e. 470 voxels of solid furniture with nothing empty to find. Rank by `margin`, not by SC.

## What separates a good viewpoint from a bad one

Over these 210 frames SC and SSC correlate **+0.08**, so the two are barely related: ranking the viewpoints by one does not rank them by the other.

Correlation of each measured property of the shot with the two scores. These are measurements of THIS room, not general claims:

The numbers are Pearson correlations, not percentages: **+1** means the score rises in step with that property, **-1** that it falls in step, **0** that there is no straight-line relationship. Squaring one gives the share of the spread it accounts for, so +0.70 is about half of it and +0.30 about a tenth. They are associations, not causes, and they only see straight lines -- a property that helps up to a point and hurts after it (the bands below) reports near zero.

The **per step** columns are the same relationship in this room's own units: the slope of the least-squares line through the frames, read over the step in the second column. It is the correlation with the units left in (slope = r x std(score) / std(property)), so it says what the score did ON AVERAGE across the sweep as that property changed -- not what any one frame would score, and only inside the range the sweep covered.

| property of the viewpoint | a step of | vs SC | SC per step | vs SSC | SSC per step |
| --- | --- | ---: | ---: | ---: | ---: |
| camera height above the floor | +10 cm | +0.70 | +5.1 | -0.03 | -0.1 |
| how far the camera looks down | +5 deg | +0.06 | +0.5 | +0.29 | +1.2 |
| median distance to what it sees | +10 cm | +0.46 | +1.9 | -0.30 | -0.6 |
| share of the frame closer than 1.5 m | +10 points | -0.59 | -4.2 | +0.31 | +1.0 |
| room hidden behind the visible surface | +10 cm | -0.71 | -4.5 | +0.04 | +0.1 |
| size of the SC set | +1000 voxels | -0.63 | -6.0 | -0.10 | -0.4 |
| how much of the SC set is occupied | +10 points | +0.43 | +5.4 | +0.06 | +0.3 |

The shot property that moves SC most here is **room hidden behind the visible surface** (-0.71, about 51% of the spread in SC); for SSC it is **share of the frame closer than 1.5 m** (+0.31, about 9%). Geometry explains SC more than SSC (mean |correlation| 0.51 against 0.16) -- the mean of the absolute values down each column.

Even the strongest of them leaves a lot unexplained: frames scatter 13.2 SC either side of its line, against an SC range of 37 to 100 across the sweep. These are trends over 210 frames, not rules for one.

Those rows are not separate effects either. The properties move together, so a strong correlation in one row is often the same relationship seen from another angle:

| | camera height above the floor | how far the camera looks down | median distance to what it sees | share of the frame closer than 1.5 m | room hidden behind the visible surface | size of the SC set | how much of the SC set is occupied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| camera height above the floor | +1.00 | +0.38 | +0.34 | -0.45 | -0.84 | -0.89 | +0.03 |
| how far the camera looks down | +0.38 | +1.00 | -0.62 | +0.51 | -0.13 | -0.43 | -0.06 |
| median distance to what it sees | +0.34 | -0.62 | +1.00 | -0.90 | -0.46 | -0.18 | +0.01 |
| share of the frame closer than 1.5 m | -0.45 | +0.51 | -0.90 | +1.00 | +0.55 | +0.30 | -0.17 |
| room hidden behind the visible surface | -0.84 | -0.13 | -0.46 | +0.55 | +1.00 | +0.93 | -0.29 |
| size of the SC set | -0.89 | -0.43 | -0.18 | +0.30 | +0.93 | +1.00 | -0.15 |
| how much of the SC set is occupied | +0.03 | -0.06 | +0.01 | -0.17 | -0.29 | -0.15 | +1.00 |

The tightest pair here is **room hidden behind the visible surface** and **size of the SC set** at +0.93: close enough to be largely one measurement, so the table above cannot say which of the two a score is really following.

The mechanism is the SC set itself: the voxels no depth pixel reached. Group the frames by how much room is hidden behind what they see --

| hidden room behind the surface | frames | median | SC set | occupied | SC | SSC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| least (bottom 25%) | 53 | 0.04 m | 552 | 100% | 93.7 | 29.6 |
| 25-50% | 52 | 0.26 m | 1208 | 95% | 66.3 | 37.5 |
| 50-75% | 52 | 0.54 m | 3386 | 88% | 71.5 | 34.9 |
| most (top 25%) | 53 | 0.80 m | 5761 | 85% | 47.7 | 32.9 |

And the two ends of the ranking, as medians of the best and worst tenth by `margin`:

| | best 10% | worst 10% |
| --- | ---: | ---: |
| camera height (m) | 1.26 | 1.02 |
| tilt (deg) | 13.60 | 16.40 |
| median depth (m) | 2.13 | 1.59 |
| closer than 1.5 m (%) | 25.10 | 45.80 |
| hidden room behind surface (m) | 0.27 | 0.40 |
| SC set (voxels) | 1839.00 | 1749.00 |
| SC set occupied (%) | 50.70 | 91.10 |
| SC | 65.20 | 46.20 |
| SSC | 26.40 | 30.00 |

What the two ends differ in most, largest first: **how much of the SC set is occupied** (50.70 against 91.10); **median distance to what it sees** (2.13 against 1.59); **camera height above the floor** (1.26 against 1.02).

> 28 of 210 viewpoints beat the trivial baseline. The ones that do are the frames where the hidden volume is neither inside a solid (nothing to find) nor enormous (too much to guess); on this room that is a median depth of 2.13 m against 1.59 m for the worst tenth.


### Worked example: room hidden behind the visible surface

![SC against room hidden behind the visible surface](figures/slope_free_mean.png)

Each dot is one sampled frame: where the camera was on that axis, and what CleanerS scored on it. The red line is the least-squares fit, and the slope of that line is the `per step` column.

Every frame contributes one product to the covariance: how far it sits from the average on each axis, multiplied. Both above average, or both below, and the product is positive; one of each and it is negative. Their average is the covariance, which still carries the units of both axes, so dividing by the two standard deviations strips the units out and pins the result between -1 and +1. That is r.

```
room hidden behind the visible surface mean    0.419   std    0.297
SC                           mean     69.3   std     18.8

cov    = mean((property - its mean) x (SC - its mean))
       = -3.9755
r      = cov / (std(property) x std(SC))
       = -3.9755 / (0.2973 x 18.75)
       = -0.7129
slope  = r x std(SC) / std(property)
       = -0.7129 x 18.8 / 0.297
       = -45.0 SC per unit
step   = +10 cm  ->  -45.0 x 0.1 = -4.5 SC per step
```

The same answer without fitting anything, as a check: split the 210 frames at the median room hidden behind the visible surface and compare the halves.

```
lower half: mean 0.157 -> mean SC 79.4
upper half: mean 0.682 -> mean SC 59.1
-20.3 SC over +0.525  ->  -3.9 SC per step (fit said -4.5)
```

That is the green dashed line on the plot. The two agree, so the number is not an artefact of the fit -- but the frames scatter 13.2 SC either side of the line, so it describes the sweep as a whole and predicts no single frame.

### How these were measured

Per frame, in `reconstruction_GT/sweep_eval.py` (`viewpoint_stats`), on the same 640x480 crop the model is given:

- **median distance** and **closer than 1.5 m**: the depth image itself, over valid pixels.
- **floor**: those pixels unprojected with the frame's tracked pose, counted where world z < 12 cm. Recorded in the CSV, but deliberately kept OUT of the correlations above: a room shot from standing height has almost no floor in any frame, and a room where it varies varies it by lowering the camera, which is already a column of its own.
- **hidden room behind the visible surface**: each depth pixel is a ray; it is continued past the surface it hit to where it leaves the ANNOTATED room shell, and the mean of that remaining distance is taken. Zero when the ray dies on a wall or inside furniture, large when it dies on the near side of something with space behind it. Measured against the room in `gt/solids.json`, not against the fused surface, so it is the same volume the ground truth describes.
- **SC set** and **occupied**: the scored set itself -- `label_weight > 0`, `label != 255`, `mapping == 307200` -- and the share of it the ground truth calls occupied, which is what a model predicting "occupied" everywhere would score.

Correlations are Pearson over the 210 sampled frames. They describe this room; a larger room, or one shot from a doorway, would spread differently.

These are scored in memory as the sweep runs, against the same ground truth and the same protocol; where a frame was also written to disk and scored from its files, the two agree to 0.2 SC. `--keep N` writes the N best and N worst out in full -- frame, ground truth, prediction and plys -- and they are the `live_...` rows in the per-frame table above.
