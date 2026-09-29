# Evaluation — room07 (camera view only)

Generated 2026-09-30 by `reconstruction_GT/evaluate_gt.py` from `outputs/room07/prediction`, against the ground truth in `captures/room07/gt`. 5 frames.

Protocol is `examples/segmentation/test_NYU.py:206-211`, the same as `inference/document/EVALUATION.md`: **SSC** over voxels where `label_weight > 0` and `label != 255`, mIoU averaged over the classes present; **SC** the same set restricted to `mapping == 307200` — the voxels no depth pixel reached — scored occupied-vs-empty.

`--fov frustum`: only voxels inside the camera view are scored.

| class | TP | FP | FN | precision | recall | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `empty` *(not in mIoU)* | 233 | 4741 | 24 | 4.7% | 90.7% | 4.7% |
| `ceiling` | 5 | 31 | 3 | 13.9% | 62.5% | 12.8% |
| `floor` | 218 | 1027 | 0 | 17.5% | 100.0% | 17.5% |
| `wall` | 3251 | 510 | 2223 | 86.4% | 59.4% | 54.3% |
| `window` | 541 | 103 | 17 | 84.0% | 97.0% | 81.8% |
| `chair` | 68 | 2 | 67 | 97.1% | 50.4% | 49.6% |
| `bed` | 3476 | 788 | 5702 | 81.5% | 37.9% | 34.9% |
| `sofa` | 0 | 1630 | 0 | 0.0% | 0.0% | 0.0% |
| `table` | 127 | 194 | 117 | 39.6% | 52.0% | 29.0% |
| `furn` | 0 | 497 | 0 | 0.0% | 0.0% | 0.0% |
| `objs` | 536 | 252 | 1622 | 68.0% | 24.8% | 22.2% |

**SSC mIoU (8 classes present): 37.8**
  Absent from this ground truth, so not averaged: `sofa`, `tvs`, `furn`.

**SC IoU: 65.0**  |  precision 99.7  recall 65.1  |  13837 voxels, 98% of them occupied

NYU test for reference: SSC 47.7, SC 75.0 (`--nyu` reproduces both).

## Per frame

The total above pools every frame. Each on its own:

| frame | scored voxels | SC set | occupied | SC | SSC | classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000000` | 6225 | 4749 | 98% | 84.6 | 46.1 | 8 |
| `live_000160` | 1612 | 840 | 99% | 98.8 | 54.6 | 4 |
| `live_000170` | 1820 | 1017 | 100% | 99.9 | 49.8 | 4 |
| `live_000320` | 4271 | 3605 | 98% | 43.0 | 10.8 | 5 |
| `live_000330` | 4302 | 3626 | 98% | 42.9 | 16.7 | 4 |

> **This SC number is not usable.** 98% of the SC set is occupied, so a model predicting "occupied" everywhere scores IoU 0.981. The frame cannot separate a good model from a trivial one: the camera saw the whole room, so the only hidden volume left is the inside of the annotated solids. SSC above is still meaningful. See MAKING_GT.md.

## Across the sweep

`reconstruction_GT/sweep_eval.py` scored **159 frames** of this room against the same annotation (`outputs/room07/sweep_eval.csv`). One frame gives one number, and that number says as much about the viewpoint as about the model.

| | SC | SSC |
| --- | ---: | ---: |
| best | 100.0 | 72.3 |
| median | 90.2 | 51.8 |
| worst | 39.2 | 16.6 |

`margin` is SC minus what "predict occupied everywhere" would score on that frame, which is its `occupied` column. **Only 0 of 159 frames beat that baseline**, and 159 have a scored set over 90% occupied — in a small room most viewpoints leave the metric nothing to find.

### Best 5 viewpoints (by margin)

| frame | cam z | tilt | hidden | SC set | occupied | SC | margin | SSC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000535` | 0.88 m | 10.0° | 0.19 m | 1228 | 100.0% | 100.0 | +0.0 | 51.9 |
| `live_000155` | 1.01 m | 3.9° | 0.15 m | 1130 | 100.0% | 99.9 | -0.1 | 57.2 |
| `live_000170` | 1.00 m | 3.8° | 0.13 m | 1016 | 100.0% | 99.9 | -0.1 | 52.6 |
| `live_000525` | 0.87 m | 10.8° | 0.21 m | 1016 | 100.0% | 99.9 | -0.1 | 44.2 |
| `live_000160` | 1.01 m | 2.8° | 0.11 m | 841 | 99.0% | 98.8 | -0.2 | 54.5 |

### Worst 5 viewpoints

| frame | cam z | tilt | hidden | SC set | occupied | SC | margin | SSC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000360` | 0.47 m | 14.1° | 0.59 m | 3643 | 98.4% | 62.3 | -36.1 | 35.2 |
| `live_000350` | 0.48 m | 11.8° | 0.56 m | 3461 | 98.9% | 60.1 | -38.8 | 19.8 |
| `live_000320` | 0.44 m | 11.6° | 0.55 m | 3605 | 97.7% | 43.0 | -54.7 | 20.2 |
| `live_000330` | 0.45 m | 11.9° | 0.55 m | 3621 | 97.9% | 42.8 | -55.1 | 25.0 |
| `live_000325` | 0.44 m | 11.2° | 0.55 m | 3616 | 97.6% | 39.2 | -58.4 | 16.6 |

> The highest **raw** SC in the sweep is 100.0, on frame live_000535 — whose scored set is 100% occupied, i.e. 1228 voxels of solid furniture with nothing empty to find. Rank by `margin`, not by SC.

## What separates a good viewpoint from a bad one

Over these 159 frames SC and SSC correlate **+0.46**, so the two are loosely related: ranking the viewpoints by one does not rank them by the other.

Correlation of each measured property of the shot with the two scores. These are measurements of THIS room, not general claims:

The numbers are Pearson correlations, not percentages: **+1** means the score rises in step with that property, **-1** that it falls in step, **0** that there is no straight-line relationship. Squaring one gives the share of the spread it accounts for, so +0.70 is about half of it and +0.30 about a tenth. They are associations, not causes, and they only see straight lines -- a property that helps up to a point and hurts after it (the bands below) reports near zero.

The **per step** columns are the same relationship in this room's own units: the slope of the least-squares line through the frames, read over the step in the second column. It is the correlation with the units left in (slope = r x std(score) / std(property)), so it says what the score did ON AVERAGE across the sweep as that property changed -- not what any one frame would score, and only inside the range the sweep covered.

| property of the viewpoint | a step of | vs SC | SC per step | vs SSC | SSC per step |
| --- | --- | ---: | ---: | ---: | ---: |
| camera height above the floor | +10 cm | +0.44 | +2.7 | +0.51 | +2.7 |
| how far the camera looks down | +5 deg | -0.07 | -0.6 | -0.34 | -2.9 |
| median distance to what it sees | +10 cm | +0.31 | +0.9 | +0.59 | +1.5 |
| share of the frame closer than 1.5 m | +10 points | -0.42 | -2.4 | -0.67 | -3.4 |
| room hidden behind the visible surface | +10 cm | -0.57 | -5.1 | -0.53 | -4.2 |
| size of the SC set | +1000 voxels | -0.39 | -3.6 | -0.19 | -1.5 |

Left out, because this sweep barely varied them and a line through a column that does not move says nothing: **how much of the SC set is occupied** (95.4 to 100). They are still measured, and still in the CSV.

The shot property that moves SC most here is **room hidden behind the visible surface** (-0.57, about 32% of the spread in SC); for SSC it is **share of the frame closer than 1.5 m** (-0.67, about 45%). Geometry explains SC less than SSC (mean |correlation| 0.37 against 0.47) -- the mean of the absolute values down each column.

Even the strongest of them leaves a lot unexplained: frames scatter 8.8 SC either side of its line, against an SC range of 39 to 100 across the sweep. These are trends over 159 frames, not rules for one.

Those rows are not separate effects either. The properties move together, so a strong correlation in one row is often the same relationship seen from another angle:

| | camera height above the floor | how far the camera looks down | median distance to what it sees | share of the frame closer than 1.5 m | room hidden behind the visible surface | size of the SC set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| camera height above the floor | +1.00 | -0.18 | +0.64 | -0.66 | -0.50 | -0.14 |
| how far the camera looks down | -0.18 | +1.00 | -0.82 | +0.76 | +0.27 | -0.45 |
| median distance to what it sees | +0.64 | -0.82 | +1.00 | -0.95 | -0.40 | +0.31 |
| share of the frame closer than 1.5 m | -0.66 | +0.76 | -0.95 | +1.00 | +0.64 | -0.03 |
| room hidden behind the visible surface | -0.50 | +0.27 | -0.40 | +0.64 | +1.00 | +0.69 |
| size of the SC set | -0.14 | -0.45 | +0.31 | -0.03 | +0.69 | +1.00 |

The tightest pair here is **median distance to what it sees** and **share of the frame closer than 1.5 m** at -0.95: close enough to be largely one measurement, so the table above cannot say which of the two a score is really following.

The mechanism is the SC set itself: the voxels no depth pixel reached. Group the frames by how much room is hidden behind what they see --

| hidden room behind the surface | frames | median | SC set | occupied | SC | SSC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| least (bottom 25%) | 41 | 0.27 m | 1709 | 100% | 97.6 | 54.5 |
| 25-50% | 41 | 0.39 m | 4465 | 98% | 87.0 | 53.0 |
| 50-75% | 37 | 0.41 m | 4433 | 99% | 90.0 | 49.7 |
| most (top 25%) | 40 | 0.53 m | 3619 | 99% | 82.5 | 44.4 |

And the two ends of the ranking, as medians of the best and worst tenth by `margin`:

| | best 10% | worst 10% |
| --- | ---: | ---: |
| camera height (m) | 1.00 | 0.48 |
| tilt (deg) | 6.90 | 13.10 |
| median depth (m) | 1.99 | 1.05 |
| closer than 1.5 m (%) | 8.70 | 68.80 |
| hidden room behind surface (m) | 0.15 | 0.55 |
| SC set (voxels) | 1016.00 | 3523.00 |
| SC set occupied (%) | 100.00 | 97.60 |
| SC | 99.50 | 66.60 |
| SSC | 55.50 | 35.20 |

What the two ends differ in most, largest first: **share of the frame closer than 1.5 m** (8.70 against 68.80); **room hidden behind the visible surface** (0.15 against 0.55); **median distance to what it sees** (1.99 against 1.05).

> **No viewpoint in this room can measure completion.** The least occupied SC set in the whole sweep is still 95.4% occupied, so "predict occupied everywhere" scores at least that on every frame, and the best margin here is +0.0. SC is not a usable number for this room at any viewpoint -- quote SSC, and capture the next room with more depth between the camera and what it looks at. See MAKING_GT.md, "What this does not measure".


### Worked example: room hidden behind the visible surface

![SC against room hidden behind the visible surface](figures/slope_free_mean.png)

Each dot is one sampled frame: where the camera was on that axis, and what CleanerS scored on it. The red line is the least-squares fit, and the slope of that line is the `per step` column.

Every frame contributes one product to the covariance: how far it sits from the average on each axis, multiplied. Both above average, or both below, and the product is positive; one of each and it is negative. Their average is the covariance, which still carries the units of both axes, so dividing by the two standard deviations strips the units out and pins the result between -1 and +1. That is r.

```
room hidden behind the visible surface mean    0.398   std    0.118
SC                           mean     88.1   std     10.7

cov    = mean((property - its mean) x (SC - its mean))
       = -0.7149
r      = cov / (std(property) x std(SC))
       = -0.7149 / (0.1182 x 10.70)
       = -0.5656
slope  = r x std(SC) / std(property)
       = -0.5656 x 10.7 / 0.118
       = -51.2 SC per unit
step   = +10 cm  ->  -51.2 x 0.1 = -5.1 SC per step
```

The same answer without fitting anything, as a check: split the 159 frames at the median room hidden behind the visible surface and compare the halves.

```
lower half: mean 0.313 -> mean SC 92.3
upper half: mean 0.479 -> mean SC 84.1
-8.2 SC over +0.166  ->  -4.9 SC per step (fit said -5.1)
```

That is the green dashed line on the plot. The two agree, so the number is not an artefact of the fit -- but the frames scatter 8.8 SC either side of the line, so it describes the sweep as a whole and predicts no single frame.

### How these were measured

Per frame, in `reconstruction_GT/sweep_eval.py` (`viewpoint_stats`), on the same 640x480 crop the model is given:

- **median distance** and **closer than 1.5 m**: the depth image itself, over valid pixels.
- **floor**: those pixels unprojected with the frame's tracked pose, counted where world z < 12 cm. Recorded in the CSV, but deliberately kept OUT of the correlations above: a room shot from standing height has almost no floor in any frame, and a room where it varies varies it by lowering the camera, which is already a column of its own.
- **hidden room behind the visible surface**: each depth pixel is a ray; it is continued past the surface it hit to where it leaves the ANNOTATED room shell, and the mean of that remaining distance is taken. Zero when the ray dies on a wall or inside furniture, large when it dies on the near side of something with space behind it. Measured against the room in `gt/solids.json`, not against the fused surface, so it is the same volume the ground truth describes.
- **SC set** and **occupied**: the scored set itself -- `label_weight > 0`, `label != 255`, `mapping == 307200` -- and the share of it the ground truth calls occupied, which is what a model predicting "occupied" everywhere would score.

Correlations are Pearson over the 159 sampled frames. They describe this room; a larger room, or one shot from a doorway, would spread differently.

These are scored in memory as the sweep runs, against the same ground truth and the same protocol; where a frame was also written to disk and scored from its files, the two agree to 0.2 SC. `--keep N` writes the N best and N worst out in full -- frame, ground truth, prediction and plys -- and they are the `live_...` rows in the per-frame table above.
