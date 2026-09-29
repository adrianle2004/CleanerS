# Evaluation — room07 (camera view only)

Generated 2026-09-25 by `reconstruction_GT/evaluate_gt.py` from `outputs/room07/prediction`, against the ground truth in `captures/room07/gt`. 5 frames.

Protocol is `examples/segmentation/test_NYU.py:206-211`, the same as `inference/document/EVALUATION.md`: **SSC** over voxels where `label_weight > 0` and `label != 255`, mIoU averaged over the classes present; **SC** the same set restricted to `mapping == 307200` — the voxels no depth pixel reached — scored occupied-vs-empty.

`--fov frustum`: only voxels inside the camera view are scored.

| class | TP | FP | FN | precision | recall | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `empty` *(not in mIoU)* | 243 | 5610 | 18 | 4.2% | 93.1% | 4.1% |
| `ceiling` | 5 | 31 | 3 | 13.9% | 62.5% | 12.8% |
| `floor` | 237 | 1035 | 0 | 18.6% | 100.0% | 18.6% |
| `wall` | 3169 | 508 | 2270 | 86.2% | 58.3% | 53.3% |
| `window` | 541 | 114 | 16 | 82.6% | 97.1% | 80.6% |
| `chair` | 68 | 4 | 67 | 94.4% | 50.4% | 48.9% |
| `bed` | 3107 | 828 | 6301 | 79.0% | 33.0% | 30.4% |
| `sofa` | 0 | 1377 | 0 | 0.0% | 0.0% | 0.0% |
| `table` | 127 | 164 | 117 | 43.6% | 52.0% | 31.1% |
| `furn` | 0 | 472 | 0 | 0.0% | 0.0% | 0.0% |
| `objs` | 488 | 339 | 1690 | 59.0% | 22.4% | 19.4% |

**SSC mIoU (8 classes present): 36.9**
  Absent from this ground truth, so not averaged: `sofa`, `tvs`, `furn`.

**SC IoU: 59.3**  |  precision 99.8  recall 59.3  |  14055 voxels, 98% of them occupied

NYU test for reference: SSC 47.7, SC 75.0 (`--nyu` reproduces both).

## Per frame

The total above pools every frame. Each on its own:

| frame | scored voxels | SC set | occupied | SC | SSC | classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000000` | 6225 | 4749 | 98% | 84.6 | 46.1 | 8 |
| `live_000160` | 1644 | 883 | 99% | 99.7 | 54.3 | 4 |
| `live_000170` | 1926 | 1069 | 100% | 100.0 | 51.8 | 4 |
| `live_000320` | 4338 | 3676 | 98% | 30.5 | 14.4 | 4 |
| `live_000330` | 4334 | 3678 | 98% | 33.0 | 11.8 | 4 |

> **This SC number is not usable.** 98% of the SC set is occupied, so a model predicting "occupied" everywhere scores IoU 0.981. The frame cannot separate a good model from a trivial one: the camera saw the whole room, so the only hidden volume left is the inside of the annotated solids. SSC above is still meaningful. See MAKING_GT.md.

## Across the sweep

`reconstruction_GT/sweep_eval.py` scored **159 frames** of this room against the same annotation (`outputs/room07/sweep_eval.csv`). One frame gives one number, and that number says as much about the viewpoint as about the model.

| | SC | SSC |
| --- | ---: | ---: |
| best | 100.0 | 70.9 |
| median | 89.3 | 49.9 |
| worst | 30.5 | 16.0 |

`margin` is SC minus what "predict occupied everywhere" would score on that frame, which is its `occupied` column. **Only 1 of 159 frames beat that baseline**, and 159 have a scored set over 90% occupied — in a small room most viewpoints leave the metric nothing to find.

### Best 5 viewpoints (by margin)

| frame | cam z | tilt | hidden | SC set | occupied | SC | margin | SSC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000160` | 1.01 m | 3.1° | 0.12 m | 883 | 99.4% | 99.7 | +0.2 | 54.2 |
| `live_000170` | 1.01 m | 4.1° | 0.14 m | 1069 | 100.0% | 100.0 | +0.0 | 53.1 |
| `live_000535` | 0.88 m | 10.1° | 0.21 m | 1278 | 100.0% | 100.0 | +0.0 | 52.3 |
| `live_000150` | 1.01 m | 4.2° | 0.21 m | 1720 | 100.0% | 99.8 | -0.2 | 66.9 |
| `live_000155` | 1.01 m | 4.1° | 0.16 m | 1218 | 100.0% | 99.8 | -0.2 | 53.5 |

### Worst 5 viewpoints

| frame | cam z | tilt | hidden | SC set | occupied | SC | margin | SSC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000350` | 0.48 m | 12.0° | 0.58 m | 3512 | 98.8% | 54.2 | -44.6 | 27.2 |
| `live_000315` | 0.46 m | 11.4° | 0.54 m | 3667 | 98.2% | 44.1 | -54.1 | 21.6 |
| `live_000325` | 0.44 m | 11.4° | 0.56 m | 3644 | 97.4% | 32.6 | -64.8 | 16.0 |
| `live_000330` | 0.44 m | 12.1° | 0.57 m | 3677 | 97.9% | 33.0 | -64.9 | 22.7 |
| `live_000320` | 0.44 m | 11.7° | 0.56 m | 3676 | 97.6% | 30.5 | -67.1 | 26.2 |

> The highest **raw** SC in the sweep is 100.0, on frame live_000170 — whose scored set is 100% occupied, i.e. 1069 voxels of solid furniture with nothing empty to find. Rank by `margin`, not by SC.

## What separates a good viewpoint from a bad one

Over these 159 frames SC and SSC correlate **+0.50**, so the two are loosely related: ranking the viewpoints by one does not rank them by the other.

Correlation of each measured property of the shot with the two scores. These are measurements of THIS room, not general claims:

The numbers are Pearson correlations, not percentages: **+1** means the score rises in step with that property, **-1** that it falls in step, **0** that there is no straight-line relationship. Squaring one gives the share of the spread it accounts for, so +0.70 is about half of it and +0.30 about a tenth. They are associations, not causes, and they only see straight lines -- a property that helps up to a point and hurts after it (the bands below) reports near zero.

The **per step** columns are the same relationship in this room's own units: the slope of the least-squares line through the frames, read over the step in the second column. It is the correlation with the units left in (slope = r x std(score) / std(property)), so it says what the score did ON AVERAGE across the sweep as that property changed -- not what any one frame would score, and only inside the range the sweep covered.

| property of the viewpoint | a step of | vs SC | SC per step | vs SSC | SSC per step |
| --- | --- | ---: | ---: | ---: | ---: |
| camera height above the floor | +10 cm | +0.53 | +3.8 | +0.55 | +3.1 |
| how far the camera looks down | +5 deg | -0.08 | -1.0 | -0.38 | -3.5 |
| median distance to what it sees | +10 cm | +0.40 | +1.4 | +0.61 | +1.7 |
| share of the frame closer than 1.5 m | +10 points | -0.49 | -3.3 | -0.70 | -3.8 |
| room hidden behind the visible surface | +10 cm | -0.56 | -6.0 | -0.62 | -5.3 |
| size of the SC set | +1000 voxels | -0.35 | -3.8 | -0.23 | -2.0 |

Left out, because this sweep barely varied them and a line through a column that does not move says nothing: **how much of the SC set is occupied** (95.6 to 100). They are still measured, and still in the CSV.

The shot property that moves SC most here is **room hidden behind the visible surface** (-0.56, about 31% of the spread in SC); for SSC it is **share of the frame closer than 1.5 m** (-0.70, about 49%). Geometry explains SC less than SSC (mean |correlation| 0.40 against 0.52) -- the mean of the absolute values down each column.

Even the strongest of them leaves a lot unexplained: frames scatter 10.4 SC either side of its line, against an SC range of 30 to 100 across the sweep. These are trends over 159 frames, not rules for one.

Those rows are not separate effects either. The properties move together, so a strong correlation in one row is often the same relationship seen from another angle:

| | camera height above the floor | how far the camera looks down | median distance to what it sees | share of the frame closer than 1.5 m | room hidden behind the visible surface | size of the SC set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| camera height above the floor | +1.00 | -0.20 | +0.66 | -0.69 | -0.52 | -0.13 |
| how far the camera looks down | -0.20 | +1.00 | -0.81 | +0.74 | +0.27 | -0.47 |
| median distance to what it sees | +0.66 | -0.81 | +1.00 | -0.95 | -0.43 | +0.30 |
| share of the frame closer than 1.5 m | -0.69 | +0.74 | -0.95 | +1.00 | +0.66 | -0.04 |
| room hidden behind the visible surface | -0.52 | +0.27 | -0.43 | +0.66 | +1.00 | +0.68 |
| size of the SC set | -0.13 | -0.47 | +0.30 | -0.04 | +0.68 | +1.00 |

The tightest pair here is **median distance to what it sees** and **share of the frame closer than 1.5 m** at -0.95: close enough to be largely one measurement, so the table above cannot say which of the two a score is really following.

The mechanism is the SC set itself: the voxels no depth pixel reached. Group the frames by how much room is hidden behind what they see --

| hidden room behind the surface | frames | median | SC set | occupied | SC | SSC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| least (bottom 25%) | 40 | 0.28 m | 1779 | 100% | 97.7 | 53.5 |
| 25-50% | 41 | 0.40 m | 4482 | 98% | 85.6 | 51.8 |
| 50-75% | 38 | 0.43 m | 4508 | 98% | 87.0 | 46.5 |
| most (top 25%) | 40 | 0.53 m | 3658 | 99% | 77.0 | 44.0 |

And the two ends of the ranking, as medians of the best and worst tenth by `margin`:

| | best 10% | worst 10% |
| --- | ---: | ---: |
| camera height (m) | 1.01 | 0.48 |
| tilt (deg) | 7.20 | 12.10 |
| median depth (m) | 1.98 | 1.03 |
| closer than 1.5 m (%) | 8.80 | 71.80 |
| hidden room behind surface (m) | 0.16 | 0.56 |
| SC set (voxels) | 1069.00 | 3644.00 |
| SC set occupied (%) | 100.00 | 97.60 |
| SC | 99.70 | 64.50 |
| SSC | 53.50 | 28.50 |

What the two ends differ in most, largest first: **share of the frame closer than 1.5 m** (8.80 against 71.80); **room hidden behind the visible surface** (0.16 against 0.56); **camera height above the floor** (1.01 against 0.48).

> **No viewpoint in this room can measure completion.** The least occupied SC set in the whole sweep is still 95.6% occupied, so "predict occupied everywhere" scores at least that on every frame, and the best margin here is +0.2. SC is not a usable number for this room at any viewpoint -- quote SSC, and capture the next room with more depth between the camera and what it looks at. See MAKING_GT.md, "What this does not measure".


### Worked example: room hidden behind the visible surface

![SC against room hidden behind the visible surface](figures/slope_free_mean.png)

Each dot is one sampled frame: where the camera was on that axis, and what CleanerS scored on it. The red line is the least-squares fit, and the slope of that line is the `per step` column.

Every frame contributes one product to the covariance: how far it sits from the average on each axis, multiplied. Both above average, or both below, and the product is positive; one of each and it is negative. Their average is the covariance, which still carries the units of both axes, so dividing by the two standard deviations strips the units out and pins the result between -1 and +1. That is r.

```
room hidden behind the visible surface mean    0.409   std    0.117
SC                           mean     87.0   std     12.5

cov    = mean((property - its mean) x (SC - its mean))
       = -0.8136
r      = cov / (std(property) x std(SC))
       = -0.8136 / (0.1168 x 12.53)
       = -0.5561
slope  = r x std(SC) / std(property)
       = -0.5561 x 12.5 / 0.117
       = -59.6 SC per unit
step   = +10 cm  ->  -59.6 x 0.1 = -6.0 SC per step
```

The same answer without fitting anything, as a check: split the 159 frames at the median room hidden behind the visible surface and compare the halves.

```
lower half: mean 0.326 -> mean SC 91.9
upper half: mean 0.489 -> mean SC 82.2
-9.7 SC over +0.163  ->  -6.0 SC per step (fit said -6.0)
```

That is the green dashed line on the plot. The two agree, so the number is not an artefact of the fit -- but the frames scatter 10.4 SC either side of the line, so it describes the sweep as a whole and predicts no single frame.

### How these were measured

Per frame, in `reconstruction_GT/sweep_eval.py` (`viewpoint_stats`), on the same 640x480 crop the model is given:

- **median distance** and **closer than 1.5 m**: the depth image itself, over valid pixels.
- **floor**: those pixels unprojected with the frame's tracked pose, counted where world z < 12 cm. Recorded in the CSV, but deliberately kept OUT of the correlations above: a room shot from standing height has almost no floor in any frame, and a room where it varies varies it by lowering the camera, which is already a column of its own.
- **hidden room behind the visible surface**: each depth pixel is a ray; it is continued past the surface it hit to where it leaves the ANNOTATED room shell, and the mean of that remaining distance is taken. Zero when the ray dies on a wall or inside furniture, large when it dies on the near side of something with space behind it. Measured against the room in `gt/solids.json`, not against the fused surface, so it is the same volume the ground truth describes.
- **SC set** and **occupied**: the scored set itself -- `label_weight > 0`, `label != 255`, `mapping == 307200` -- and the share of it the ground truth calls occupied, which is what a model predicting "occupied" everywhere would score.

Correlations are Pearson over the 159 sampled frames. They describe this room; a larger room, or one shot from a doorway, would spread differently.

These are scored in memory as the sweep runs, against the same ground truth and the same protocol; where a frame was also written to disk and scored from its files, the two agree to 0.2 SC. `--keep N` writes the N best and N worst out in full -- frame, ground truth, prediction and plys -- and they are the `live_...` rows in the per-frame table above.
