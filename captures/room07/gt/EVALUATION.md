# Evaluation — room07 (camera view only)

Generated 2026-10-08 by `reconstruction_GT/evaluate_gt.py` from `outputs/room07/prediction`, against the ground truth in `captures/room07/gt`. 5 frames.

Protocol is `examples/segmentation/test_NYU.py:206-211`, the same as `inference/document/EVALUATION.md`: **SSC** over voxels where `label_weight > 0` and `label != 255`, mIoU averaged over the classes present; **SC** the same set restricted to `mapping == 307200` — the voxels no depth pixel reached — scored occupied-vs-empty.

`--fov frustum`: only voxels inside the camera view are scored.

| class | TP | FP | FN | precision | recall | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `empty` *(not in mIoU)* | 333 | 3545 | 22 | 8.6% | 93.8% | 8.5% |
| `ceiling` | 4 | 22 | 4 | 15.4% | 50.0% | 13.3% |
| `floor` | 202 | 1318 | 0 | 13.3% | 100.0% | 13.3% |
| `wall` | 2713 | 384 | 2080 | 87.6% | 56.6% | 52.4% |
| `window` | 528 | 72 | 12 | 88.0% | 97.8% | 86.3% |
| `chair` | 76 | 3 | 50 | 96.2% | 60.3% | 58.9% |
| `bed` | 4588 | 1189 | 4140 | 79.4% | 52.6% | 46.3% |
| `sofa` | 0 | 979 | 0 | 0.0% | 0.0% | 0.0% |
| `table` | 161 | 233 | 94 | 40.9% | 63.1% | 33.0% |
| `furn` | 0 | 620 | 0 | 0.0% | 0.0% | 0.0% |
| `objs` | 317 | 134 | 2097 | 70.3% | 13.1% | 12.4% |

**SSC mIoU (8 classes present): 39.5**
  Absent from this ground truth, so not averaged: `sofa`, `tvs`, `furn`.

**SC IoU: 72.4**  |  precision 99.8  recall 72.5  |  13257 voxels, 97% of them occupied

NYU test for reference: SSC 47.7, SC 75.0 (`--nyu` reproduces both).

## Error budget

What is known to be wrong with the numbers above, and by how much. Measured, not estimated; `document/DRIFT_AND_ALIGNMENT.md` has the method for each row.

| term | size | effect on the score |
| --- | ---: | --- |
| **annotation fidelity** — a box standing in for real furniture | 36–56 mm | **the dominant term.** Not a placement error: it is what the ground truth *defines*, so it cannot be corrected, only annotated more finely (`shape: mesh`, smaller solids) |
| frame pose vs the reconstruction | 5–9 mm, corrected on 4 frames | ±0.1 SSC |
| sensor depth noise, single frame | 8–13 mm plane RMS | inside the row above |
| reconstruction self-consistency | ≤2.6 mm | fusing the whole sweep smears a wall by no more than this over one frame's view of it, so the reconstruction is not warped |
| trajectory drift, whole sweep | 156–175 mm | **cancels.** It is a near-global change of world frame, and everything scored here is relative: the annotation and each frame's grid live in the same reconstruction |
| camera height: tape vs floor-plane fit | ~60 mm | the tape's, not the geometry's — the two rooms disagree in *opposite* directions, where a scale error would push both the same way |

> The pose correction on 4 frames is a **polish, not a fix**: it was built expecting a ~100 mm error and the error is 5–9 mm, a tenth of a voxel. It is kept because it is measured and recorded (each frame's `pose_refined` block holds its fitness, RMSE and the values it replaced), not because these numbers needed it. Re-run `refine_frame_pose.py` after any `export_frame`, since a freshly exported frame carries the raw tracked pose.

So the figure worth improving is the annotation, not the poses.


## Per frame

The total above pools every frame, which weights every voxel equally -- and the frames with the most occupied SC sets are exactly the ones SC cannot measure, so they dominate it. Read the **margin** column instead: SC minus what "predict occupied everywhere" scores on that frame's own set, which is its occupancy. Positive means the model beat the trivial answer.

| frame | scored voxels | SC set | occupied | SC | margin | SSC | classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000000` | 5944 | 4412 | 98% | 87.1 | -10.4 | 50.3 | 7 |
| `live_000320` | 4054 | 3419 | 96% | 53.4 | -43.0 | 15.2 | 5 |
| `live_000325` | 4069 | 3444 | 96% | 55.5 | -40.8 | 16.9 | 5 |
| `live_000530` | 1632 | 956 | 100% | 100.0 | +0.0 | 40.5 | 4 |
| `live_000535` | 1722 | 1026 | 100% | 100.0 | +0.0 | 43.0 | 4 |

No frame here has a positive margin, so no viewpoint in this capture measures completion. Quote SSC.

> **This SC number is not usable.** 97% of the SC set is occupied, so a model predicting "occupied" everywhere scores IoU 0.973. The frame cannot separate a good model from a trivial one: the camera saw the whole room, so the only hidden volume left is the inside of the annotated solids. SSC above is still meaningful. See reconstruction_GT/document/MAKING_GT.md.


**What the occupancy figure is, and is not.** It counts only voxels INSIDE the annotated room: everything beyond the shell is 255 and enters neither side of the fraction. So it does not say the room is full -- it says how much of the hidden volume within these walls is furniture and wall interior, which rises as the room gets smaller, because a camera standing in a small room sees nearly all of its free space. It is a measurement of the ANNOTATION, and the shell is the lever: on room07, growing the shell 0.3 m each way moves it from 97.6% to 68.1%, and 0.6 m to 43.0%, without touching a single piece of furniture (wall thickness barely matters: 4 cm vs 2 cm gives 97.6% vs 97.5%). Growing the shell is not a legitimate fix -- those voxels are outside the room, and calling them empty would assert free space where there is a wall. The honest shell sits at the walls, and this number is its consequence.

**SSC is not affected by it.** SSC averages per-class IoU over the classes present and leaves `empty` out of that average, so a mostly-occupied set is what it wants rather than a defect, and most of its set is genuinely hidden -- it measures completion, not visible segmentation.

## Across the sweep

`reconstruction_GT/sweep_eval.py` scored **159 frames** of this room against the same annotation (`outputs/room07/sweep_eval.csv`). One frame gives one number, and that number says as much about the viewpoint as about the model.

| | SC | SSC |
| --- | ---: | ---: |
| best | 100.0 | 64.1 |
| median | 92.6 | 45.1 |
| worst | 53.2 | 15.0 |

`margin` is SC minus what "predict occupied everywhere" would score on that frame, which is its `occupied` column. **Only 0 of 159 frames beat that baseline**, and 159 have a scored set over 90% occupied — in a small room most viewpoints leave the metric nothing to find.

### Best 5 viewpoints (by margin)

| frame | cam z | tilt | hidden | SC set | occupied | SC | margin | SSC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000530` | 0.88 m | 12.2° | 0.19 m | 957 | 100.0% | 100.0 | +0.0 | 39.2 |
| `live_000535` | 0.90 m | 11.7° | 0.19 m | 1053 | 100.0% | 100.0 | +0.0 | 42.4 |
| `live_000540` | 0.95 m | 12.3° | 0.22 m | 1183 | 100.0% | 99.7 | -0.3 | 41.6 |
| `live_000150` | 1.03 m | 5.0° | 0.20 m | 1577 | 100.0% | 99.6 | -0.4 | 45.1 |
| `live_000155` | 1.03 m | 4.9° | 0.15 m | 1090 | 100.0% | 99.6 | -0.4 | 45.7 |

### Worst 5 viewpoints

| frame | cam z | tilt | hidden | SC set | occupied | SC | margin | SSC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `live_000465` | 0.90 m | 14.0° | 0.52 m | 3193 | 94.7% | 66.7 | -27.9 | 46.2 |
| `live_000460` | 0.90 m | 14.1° | 0.52 m | 3159 | 94.9% | 66.9 | -28.0 | 45.0 |
| `live_000455` | 0.91 m | 14.2° | 0.51 m | 3198 | 94.5% | 65.5 | -29.0 | 43.9 |
| `live_000325` | 0.46 m | 13.0° | 0.55 m | 3441 | 96.4% | 55.4 | -41.0 | 17.1 |
| `live_000320` | 0.46 m | 13.4° | 0.55 m | 3422 | 96.4% | 53.2 | -43.2 | 15.0 |

> The highest **raw** SC in the sweep is 100.0, on frame live_000530 — whose scored set is 100% occupied, i.e. 957 voxels of solid furniture with nothing empty to find. Rank by `margin`, not by SC.

## What separates a good viewpoint from a bad one

Over these 159 frames SC and SSC correlate **+0.09**, so the two are barely related: ranking the viewpoints by one does not rank them by the other.

Correlation of each measured property of the shot with the two scores. These are measurements of THIS room, not general claims:

The numbers are Pearson correlations, not percentages: **+1** means the score rises in step with that property, **-1** that it falls in step, **0** that there is no straight-line relationship. Squaring one gives the share of the spread it accounts for, so +0.70 is about half of it and +0.30 about a tenth. They are associations, not causes, and they only see straight lines -- a property that helps up to a point and hurts after it (the bands below) reports near zero.

The **per step** columns are the same relationship in this room's own units: the slope of the least-squares line through the frames, read over the step in the second column. It is the correlation with the units left in (slope = r x std(score) / std(property)), so it says what the score did ON AVERAGE across the sweep as that property changed -- not what any one frame would score, and only inside the range the sweep covered.

| property of the viewpoint | a step of | vs SC | SC per step | vs SSC | SSC per step |
| --- | --- | ---: | ---: | ---: | ---: |
| camera height above the floor | +10 cm | +0.20 | +1.0 | +0.60 | +2.9 |
| how far the camera looks down | +5 deg | +0.05 | +0.4 | -0.51 | -3.9 |
| median distance to what it sees | +10 cm | +0.06 | +0.1 | +0.63 | +1.5 |
| share of the frame closer than 1.5 m | +10 points | -0.19 | -0.9 | -0.69 | -3.1 |
| room hidden behind the visible surface | +10 cm | -0.49 | -3.6 | -0.56 | -3.9 |
| size of the SC set | +1000 voxels | -0.45 | -3.5 | -0.10 | -0.8 |

Left out, because this sweep barely varied them and a line through a column that does not move says nothing: **how much of the SC set is occupied** (94.2 to 100). They are still measured, and still in the CSV.

The shot property that moves SC most here is **room hidden behind the visible surface** (-0.49, about 24% of the spread in SC); for SSC it is **share of the frame closer than 1.5 m** (-0.69, about 47%). Geometry explains SC less than SSC (mean |correlation| 0.24 against 0.51) -- the mean of the absolute values down each column.

Even the strongest of them leaves a lot unexplained: frames scatter 7.5 SC either side of its line, against an SC range of 53 to 100 across the sweep. These are trends over 159 frames, not rules for one.

Those rows are not separate effects either. The properties move together, so a strong correlation in one row is often the same relationship seen from another angle:

| | camera height above the floor | how far the camera looks down | median distance to what it sees | share of the frame closer than 1.5 m | room hidden behind the visible surface | size of the SC set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| camera height above the floor | +1.00 | -0.28 | +0.68 | -0.70 | -0.50 | -0.09 |
| how far the camera looks down | -0.28 | +1.00 | -0.84 | +0.79 | +0.32 | -0.42 |
| median distance to what it sees | +0.68 | -0.84 | +1.00 | -0.95 | -0.40 | +0.32 |
| share of the frame closer than 1.5 m | -0.70 | +0.79 | -0.95 | +1.00 | +0.64 | -0.04 |
| room hidden behind the visible surface | -0.50 | +0.32 | -0.40 | +0.64 | +1.00 | +0.68 |
| size of the SC set | -0.09 | -0.42 | +0.32 | -0.04 | +0.68 | +1.00 |

The tightest pair here is **median distance to what it sees** and **share of the frame closer than 1.5 m** at -0.95: close enough to be largely one measurement, so the table above cannot say which of the two a score is really following.

The mechanism is the SC set itself: the voxels no depth pixel reached. Group the frames by how much room is hidden behind what they see --

| hidden room behind the surface | frames | median | SC set | occupied | SC | SSC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| least (bottom 25%) | 41 | 0.27 m | 1647 | 100% | 98.3 | 45.9 |
| 25-50% | 41 | 0.39 m | 4293 | 97% | 87.5 | 48.3 |
| 50-75% | 37 | 0.41 m | 4123 | 98% | 91.6 | 46.0 |
| most (top 25%) | 40 | 0.53 m | 3440 | 97% | 88.8 | 35.0 |

And the two ends of the ranking, as medians of the best and worst tenth by `margin`:

| | best 10% | worst 10% |
| --- | ---: | ---: |
| camera height (m) | 1.02 | 0.90 |
| tilt (deg) | 7.00 | 14.10 |
| median depth (m) | 1.99 | 1.68 |
| closer than 1.5 m (%) | 10.30 | 40.60 |
| hidden room behind surface (m) | 0.19 | 0.53 |
| SC set (voxels) | 1047.00 | 3411.00 |
| SC set occupied (%) | 100.00 | 94.90 |
| SC | 99.40 | 69.00 |
| SSC | 45.60 | 45.10 |

What the two ends differ in most, largest first: **room hidden behind the visible surface** (0.19 against 0.53); **size of the SC set** (1047.00 against 3411.00); **share of the frame closer than 1.5 m** (10.30 against 40.60).

> **No viewpoint in this room can measure completion.** The least occupied SC set in the whole sweep is still 94.2% occupied, so "predict occupied everywhere" scores at least that on every frame, and the best margin here is +0.0. SC is not a usable number for this room at any viewpoint -- quote SSC, and capture the next room with more depth between the camera and what it looks at. See reconstruction_GT/document/MAKING_GT.md, "What this does not measure".


### The same thing measured on the benchmarks

NYU (what CleanerS was trained and evaluated on) and Occ-ScanNet, put through this identical measurement: NYU 654 frames, ScanNet 1500 frames.


**Correlation with SC**

| property of the viewpoint | here | NYU | ScanNet |
| --- | ---: | ---: | ---: |
| camera height above the floor | +0.20 | +0.01 | -0.10 |
| how far the camera looks down | +0.05 | +0.18 | -0.10 |
| median distance to what it sees | +0.06 | -0.15 | +0.01 |
| share of the frame closer than 1.5 m | -0.19 | +0.01 | -0.02 |
| room hidden behind the visible surface | -0.49 | +0.08 | +0.08 |
| size of the SC set | -0.45 | -0.36 | -0.47 |
| **mean \|correlation\|** | **0.24** | **0.13** | **0.13** |

**Correlation with SSC**

| property of the viewpoint | here | NYU | ScanNet |
| --- | ---: | ---: | ---: |
| camera height above the floor | +0.60 | +0.05 | -0.07 |
| how far the camera looks down | -0.51 | +0.11 | -0.13 |
| median distance to what it sees | +0.63 | -0.11 | +0.02 |
| share of the frame closer than 1.5 m | -0.69 | +0.00 | -0.01 |
| room hidden behind the visible surface | -0.56 | +0.08 | +0.13 |
| size of the SC set | -0.10 | -0.13 | -0.24 |
| **mean \|correlation\|** | **0.51** | **0.08** | **0.10** |

Viewpoint matters far more here than in the benchmarks (mean |correlation| with SC 0.24 against 0.13). The rows above therefore describe this capture, not the metric: in a large scene the shot barely predicts the score.

| | this room | NYU | ScanNet |
| --- | ---: | ---: | ---: |
| median camera height | 1.03 m | 1.34 m | 1.43 m |
| median distance to the scene | 1.85 m | 2.49 m | 1.77 m |
| median room hidden behind the surface | 0.40 m | 1.72 m | 1.64 m |
| median SC set occupied | 99% | 55% | 45% |
| median SC set | 3441 voxels | 10811 voxels | 6226 voxels |

The gap that drives the rest is **how much room is hidden behind what the camera sees**: NYU 1.72 m, ScanNet 1.64 m, this room 0.40 m. A frame that hides nothing cannot be asked to complete anything, and a room too small to hide anything cannot produce such a frame.

### Worked example: room hidden behind the visible surface

![SC against room hidden behind the visible surface](figures/slope_free_mean.png)

Each dot is one sampled frame: where the camera was on that axis, and what CleanerS scored on it. The red line is the least-squares fit, and the slope of that line is the `per step` column.

Every frame contributes one product to the covariance: how far it sits from the average on each axis, multiplied. Both above average, or both below, and the product is positive; one of each and it is negative. Their average is the covariance, which still carries the units of both axes, so dividing by the two standard deviations strips the units out and pins the result between -1 and +1. That is r.

```
room hidden behind the visible surface mean    0.398   std    0.118
SC                           mean     90.3   std      8.6

cov    = mean((property - its mean) x (SC - its mean))
       = -0.5008
r      = cov / (std(property) x std(SC))
       = -0.5008 / (0.1182 x 8.58)
       = -0.4938
slope  = r x std(SC) / std(property)
       = -0.4938 x 8.6 / 0.118
       = -35.9 SC per unit
step   = +10 cm  ->  -35.9 x 0.1 = -3.6 SC per step
```

The same answer without fitting anything, as a check: split the 159 frames at the median room hidden behind the visible surface and compare the halves.

```
lower half: mean 0.313 -> mean SC 93.4
upper half: mean 0.479 -> mean SC 87.5
-5.9 SC over +0.166  ->  -3.5 SC per step (fit said -3.6)
```

That is the green dashed line on the plot. The two agree, so the number is not an artefact of the fit -- but the frames scatter 7.5 SC either side of the line, so it describes the sweep as a whole and predicts no single frame.

### How these were measured

Per frame, in `reconstruction_GT/sweep_eval.py` (`viewpoint_stats`), on the same 640x480 crop the model is given:

- **median distance** and **closer than 1.5 m**: the depth image itself, over valid pixels.
- **floor**: those pixels unprojected with the frame's tracked pose, counted where world z < 12 cm. Recorded in the CSV, but deliberately kept OUT of the correlations above: a room shot from standing height has almost no floor in any frame, and a room where it varies varies it by lowering the camera, which is already a column of its own.
- **hidden room behind the visible surface**: each depth pixel is a ray; it is continued past the surface it hit to where it leaves the ANNOTATED room shell, and the mean of that remaining distance is taken. Zero when the ray dies on a wall or inside furniture, large when it dies on the near side of something with space behind it. Measured against the room in `gt/solids.json`, not against the fused surface, so it is the same volume the ground truth describes.
- **SC set** and **occupied**: the scored set itself -- `label_weight > 0`, `label != 255`, `mapping == 307200` -- and the share of it the ground truth calls occupied, which is what a model predicting "occupied" everywhere would score.

Correlations are Pearson over the 159 sampled frames. They describe this room; a larger room, or one shot from a doorway, would spread differently.

These are scored in memory as the sweep runs, against the same ground truth and the same protocol; where a frame was also written to disk and scored from its files, the two agree to 0.2 SC. `--keep N` writes the N best and N worst out in full -- frame, ground truth, prediction and plys -- and they are the `live_...` rows in the per-frame table above.
