# Occ-ScanNet evaluation

Generated 2026-09-23 by `inference/evaluate_scannet.py` from `./outputs/scannet`; rerun it after regenerating the predictions. CleanerS checkpoint `CleanerS_ckpt.pth`, trained on NYU only — every number here is a zero-shot transfer to ScanNet. Inputs use ScanNet's own per-scene depth calibration (`scans/<scene>/<scene>.txt`).

Protocol is `examples/segmentation/test_NYU.py`'s, the same as `EVALUATION.md`, applied to the released Occ-ScanNet labels (`target`): **SSC** over voxels where `label_weight > 0` and `label != 255`, mIoU averaged over classes 1-11; **SC** the same restricted to `mapping == 307200`, the voxels no depth pixel reached, scored occupied-vs-empty. `label_weight` is rebuilt the way the NYU files were made: GT object, or `tsdf < -0.5`.

`precision = TP/(TP+FP)` — of what was predicted, how much was right. `recall = TP/(TP+FN)` — of what was there, how much was found. `IoU = TP/(TP+FP+FN)` — both failures in one number, which is why it is the headline.

Other scoring rules — the all-voxel rule Occ-ScanNet papers use, and a view-restricted GT — follow after the per-class sections, together with the comparison to ISO and what differs from NYU.

## `val` split — 19764 frames, 681 scenes

| class | TP | FP | FN | precision | recall | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `empty` *(not in mIoU)* | 77,547,097 | 12,628,912 | 15,667,724 | 86.0% | 83.2% | 73.3% |
| `ceiling` | 168,367 | 176,659 | 57,511 | 48.8% | 74.5% | 41.8% |
| `floor` | 3,181,343 | 9,153,435 | 10,511,401 | 25.8% | 23.2% | 13.9% |
| `wall` | 7,532,582 | 4,597,270 | 4,868,675 | 62.1% | 60.7% | 44.3% |
| `window` | 198,874 | 500,772 | 816,130 | 28.4% | 19.6% | 13.1% |
| `chair` | 1,844,076 | 1,235,156 | 4,710,151 | 59.9% | 28.1% | 23.7% |
| `bed` | 1,134,782 | 1,722,916 | 1,875,103 | 39.7% | 37.7% | 24.0% |
| `sofa` | 2,300,956 | 3,783,496 | 1,763,669 | 37.8% | 56.6% | 29.3% |
| `table` | 2,941,406 | 3,909,954 | 4,092,112 | 42.9% | 41.8% | 26.9% |
| `tvs` | 20,492 | 28,592 | 58,939 | 41.7% | 25.8% | 19.0% |
| `furn` | 3,869,735 | 8,526,369 | 5,300,646 | 31.2% | 42.2% | 21.9% |
| `objs` | 698,619 | 5,717,314 | 2,258,784 | 10.9% | 23.6% | 8.1% |

**SSC mIoU (classes 1-11): 24.2**  |  mean precision 39.0  |  mean recall 39.5

**SC (binary, unobserved region only): IoU 49.1  |  precision 63.5  |  recall 68.4**

### What the SC IoU is made of

**Every number in this section is computed from this run's own predictions** — `./outputs/scannet`, 19,764 `val` frames — against the Occ-ScanNet labels. None of it is copied from the paper.

SC IoU of **49.1%** does **not** mean 49.1% of voxels are right and the rest wrong. It is a share of one particular set, and empty voxels correctly left empty are not in that set. Step by step:

**1. Which voxels are scored.**

| filter | voxels left | % of all |
| --- | ---: | ---: |
| every voxel (19,764 frames × 60 × 36 × 60) | 2,561,414,400 | 100% |
| `label_weight` true (GT object, or hidden behind a surface) | 608,307,785 | 23.7% |
| and GT ≠ `255` (annotated) — the SSC set | 153,419,174 | 6.0% |
| and **no depth pixel landed in it** — the SC set | **133,129,938** | **5.2%** |

The last filter drops the surfaces the camera saw directly. SC only scores what the model had to **infer**.

**2. Collapse to occupied vs empty.** Occupied is any class 1–11, in the prediction and in the GT alike. Class names do not matter here: a chair predicted as a table counts as correct.

| | GT occupied | GT `empty` |
| --- | ---: | ---: |
| **predicted occupied** | TP **27,286,692** | FP **15,667,724** |
| **predicted `empty`** | FN **12,628,425** | TN **77,547,097** |

Counts are pooled over all frames, not averaged per frame.

**3. The IoU.**

```
IoU = TP / (TP + FP + FN) = 27,286,692 / 55,582,841 = 49.1%
```

The denominator is the **union**: every voxel occupied in the GT, in the prediction, or both. TN — the 77,547,097 empty voxels correctly left empty — does not appear in it at all.

**4. So the other 50.9% is** the part of the union where the two disagree:

| share of the union | what it is | voxels |
| ---: | --- | ---: |
| **49.1%** | both say occupied (TP) | 27,286,692 |
| **22.7%** | **missed** — GT occupied, predicted `empty` (FN) | 12,628,425 |
| **28.2%** | **invented** — predicted occupied, GT `empty` (FP) | 15,667,724 |

- **Missed**, by GT class: `floor` 27%, `furn` 19%, `chair` 13%, `wall` 12%, `table` 12%.
- **Invented**, by the class the model gave: `floor` 55%, `furn` 20%, `bed` 8%, `sofa` 5%, `wall` 5%.
- **The same errors over the whole SC set**, TN included: (15,667,724 + 12,628,425) / 133,129,938 = **21.3% of SC voxels wrong**. IoU is stricter than that accuracy-style figure because it leaves out the easy empty-and-correct voxels.
- **Precision 63.5%** — of what the model filled, how much was really occupied: `TP / (TP + FP)`. **Recall 68.4%** — of what was really occupied, how much it filled: `TP / (TP + FN)`.

Every class's share of the missed and invented voxels:

| class | missed (GT this class, predicted `empty`) | % of all missed | invented (predicted this class, GT `empty`) | % of all invented |
| --- | ---: | ---: | ---: | ---: |
| `ceiling` | 11,573 | 0.1% | 45,174 | 0.3% |
| `floor` | 3,444,612 | 27.3% | 8,663,684 | 55.3% |
| `wall` | 1,572,263 | 12.5% | 814,251 | 5.2% |
| `window` | 128,261 | 1.0% | 46,730 | 0.3% |
| `chair` | 1,609,402 | 12.7% | 65,731 | 0.4% |
| `bed` | 654,079 | 5.2% | 1,274,257 | 8.1% |
| `sofa` | 790,387 | 6.3% | 843,838 | 5.4% |
| `table` | 1,481,464 | 11.7% | 309,429 | 2.0% |
| `tvs` | 1,696 | 0.0% | 1,928 | 0.0% |
| `furn` | 2,455,115 | 19.4% | 3,066,240 | 19.6% |
| `objs` | 479,573 | 3.8% | 536,462 | 3.4% |

### Scene completion (SC) per class

SC above is one binary number. This breaks it down per class, over the same region: scored voxels **no depth pixel reached** — the part of each object the model has to complete rather than read off the input.

**The region.** Every voxel below is *scored* (`label_weight` true, GT not
`255`) **and** has no depth pixel in it (`mapping == 307200`). Call a class
`c`; "predicted empty" means the model output class 0.

**What each column means:**

| column | what it counts |
| --- | --- |
| **unseen GT voxels** | voxels in the region whose GT is `c` |
| **% of the class unseen** | *unseen GT voxels* ÷ all scored GT voxels of `c`, seen or not. How much of the class the model has to complete rather than read off the input |
| **IoU, all scored** | the per-class IoU from the metrics table above, over every scored voxel whether seen or not. Copied here for comparison |
| **SC IoU, label must match** | the ordinary IoU inside the region. TP = GT `c`, predicted `c`. FP = predicted `c`, GT anything else (including `empty`). FN = GT `c`, predicted anything else (including `empty`). `TP / (TP + FP + FN)` |
| **SC IoU, any class counts** | occupancy IoU inside the region — did the model fill it, whatever it called it. TP = GT `c`, predicted **any** non-empty class. FP = predicted `c`, GT `empty`. FN = GT `c`, predicted `empty`. A voxel predicted `c` whose GT is *another* class is not an error here: occupied was right |
| **unseen, filled as another class** | GT `c`, predicted a different non-empty class, ÷ *unseen GT voxels*. Filled, but misnamed |
| **unseen, left empty** | GT `c`, predicted `empty`, ÷ *unseen GT voxels*. Not filled at all |

The last two columns and the correctly labelled share add up to 100% of the
class's unseen voxels, so the correct share is 100% minus both. They describe
misses only — false positives appear in the two IoU columns, not here.

The gap between *label must match* and *any class counts* comes from exactly
two kinds of voxel, both counted against the first and neither against the
second: the class's own unseen voxels filled as another class (the
*filled as another class* column), and other classes' unseen voxels predicted
as `c`. Both are volume the model completed but misnamed.

**The two bottom rows.** *mean, classes 1-11* is the plain average of the
eleven rows, every class weighted equally — the same way SSC mIoU is taken.
*pooled* adds up TP, FP and FN over all eleven classes **before** dividing, so
large classes weigh more. For *any class counts* that pooled sum is exactly the
binary SC matrix — the headline **SC IoU** printed under the metrics table —
which the script asserts on every run. Mean and pooled differ only because of
that weighting.

| class | unseen GT voxels | % of the class unseen | IoU, all scored | SC IoU, label must match | SC IoU, any class counts | unseen, filled as another class | unseen, left empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ceiling` | 116,393 | 52% | 41.8% | 37.7% | 64.9% | 20% | 10% |
| `floor` | 8,612,139 | 63% | 13.9% | 7.5% | 29.9% | 45% | 40% |
| `wall` | 6,279,870 | 51% | 44.3% | 33.8% | 66.4% | 24% | 25% |
| `window` | 607,102 | 60% | 13.1% | 11.1% | 73.2% | 64% | 21% |
| `chair` | 4,910,593 | 75% | 23.7% | 20.5% | 66.3% | 43% | 33% |
| `bed` | 2,394,677 | 80% | 24.0% | 23.3% | 47.4% | 34% | 27% |
| `sofa` | 3,189,036 | 78% | 29.3% | 29.1% | 59.5% | 21% | 25% |
| `table` | 5,045,967 | 72% | 26.9% | 23.1% | 66.6% | 36% | 29% |
| `tvs` | 47,539 | 60% | 19.0% | 15.6% | 92.7% | 75% | 4% |
| `furn` | 6,828,156 | 74% | 21.9% | 19.6% | 44.2% | 26% | 36% |
| `objs` | 1,883,645 | 64% | 8.1% | 7.1% | 58.0% | 55% | 25% |
| **mean, classes 1-11** | | | **24.2%** | **20.8%** | **60.8%** | | |
| **pooled** — the headline SC IoU | | | | | **49.1%** | | |

## Floor removed

Occ-ScanNet places the grid 5 cm below world *z = 0* instead of below the annotated floor (NYU's rule), so the ScanNet floor usually lands 1–2 layers above the layer the model learned it in. That is a dataset-construction difference, not a model error, so the same protocol is also scored with floor taken out: a voxel that is floor in the GT **or** in the prediction is not scored, and mIoU is over the other 10 classes. The NYU test split is scored the same way for comparison.

| class | TP | FP | FN | precision | recall | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `empty` *(not in mIoU)* | 77,547,097 | 9,184,075 | 7,004,040 | 89.4% | 91.7% | 82.7% |
| `ceiling` | 168,367 | 113,253 | 57,511 | 59.8% | 74.5% | 49.6% |
| `floor` | 0 | 0 | 0 | -- | -- | -- |
| `wall` | 7,532,582 | 3,817,688 | 4,794,703 | 66.4% | 61.1% | 46.7% |
| `window` | 198,874 | 484,335 | 814,363 | 29.1% | 19.6% | 13.3% |
| `chair` | 1,844,076 | 587,163 | 4,631,959 | 75.8% | 28.5% | 26.1% |
| `bed` | 1,134,782 | 1,552,987 | 1,845,898 | 42.2% | 38.1% | 25.0% |
| `sofa` | 2,300,956 | 2,541,279 | 1,717,450 | 47.5% | 57.3% | 35.1% |
| `table` | 2,941,406 | 2,283,521 | 4,002,687 | 56.3% | 42.4% | 31.9% |
| `tvs` | 20,492 | 28,264 | 58,939 | 42.0% | 25.8% | 19.0% |
| `furn` | 3,869,735 | 6,963,716 | 5,181,625 | 35.7% | 42.8% | 24.2% |
| `objs` | 698,619 | 4,759,728 | 2,206,834 | 12.8% | 24.0% | 9.1% |

**SSC mIoU (classes 1-11 except floor): 28.0**  |  mean precision 46.8  |  mean recall 41.4

**SC (binary, unobserved region only): IoU 57.3  |  precision 75.6  |  recall 70.3**

| | SC IoU | SC precision | SC recall | SSC mIoU |
| --- | ---: | ---: | ---: | ---: |
| Occ-ScanNet val | 49.1 | 63.5 | 68.4 | 24.2 |
| Occ-ScanNet val, floor removed | 57.3 | 75.6 | 70.3 | 28.0 |
| NYU test | 75.0 | 88.0 | 83.5 | 47.7 |
| NYU test, floor removed | 70.2 | 85.2 | 79.9 | 43.5 |

## All scoring rules at a glance

| scored set | GT | SC IoU | SC precision | SC recall | SSC mIoU |
| --- | --- | ---: | ---: | ---: | ---: |
| all-voxels | `target` | 6.8 | 6.9 | 79.0 | 4.4 |
| all-voxels | `target_fov` | 23.8 | 25.4 | 79.0 | 10.9 |
| cleaners | `target` | 49.1 | 63.5 | 68.4 | 24.2 |
| cleaners | `target_fov` | 49.3 | 63.9 | 68.4 | 24.2 |
| cleaners, **floor removed** | `target` | 57.3 | 75.6 | 70.3 | 28.0 |
| cleaners, **floor removed** | `target_fov` | 57.5 | 76.0 | 70.3 | 28.0 |
| *NYU test, cleaners* | *NYU `Label`* | *75.0* | *88.0* | *83.5* | *47.7* |
| *NYU test, cleaners, floor removed* | *NYU `Label`* | *70.2* | *85.2* | *79.9* | *43.5* |

- **all-voxels / `target`** is how Occ-ScanNet papers score, so it is the row to put next to them.
- **`target_fov`** marks voxels outside the colour camera's view as 255. The released GT calls them `empty` although no camera looked at them, so predicting anything there is scored wrong.
- **cleaners** is CleanerS's own NYU protocol: only GT objects plus occluded space (`tsdf < -0.5`) are scored, and SC only where no depth pixel landed.
- **floor removed**: the same, but a voxel that is floor in the GT *or* in the prediction is not scored, and mIoU is over the other 10 classes. Occ-ScanNet places the grid 5 cm below world *z = 0* instead of below the annotated floor (NYU's rule), so its floor usually lands 1–2 layers above the layer the model learned from NYU; removing floor takes that construction difference out of the comparison. The NYU rows apply the same removal to CleanerS's own test split.

## NYU-like views

Only frames whose camera looks like an NYU frame: pitch -23.3° to -0.1° and camera height 1.07 to 1.51 m, the 5th–95th percentile of NYU's own 1449 frames. ScanNet is hand-held and looks down more (median pitch -26.1° here vs -13.5° on NYU); grid placement is the same recipe on both (centre ~3.3 m ahead, heading spread 0–45°), so pitch is the viewpoint difference that matters. **4089 of 19764 frames** qualify.

| scored set | GT | all frames: SC IoU | all frames: SSC mIoU | NYU-like: SC IoU | NYU-like: SSC mIoU |
| --- | --- | ---: | ---: | ---: | ---: |
| all-voxels | `target` | 6.8 | 4.4 | **8.0** | **5.0** |
| all-voxels | `target_fov` | 23.8 | 10.9 | **23.6** | **11.6** |
| cleaners | `target` | 49.1 | 24.2 | **49.9** | **22.7** |
| cleaners | `target_fov` | 49.3 | 24.2 | **50.2** | **22.7** |

For reference, CleanerS on NYU test with the cleaners protocol: SC 75.0 / SSC mIoU 47.7. The **cleaners / NYU-like** row is the closest like-for-like this data allows: same scoring rule, similar viewpoints. What still differs is the GT recipe (CompleteScanNet nearest-neighbour voxels vs NYU's 2 cm CAD solids with the 4×4×4 rule) and the sensor.

Per class, cleaners / `target_fov`, NYU-like views:

| ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furn | objs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6.0 | 10.4 | 46.1 | 12.1 | 27.0 | 31.7 | 32.7 | 28.0 | 18.7 | 26.8 | 10.3 |

## Against published results

| method | trained on | scored set | IoU | ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furn | objs | mIoU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CleanerS (ours)** | NYU | all-voxels | **6.8** | 0.5 | 7.7 | 3.0 | 2.2 | 3.8 | 11.9 | 5.9 | 7.6 | 3.1 | 2.6 | 0.6 | **4.4** |
| ISO | Occ-ScanNet | all-voxels | 42.2 | 19.9 | 41.9 | 22.4 | 17.0 | 29.1 | 42.4 | 42.0 | 29.6 | 10.6 | 36.4 | 24.6 | 28.7 |
| MonoScene | Occ-ScanNet | all-voxels | 41.6 | 15.2 | 44.7 | 22.4 | 12.6 | 26.1 | 27.0 | 35.9 | 28.3 | 6.6 | 32.2 | 19.8 | 24.6 |
| **CleanerS (ours)** | NYU | cleaners, `target_fov` | **49.3** | 41.8 | 14.0 | 44.3 | 13.1 | 23.7 | 24.1 | 29.4 | 26.9 | 19.0 | 22.0 | 8.1 | **24.2** |
| CleanerS, NYU test (654 frames) | NYU | cleaners | 75.0 | 46.3 | 93.9 | 43.2 | 33.7 | 38.5 | 62.2 | 54.8 | 33.8 | 39.2 | 45.7 | 33.8 | 47.7 |
| CleanerS, NYU test (654 frames) | NYU | all-voxels | 14.7 | 2.3 | 69.2 | 7.3 | 8.4 | 4.4 | 32.9 | 16.6 | 7.0 | 11.3 | 12.8 | 3.8 | 16.0 |

ISO and MonoScene numbers are the ISO paper's Table 1 (RGB input, trained on Occ-ScanNet train). CleanerS uses RGB **and** sensor depth but has never seen ScanNet, so the comparison mixes an input advantage with a domain gap.

The two NYU rows score **the same predictions** on CleanerS's own test split. The cleaners row reproduces `document/EVALUATION.md`; the all-voxels row is what the Occ-ScanNet protocol does to them with no change of dataset. The model fills most of the space the camera saw through with objects, and NYU's `label_weight` never scores that space, so the protocol alone accounts for part of the drop.

## Why ScanNet still scores below NYU: annotation or model?

Same scoring rule, so the remaining gap is the ground truth, the model, or the input. Two checks separate them.

### 1. Floor height — the grid assumption breaks, and the model follows a habit

Share of floor voxels in each height layer (layer 0 = 5 cm below the grid's assumed floor to 3 cm above it; one layer = 8 cm):

| | layer 0 | layer 1 | layer 2 | layer 3 | layers 4+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| NYU GT | 98% | 2% | 0% | 0% | 0% |
| NYU prediction | 95% | 4% | 0% | 0% | 0% |
| Occ-ScanNet GT | 13% | 38% | 31% | 7% | 11% |
| Occ-ScanNet prediction | 96% | 3% | 1% | 0% | 0% |

GT floor voxels are predicted as floor 98% of the time on NYU, and 23% on Occ-ScanNet (there: `empty` 25%, `floor` 23%, `table` 12%, `furn` 11%).

- **Annotation / dataset construction.** Occ-ScanNet puts the bottom of every grid 5 cm below world *z = 0* (`height_belowfloor = -0.05` in its `generate_gt.py`), assuming the ScanNet floor sits at 0. NYU places it 5 cm below the *annotated* floor, so NYU's floor is always layer 0. ScanNet scans are not built that way, and the GT floor lands mostly one or two layers up.
- **Model.** The depth it is given shows the floor at that higher layer — the same grid is used for the input — yet it still paints floor in layer 0. It learned "floor = bottom layer" from NYU, where that was always true, rather than reading the height from depth.
- Every other class's height relative to the floor shifts with it, so this is not only a floor problem. It is fixable without new data: place each grid 5 cm below the floor the GT annotates, as NYU does, re-encode and re-run.

### 2. Recognition on measured surfaces — the model and the transfer

Only voxels a depth pixel landed in **and** the GT calls an object. Nothing here needs completing, so hidden-part annotation cannot affect it.

| | right class | ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furn | objs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NYU recall | 76.7% | 85 | 99 | 80 | 68 | 75 | 76 | 78 | 69 | 63 | 69 | 57 |
| Occ-ScanNet recall | 52.6% | 79 | 36 | 71 | 26 | 41 | 33 | 66 | 60 | 32 | 55 | 31 |
| change | -24.1 | -6 | -63 | -9 | -42 | -34 | -43 | -12 | -8 | -31 | -14 | -25 |

Classes that drop a little are ordinary transfer loss: a new sensor and new rooms. The largest drops besides floor (`bed`, `window`, `chair`) are too big for that alone and may come from class definitions — CompleteScanNet's categories mapped onto NYU's 11 need not match what NYU's annotators called those objects. Not verified: that needs CompleteScanNet's raw labels.

### 3. What is left

Completion of hidden parts is scored against a different recipe — CompleteScanNet CAD voxels copied onto the 8 cm grid by nearest neighbour, against NYU's 2 cm solids with the 4×4×4 rule — and 14.9% of Occ-ScanNet's GT object voxels sit where the depth sensor measured free space. Separating that from the model needs GT rebuilt NYU's way from CompleteScanNet.

## RGB-D CleanerS vs RGB-only ISO

ISO scores every voxel whose GT is not 255 (`iso/models/iso.py` calls `metric.add_batch(y_pred, y_true)` with no mask; `sscMetrics.add_batch` keeps `y_true != 255`; mIoU over classes 1–11). That is this report's **all-voxels / `target`** row, so the scoring below is ISO's, not CleanerS's.

### 1. NYUv2 test — same training data, same scoring

Both models were trained on NYUv2, so this is the fair test of RGB-D against RGB-only.

| method | input | trained on | SC IoU | ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furn | objs | mIoU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CleanerS** (654 frames, rescored) | RGB + depth | NYUv2 | **14.7** | 2.3 | 69.2 | 7.3 | 8.4 | 4.4 | 32.9 | 16.6 | 7.0 | 11.3 | 12.8 | 3.8 | **16.0** |
| **CleanerS + free space from depth** | RGB + depth | NYUv2 | **17.3** | 1.8 | 67.8 | 7.0 | 8.1 | 4.9 | 40.0 | 20.7 | 6.2 | 11.1 | 14.1 | 4.6 | **16.9** |
| ISO | RGB | NYUv2 | 47.1 | 14.2 | 93.5 | 15.9 | 15.1 | 18.4 | 50.0 | 40.8 | 18.2 | 25.9 | 34.1 | 17.7 | 31.2 |
| NDC-Scene | RGB | NYUv2 | 44.2 | 12.0 | 93.5 | 13.1 | 13.8 | 15.8 | 49.6 | 39.9 | 17.2 | 24.6 | 31.0 | 15.0 | 29.0 |
| MonoScene | RGB | NYUv2 | 42.5 | 8.9 | 93.5 | 12.1 | 12.6 | 13.7 | 48.2 | 36.1 | 15.1 | 15.2 | 28.0 | 12.9 | 26.9 |
| 3DSketch | RGB + TSDF | NYUv2 | 38.6 | 8.5 | 90.5 | 9.9 | 5.7 | 10.6 | 42.3 | 29.2 | 13.9 | 9.4 | 23.8 | 8.2 | 22.9 |
| *CleanerS, its own `label_weight` rule* | *RGB + depth* | *NYUv2* | *75.0* | 46.3 | 93.9 | 43.2 | 33.7 | 38.5 | 62.2 | 54.8 | 33.8 | 39.2 | 45.7 | 33.8 | *47.7* |

**Under ISO's scoring, CleanerS is behind RGB-only ISO on NYUv2**: SC 14.7 vs 47.1 (-32.4), mIoU 16.0 vs 31.2 (-15.2). Depth input does not, by itself, win: CleanerS's published-style numbers (italic row) come from a scored set that leaves out the free space the camera saw through, where it predicts objects almost everywhere; ISO's rule counts that space.

**Why:** CleanerS's training loss uses the same `label_weight` mask as its metric (`train_utils.py:83-85`, see `EVALUATION.md`), so nothing ever penalised it for what it predicts in free space. On NYU test it predicts an object in ~97% of the free space the camera saw far from any surface (GT: 94% empty), against 7% of occluded space (measured on 200 test frames). ISO was trained on every non-255 voxel.

**CleanerS + free space from depth** sets every voxel the depth frame saw through (`tsdf > 0`) to empty — a post-process using only the model's own measured input, not part of CleanerS. With it: SC 17.3 vs ISO 47.1 (-29.8), mIoU 16.9 vs 31.2 (-14.3). Even patched, it does not overtake RGB-only ISO on both numbers; the space the camera never observed (`tsdf == 0`, ~46% of the NYU grid, still filled ~79%) is left as the model predicted.

Caveat: the ISO rows are the paper's numbers. Its NYU ground truth comes from the same SSCNet annotation downsampled to 60×36×60, but the two label files were not compared voxel for voxel here.

### 2. Occ-ScanNet val — same scoring, different training data

| method | input | trained on | SC IoU | ceiling | floor | wall | window | chair | bed | sofa | table | tvs | furn | objs | mIoU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CleanerS** (19764 frames) | RGB + depth | NYUv2 only (zero-shot) | **6.8** | 0.5 | 7.7 | 3.0 | 2.2 | 3.8 | 11.9 | 5.9 | 7.6 | 3.1 | 2.6 | 0.6 | **4.4** |
| **CleanerS + free space from depth** | RGB + depth | NYUv2 only (zero-shot) | **4.8** | 0.3 | 5.5 | 2.1 | 1.6 | 2.4 | 10.9 | 5.6 | 5.6 | 1.9 | 2.1 | 0.5 | **3.5** |
| ISO | RGB | Occ-ScanNet train | 42.2 | 19.9 | 41.9 | 22.4 | 17.0 | 29.1 | 42.4 | 42.0 | 29.6 | 10.6 | 36.4 | 24.6 | 28.7 |
| MonoScene | RGB | Occ-ScanNet train | 41.6 | 15.2 | 44.7 | 22.4 | 12.6 | 26.1 | 27.0 | 35.9 | 28.3 | 6.6 | 32.2 | 19.8 | 24.6 |

With free space from depth: -37.3 SC IoU and -25.2 mIoU against ISO.

CleanerS as-is: -35.4 SC IoU and -24.3 mIoU against ISO. Neither row isolates the input: ISO trained on Occ-ScanNet's 45,755 training frames and CleanerS never saw ScanNet. It measures whether a depth-using model trained elsewhere transfers better than an RGB model trained in-domain.

A clean RGB-D vs RGB answer on Occ-ScanNet would need CleanerS fine-tuned on Occ-ScanNet train, or ISO evaluated zero-shot from NYU.

## Per class — all-voxels / `target`

| class | GT voxels | precision | recall | IoU |
| --- | ---: | ---: | ---: | ---: |
| `empty` *(not in mIoU)* | 966,335,215 | 96.2 | 33.4 | 32.9 |
| `ceiling` | 225,878 | 0.5 | 74.5 | 0.5 |
| `floor` | 13,692,744 | 10.3 | 23.2 | 7.7 |
| `wall` | 12,401,257 | 3.1 | 60.7 | 3.0 |
| `window` | 1,015,004 | 2.4 | 19.6 | 2.2 |
| `chair` | 6,554,227 | 4.2 | 28.1 | 3.8 |
| `bed` | 3,009,885 | 14.8 | 37.7 | 11.9 |
| `sofa` | 4,064,625 | 6.2 | 56.6 | 5.9 |
| `table` | 7,033,518 | 8.4 | 41.8 | 7.6 |
| `tvs` | 79,431 | 3.4 | 25.8 | 3.1 |
| `furn` | 9,170,381 | 2.7 | 42.2 | 2.6 |
| `objs` | 2,957,403 | 0.7 | 23.6 | 0.6 |

### Biggest confusions

| GT | predicted as | voxels | share of that GT class |
| --- | --- | ---: | ---: |
| `empty` | `wall` | 233,623,210 | 24.2% |
| `empty` | `furn` | 134,210,780 | 13.9% |
| `empty` | `objs` | 101,210,358 | 10.5% |
| `empty` | `chair` | 40,583,100 | 4.2% |
| `empty` | `ceiling` | 32,676,143 | 3.4% |
| `empty` | `sofa` | 31,789,029 | 3.3% |
| `empty` | `table` | 28,317,522 | 2.9% |
| `empty` | `floor` | 27,290,006 | 2.8% |
| `empty` | `window` | 7,563,799 | 0.8% |
| `empty` | `bed` | 6,109,169 | 0.6% |

### Which classes invent volume

Each class's false positives, split into voxels the GT calls `empty` (volume grown into free or unobserved space) and voxels of another class (right volume, wrong name).

| predicted class | false positives | on GT `empty` | on another class | largest other class |
| --- | ---: | ---: | ---: | --- |
| `ceiling` | 32,807,628 | 100% | 0% | `floor` (0%) |
| `floor` | 27,779,757 | 98% | 2% | `furn` (0%) |
| `wall` | 237,406,229 | 98% | 2% | `furn` (0%) |
| `window` | 8,017,841 | 94% | 6% | `wall` (5%) |
| `chair` | 41,752,525 | 97% | 3% | `floor` (2%) |
| `bed` | 6,557,828 | 93% | 7% | `floor` (3%) |
| `sofa` | 34,728,687 | 92% | 8% | `floor` (4%) |
| `table` | 31,918,047 | 89% | 11% | `floor` (5%) |
| `tvs` | 588,725 | 95% | 5% | `furn` (1%) |
| `furn` | 139,670,909 | 96% | 4% | `floor` (1%) |
| `objs` | 106,391,210 | 95% | 5% | `wall` (2%) |

## Per class — cleaners / `target_fov`

| class | GT voxels | precision | recall | IoU |
| --- | ---: | ---: | ---: | ---: |
| `empty` *(not in mIoU)* | 92,500,805 | 85.9 | 83.3 | 73.3 |
| `ceiling` | 225,878 | 48.8 | 74.5 | 41.8 |
| `floor` | 13,692,744 | 26.0 | 23.2 | 14.0 |
| `wall` | 12,401,257 | 62.2 | 60.7 | 44.3 |
| `window` | 1,015,004 | 28.4 | 19.6 | 13.1 |
| `chair` | 6,554,227 | 59.9 | 28.1 | 23.7 |
| `bed` | 3,009,885 | 40.1 | 37.7 | 24.1 |
| `sofa` | 4,064,625 | 37.9 | 56.6 | 29.4 |
| `table` | 7,033,518 | 43.0 | 41.8 | 26.9 |
| `tvs` | 79,431 | 41.7 | 25.8 | 19.0 |
| `furn` | 9,170,381 | 31.4 | 42.2 | 22.0 |
| `objs` | 2,957,403 | 10.9 | 23.6 | 8.1 |

### Biggest confusions

| GT | predicted as | voxels | share of that GT class |
| --- | --- | ---: | ---: |
| `empty` | `floor` | 8,575,814 | 9.3% |
| `floor` | `empty` | 3,444,837 | 25.2% |
| `empty` | `furn` | 2,992,328 | 3.2% |
| `furn` | `empty` | 2,455,158 | 26.8% |
| `wall` | `objs` | 1,702,412 | 13.7% |
| `floor` | `table` | 1,626,433 | 11.9% |
| `chair` | `empty` | 1,609,440 | 24.6% |
| `wall` | `empty` | 1,572,348 | 12.7% |
| `floor` | `furn` | 1,562,653 | 11.4% |
| `table` | `empty` | 1,481,493 | 21.1% |

### Which classes invent volume

Each class's false positives, split into voxels the GT calls `empty` (volume grown into free or unobserved space) and voxels of another class (right volume, wrong name).

| predicted class | false positives | on GT `empty` | on another class | largest other class |
| --- | ---: | ---: | ---: | --- |
| `ceiling` | 176,442 | 25% | 75% | `floor` (36%) |
| `floor` | 9,065,565 | 95% | 5% | `furn` (1%) |
| `wall` | 4,584,464 | 17% | 83% | `furn` (25%) |
| `window` | 500,424 | 9% | 91% | `wall` (73%) |
| `chair` | 1,232,644 | 5% | 95% | `floor` (53%) |
| `bed` | 1,697,102 | 74% | 26% | `floor` (10%) |
| `sofa` | 3,763,377 | 22% | 78% | `floor` (33%) |
| `table` | 3,898,963 | 8% | 92% | `floor` (42%) |
| `tvs` | 28,591 | 7% | 93% | `furn` (30%) |
| `furn` | 8,452,457 | 35% | 65% | `floor` (18%) |
| `objs` | 5,707,180 | 9% | 91% | `wall` (30%) |

## Where the misses are — all-voxels / `target`

Every GT object voxel, by what the depth frame knew about it: a depth pixel landed in it, the camera saw through it (tsdf > 0, so the sensor says free space), or neither (occluded or out of view).

| region | GT object voxels | share | predicted occupied (any class) |
| --- | ---: | ---: | ---: |
| surface (a depth pixel landed) | 20,289,236 | 33.7% | 100.0% |
| seen through (tsdf > 0) | 8,962,246 | 14.9% | 99.7% |
| hidden (neither) | 30,952,871 | 51.4% | 59.3% |

Per-class recall (right class) by region:

| class | surface | seen through | hidden |
| --- | ---: | ---: | ---: |
| `ceiling` | 79.3 | 70.9 | 69.8 |
| `floor` | 36.4 | 10.8 | 17.1 |
| `wall` | 70.9 | 63.8 | 46.3 |
| `window` | 25.9 | 19.0 | 14.4 |
| `chair` | 41.1 | 43.7 | 16.2 |
| `bed` | 33.1 | 22.7 | 42.0 |
| `sofa` | 66.2 | 58.8 | 53.1 |
| `table` | 60.4 | 56.5 | 26.9 |
| `tvs` | 32.0 | 24.3 | 20.2 |
| `furn` | 54.6 | 51.5 | 35.7 |
| `objs` | 31.3 | 35.2 | 13.4 |

GT object voxels in the *seen through* row contradict the depth sensor: it measured free space there. They are 14.9% of all GT object voxels — a direct measure of GT/sensor disagreement (CAD completion or pose error).

## Does the grid's heading matter?

Both grids follow the room / scan axes rather than the camera, so the camera can look along an axis (0°) or diagonally across the volume (45°). NYU training frames span the same range (median 23° off-axis), so a drop with heading would point at the model, not at a dataset difference. Scored on all-voxels / `target`, confusion matrices pooled per bin.

| heading off-axis | frames | SC IoU | SSC mIoU |
| --- | ---: | ---: | ---: |
| 0 – 7.5° | 3372 | 6.8 | 4.8 |
| 7.5 – 15° | 3212 | 6.6 | 4.4 |
| 15 – 22.5° | 3359 | 6.8 | 4.4 |
| 22.5 – 30° | 3228 | 6.7 | 4.5 |
| 30 – 37.5° | 3293 | 6.9 | 4.4 |
| 37.5 – 45° | 3287 | 6.8 | 4.2 |

Within-scene check (removes scene difficulty): for the **591 scenes** with at least 3 frames both within 10° of an axis and within 10° of the diagonal, aligned minus diagonal is a median **+0.0 SC IoU** (aligned better in 50% of scenes) and **+0.1 SSC mIoU** (52%).

### Camera pitch

Negative = looking down. NYU's median is -13.5°.

| pitch | frames | SC IoU | SSC mIoU |
| --- | ---: | ---: | ---: |
| -90 – -40° | 1701 | 4.1 | 3.2 |
| -40 – -25° | 9013 | 6.4 | 4.3 |
| -25 – -10° | 7085 | 7.8 | 4.7 |
| -10 – 10° | 1833 | 7.0 | 3.8 |
| 10 – 90° | 132 | 4.6 | 1.9 |

## Spread across frames

Per-frame SC IoU (all-voxels / `target`): median 6.6, 10th percentile 3.7, 90th percentile 10.5. Per-frame numbers are in `per_frame.csv`.

