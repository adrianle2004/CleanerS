# What CleanerS is good at, and how to shoot for it

Measured on the two hand-annotated rooms in `captures/` (10 frames, ground
truth made by `reconstruction_GT/`), with NYU's 654 test frames and 19,764
Occ-ScanNet frames as the reference distributions.

This is the practical document: not "what does the model score" but "when
should you believe its output, and how do you hold the camera". Every number
here is reproducible with `reconstruction_GT/evaluate_gt.py` and the CSVs in
`outputs/`.

---

## Per class, on real captures

Pooled over the 10 annotated frames, NYU protocol, camera-view only:

| class | IoU | precision | recall | mostly confused with |
| --- | ---: | ---: | ---: | --- |
| `ceiling` | 61.6 | 67.3% | 87.9% | `wall` |
| `chair` | 58.9 | **96.2%** | 60.3% | `empty` |
| `wall` | 57.2 | 86.0% | 63.1% | `empty` (1694 voxels) |
| `window` | 49.8 | 82.5% | 55.7% | `empty` |
| `bed` | 49.7 | 82.0% | 55.8% | `empty` (3242) |
| `table` | 29.3 | 35.3% | 63.1% | `bed` |
| `floor` | 22.1 | **22.1%** | 100.0% | — |
| `objs` | 20.2 | 60.8% | 23.2% | `bed` (842) |
| `furn` | 10.4 | 15.8% | 23.5% | `empty` |
| `tvs` | 0.0 | — | 0.0% | `furn` |
| `sofa` | 0.0 | 0.0% | — | 1080 voxels predicted in rooms with no sofa |

### The shape of it: high precision, low recall

The structural classes all sit near 80–96% precision and 55–63% recall. **When
the model says something is there, it is usually right; when it says a voxel is
empty, it is often wrong.** The single largest error mode is real structure
coming back as `empty` — 1694 wall voxels, 3242 bed, 2470 objs.

Treat a CleanerS occupancy map as a **lower bound**. For anything that depends
on free space being genuinely free — navigation, clearance, grasp planning —
dilate the prediction, or treat `empty` as `unknown` rather than as `free`.

### Two classes to distrust outright

- **`floor` at 22.1% precision with 100% recall.** It finds every floor voxel
  and claims roughly three times too many: it paints floor into space. Useful
  as "the floor is somewhere in here", useless as a floor boundary.
- **The furniture taxonomy — `furn`, `tvs`, `sofa`.** It predicted 1080 `sofa`
  voxels in two rooms containing no sofa, missed all 258 `tvs` voxels (calling
  them `furn`), and `furn` itself scores 10.4. The fine-grained furniture
  distinction does not survive outside NYU's furniture. If a downstream task
  needs "wardrobe or sofa", this model will not supply it.

---

## How to hold the camera

The model was trained on NYU, so NYU's shot geometry is its home ground. Our
sweeps, NYU and Occ-ScanNet measured identically:

| | camera height | median depth | room hidden behind the surface |
| --- | ---: | ---: | ---: |
| NYU | 1.34 m | **2.49 m** | **1.72 m** |
| Occ-ScanNet | 1.43 m | 1.77 m | 1.64 m |
| room08 | 1.06 m | 1.67 m | 0.36 m |
| room07 | 1.00 m | 1.85 m | 0.40 m |

**Distance is the variable that matters, not height.** Aim for a median depth
around 2.5 m. Our frames sit at 1.67–1.85 m, already near the close edge of
what the model saw in training, and the single closest frame (0.9 m median)
is the one where the bed came back entirely as `furn`.

### The threshold: about 1 m of hidden extent

The relationship is not linear, it **saturates**, which is why the same
measurement looks decisive on our rooms and irrelevant on the benchmarks.
Pooled over all 2,523 frames of the four datasets, occupancy of the SC set
against hidden extent:

```
below 1 m of hidden extent:  corr -0.60   (699 frames)
above 1 m:                  corr +0.01   (1,824 frames)
```

One curve, and each dataset sits on a different part of it — median occupancy
of the SC set per band:

| hidden extent | NYU | Occ-ScanNet | room08 | room07 |
| --- | ---: | ---: | ---: | ---: |
| 0.00–0.15 m | — | — | 99% | 100% |
| 0.15–0.35 m | — | 79% | 94% | 100% |
| 0.35–0.60 m | 63% | 66% | 90% | 99% |
| 0.60–1.00 m | 53% | 57% | 86% | — |
| 1.00–1.60 m | 49% | 45% | — | — |
| 1.60–2.30 m | 57% | 41% | — | — |
| 2.30–9.00 m | 65% | 40% | — | — |

Below roughly **1 m of hidden extent the occupancy climbs toward 100% and SC
stops measuring anything**; above it the curve flattens and more hidden extent
buys nothing. NYU spans 0.73–3.18 m and Occ-ScanNet 0.63–3.41 m — both live
entirely above the knee, which is why hidden extent correlates +0.08 with SC
there. room07 spans 0.12–0.55 m and room08 0.03–0.84 m — entirely below it,
hence −0.57 and −0.73. The correlation is meaningful in both places; it is the
same relationship seen from either side of its elbow.

### Hidden extent is not hidden *space*

The quantity above measures how far a ray travels past the first surface before
it leaves the room. It counts the inside of the wardrobe exactly as it counts
the air behind it. What fraction of it is genuinely empty is the occupancy
figure, and the product of the two is the thing SC can actually test:

| | hidden extent | of it solid | **empty hidden depth** | SC |
| --- | ---: | ---: | ---: | ---: |
| NYU | 1.72 m | 55% | **0.666 m** | 78.9 |
| Occ-ScanNet | 1.64 m | 45% | **0.853 m** | 54.6 |
| room08 | 0.36 m | 87% | **0.041 m** | 67.1 |
| room07 | 0.40 m | 100% | **0.002 m** | 90.2 |

Two deficits multiply in a small room: the hidden extent is 4–5× smaller, and
what little exists is almost all object interior. room07 has **2 mm** of empty
hidden depth per ray against NYU's 666 mm — a factor of 383. The model is being
asked "what is behind this surface?" where the answer is always "more of the
same object", which is why a constant "solid" beats it.

**So the capture target is not "2.5 m of depth" for its own sake — it is
whatever clears about 1 m of hidden extent.** 2.5 m of depth is simply how NYU
gets there. room08 topped out at 0.89 m and never cleared it, which is why most
of its viewpoints cannot carry an SC number — though its best ones still can.

(The two benchmarks diverge above the knee: NYU's occupancy curls back up,
49% → 65%, while Occ-ScanNet's keeps falling, 45% → 40%. That is its sparse
ground truth again — more hidden volume there means more unlabelled emptiness
rather than more furniture.)

Height is a trap in a small room. Raising the camera from 0.6 m to 1.3 m in
room08 took the hidden volume from 0.81 m to 0.07 m and SC from 48.7 to 89.9 —
which looks like a better prediction and is not one (see below). Keep the
camera near the height of what you are looking at, stand back, and let
furniture occlude furniture.

Give it **complete RGB**. The colour frame is the input CleanerS treats as
reliable — its whole method distils a clean-*depth* teacher into a noisy-depth
student, so colour is the constant. Measured cost of damaged colour: ±1–5 SSC
points frame to frame, about 1 point pooled. `reconstruction_GT/export_frame.py`
keeps colour whole by aligning depth into the colour camera, which is also what
NYU does.

---

## Which number to quote

**SSC, not SC**, from your own captures.

SC scores only the voxels hidden behind the visible surface. In a small room
the camera sees nearly all the free space, so what remains hidden is the inside
of the bed and the inside of the walls — all solid. Measured: 98% of room07's
hidden voxels are solid, against 55% on NYU. A model that answers "solid" every
time scores 0.98 there, so SC cannot separate a good model from a trivial one.

The symptom to check on any capture is the **margin** — SC minus what "occupied
everywhere" would score:

| | SC | trivial baseline | margin |
| --- | ---: | ---: | ---: |
| NYU, least hidden extent | 80.0 | 54.1 | **+23.2** |
| NYU, most hidden extent | 82.7 | 65.3 | **+12.6** |
| room08, least hidden extent | 92.7 | 99.4 | **−2.2** |
| room08, most hidden extent | 48.8 | 85.5 | **−35.0** |

NYU is positive everywhere: the model beats the baseline whatever the shot.
Those room08 rows are **band medians, and they hide the frames that work** —
32 of its 210 swept frames do clear the baseline, six by more than +20, which
is inside NYU's own range. `live_000180` scores SC 81.9 against a 57% baseline,
a margin of +25.1. So room08 measures completion at a minority of viewpoints;
read SC frame by frame against its own baseline, never pooled over a mixed
set. room07 is the room that cannot: 0 of 159 frames clear it, best +0.0. Meanwhile the **median SSC of those same bands stays between 32.6 and 40.7**
while SC swings 48.8 to 92.7 — the model's semantic output barely notices the
viewpoint; only SC's baseline does. (Individual frames spread wider, 17.9 to
65.1; it is the band-to-band medians that are flat.)

So: `evaluate_gt.py` prints the occupancy of the SC set and warns when it
exceeds 90%. Read that warning before quoting any SC number.

---

## Where each dataset can be trusted

| | SC (completion) | SSC (semantics) |
| --- | --- | --- |
| **your rooms** | room08 at its best viewpoints (+25 margin); room07 never | **yes — the best of the three** |
| **NYU** | yes (75.0) | yes (47.7) |
| **Occ-ScanNet** | floor-free (57.9), but it reads low | no (16.4 — the GT is too sparse) |

Occ-ScanNet's ground truth comes from reconstruction rather than annotation:
94.1% of its labelled voxels are `empty` against NYU's 88.3%, `objs` covers
0.33% of voxels against NYU's 1.55%, 14% of frames have no floor at all, and
the floors that exist are tilted a median 1.94° inside the grid. Its SSC of
16.4 is mostly that, not the model.

Its SC reads low for the same reason, and it can be shown: matched against
NYU on the *baseline* rather than on the raw score, recall is comparable
(84.1% vs 87.1% in the emptiest band) while precision is not (55.8% vs 90.2%),
and the gap vanishes where the GT happens to be dense (97.3% vs 96.3% at
80–100% occupancy). The model predicts furniture that is really there and is
charged for it. `document/EVALUATION_SCANNET.md` section 3 has the full table;
`python -m reconstruction_GT.sc_gap --all` rebuilds it.

Your rooms are the opposite: dense, hand-annotated, floor present in 100% of
frames, grids level to 0.01–0.15°. That is why the per-class table at the top
of this document is trustworthy — an error there is the model's, not a hole in
the ground truth — and it is the most useful thing these captures produce.

---

## Reproducing any of this

```bash
python -m reconstruction_GT.evaluate_gt captures/room08 --report
python -m reconstruction_GT.sweep_eval   captures/room08 --every 5
```

The per-frame CSVs behind the comparisons:

```
outputs/room07/sweep_eval.csv      outputs/room08/sweep_eval.csv
outputs/nyu_viewpoints.csv         outputs/scannet_viewpoints.csv
outputs/scannet/per_frame.csv      outputs/nyu_perframe.csv
outputs/sc_gap.csv
```

All but the two sweeps are built by one command, which also prints the
NYU-vs-Occ-ScanNet comparison at matched difficulty:

```bash
python -m reconstruction_GT.sc_gap --all
```

Both rooms' `gt/EVALUATION.md` carry the same analysis per room, with the NYU
and ScanNet columns beside their own.
