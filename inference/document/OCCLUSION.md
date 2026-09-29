# What the model does behind things it cannot see

**Short version:** CleanerS reconstructs *large furniture* it cannot see far
better than it reconstructs *walls* it cannot see. On NYU's held-out test set,
occluded wall recall is **25.8%** against 79.5% for visible wall, while occluded
**bed** recall is **84.0%** — *higher* than its own visible recall. So on any
frame with a bed against a wall, the bed is confidently extruded into the space
behind it and the wall behind it dissolves. That is not a bug in this repo, it
is not the camera, and it is not the TSDF encoder. It is what the pretrained
checkpoint does on its own benchmark, and it is already inside the published
numbers — which this sweep reproduces exactly, to 0.1 IoU on every class.

Written 2026-09-04, chasing "why does the bed go through the wall in room03?".

- [The capture that raised it](#the-capture-that-raised-it)
- [Where the wall goes](#where-the-wall-goes)
- [The NYU sweep](#the-nyu-sweep)
- [Validation: we reproduce Table 1 exactly](#validation-we-reproduce-table-1-exactly)
- [Per-class IoU on the benchmark's own occlusion mask](#per-class-iou-on-the-benchmarks-own-occlusion-mask)
- [Recall by visibility](#recall-by-visibility)
- [Does the paper report this?](#does-the-paper-report-this)
- [The second artefact: our grid, not the model](#the-second-artefact-our-grid-not-the-model)
- [What this means](#what-this-means)
- [Reproducing](#reproducing)
- [Open items](#open-items)

---

## The capture that raised it

`captures/room03` — a bed with its headboard against the far wall, camera at
0.74 m, shot with `--preview`.

The capture is clean, which is what makes it a useful test case:

| | |
| --- | --- |
| valid depth | **98.2%** of pixels |
| depth range | 1.10–3.27 m, median 2.09 |
| world extent | X −1.59…1.26, Y 1.10…3.27 fwd, Z 0.02…2.28 up |
| surface points inside the grid | **100.0%** — nothing clipped on any face |

No min-Z dropout ([GEOMETRY.md §1](GEOMETRY.md)), nothing lost off the box.
The encoder was handed good geometry.

The back wall, fitted from the blank region above the bed (44174 pixels,
**rms 1.6 cm**): a plane **2.35 m** from the camera with normal
`(0.577, 0.802, 0.154)` — i.e. about **36° in yaw**, so the camera is not
square-on to it. Note `yaw` in `meta.json` was 0.

The rendered `.ply` holds 7797 voxels: bed 3533 (45.3%), floor 1809 (23.2%),
wall 1737 (22.3%), objs 656 (8.4%).

## Where the wall goes

Rasterising that fitted plane into the voxel grid and reading off what the
network predicted **in the cells the wall actually passes through**, split at
the mattress line:

| cells the wall passes through | wall | bed | empty |
| --- | ---: | ---: | ---: |
| **above 0.75 m** — wall directly visible above the bed | **86.4%** | 0.3% | 10.4% |
| **below 0.75 m** — wall hidden behind the bed | **29.4%** | 30.8% | 39.4% |

The wall is near-perfect where the camera can see it and collapses the moment
the bed occludes it. Below the mattress line the volume is either left empty or
relabelled *bed*, so the bed body simply continues through the plane. You can
see the cutoff directly in a slice along the plane (`#` wall, `B` bed, `.`
empty; rows are height, columns are X):

```
1.11  .......................##.##########################o.B...SS
1.03  ...........................#####B##################oo.o...SS
0.95  ...........................o#oo#o#o################ooooo..SS
0.87  ................................oooo#o##############..oo..SS
0.79  ...................................ooB##############ooo..SSS
0.71  .........................................###########..B..SSS
0.63  ..............................BBB.........##########.SS..SSS
0.55  .............................BBBBBBBBBB...##########SSSS.SSS
0.47  .............................BBBBBBBBBBB..######W###SSSSSSSS
```

Consistent with that, measuring every shown voxel against the observed depth
along its own camera ray:

| class | on the seen surface | behind it | median depth behind |
| --- | ---: | ---: | ---: |
| wall | **70.2%** | 22.1% | 0.63 m |
| bed | 15.2% | **81.0%** | 0.93 m (max 1.91) |
| floor | 0.1% | **99.2%** | 1.97 m (max 3.44) |

Wall is the only class predicted mostly on real observed geometry — it is a
one-voxel shell, which is why it reads as "not predicted at all" when the bed
blob sits in front of it.

## The NYU sweep

The question this raises is whether the model does this everywhere, or whether
we broke something. So: run **all 1449 NYU frames** (654 test + 795 train)
through the *same* pipeline and score visible and occluded voxels separately.

The sweep computes two independent splits of every voxel.

**1. Ours, from the TSDF** — what the depth camera could actually see:

```
visible  = weight > 0 and |tsdf| < 0.25    the surface shell, directly seen
occluded = tsdf < -0.5                     behind the surface, must be completed
```

**2. The benchmark's own**, exactly as `test_NYU.py:206-211` builds it:

```
SSC  = label_weight & (label3d != 255)                          surface + occluded
SC   = label_weight & (mapping == 307200) & (label3d != 255)    occluded only
surf = label_weight & (mapping != 307200) & (label3d != 255)    the complement
```

`mapping == 307200` is `MAPPING_SENTINEL` — the voxel projects to no image
pixel, i.e. it is not on the visible surface. **100.0%** of the voxels our TSDF
calls `occluded` carry that sentinel, so the two definitions agree. The `surf`
mask is ours; the benchmark never scores it, and it is what turns the SC column
from a bare number into a measurable *drop*.

**This is our pipeline, not the reference one.** The model input is
`Custom_TSDF`, verified bit-identical to what
`FrameLoader.build_tsdf_and_mapping()` produces (`np.array_equal` → `True`, max
diff 0; same for `Custom_Mapping`), and `--verify_encoder` re-checks that on one
frame before starting. The eval masks use the *reference* `label_weight`
(`TSDF/arr_1`) and `Mapping`, matching `test_NYU.py` — the isolated-encoder
setup described in `cleaner/dataset/NYU/NYU.py`.

**All headline numbers below are test-split only (654 frames).** The train half
was seen during training and scores about 9 IoU higher (wall 52.4 vs the
published 43.2), so mixing it in flatters every number. Train figures appear
only where labelled.

## Validation: we reproduce Table 1 exactly

Before trusting any new number, the pipeline has to produce the old ones.
Scene completion, official mask, test split:

| | prec. | recall | IoU |
| --- | ---: | ---: | ---: |
| ours | 88.0% | 83.5% | **75.0%** |
| published (CleanerS\*, Table 1) | 88.0% | 83.5% | **75.0%** |

And per-class SSC IoU against the published row:

| | ceil. | floor | wall | win. | chair | bed | sofa | table | TVs | furn. | objs. | avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| published | 46.3 | 93.9 | 43.2 | 33.7 | 38.5 | 62.2 | 54.8 | 33.7 | 39.2 | 45.7 | 33.8 | 47.7 |
| ours | 46.3 | 93.9 | 43.2 | 33.7 | 38.5 | 62.2 | 54.8 | 33.8 | 39.2 | 45.7 | 33.8 | 47.7 |

Every class matches; `table` differs by 0.1 on rounding. So the masks are the
benchmark's masks and the numbers below are computed on the same footing as the
paper's.

## Per-class IoU on the benchmark's own occlusion mask

This is the table the paper does not publish: its own SC mask, broken down by
class, against the visible-surface complement. Test split.

| class | surface IoU | **occluded IoU** | drop |
| --- | ---: | ---: | ---: |
| ceiling | 74.9 | **37.3** | **−37.7** |
| wall | 62.8 | **36.1** | **−26.7** |
| chair | 58.2 | **32.9** | **−25.3** |
| table | 48.5 | **29.7** | **−18.9** |
| window | 42.8 | 29.5 | −13.3 |
| tvs | 49.4 | 36.5 | −12.9 |
| objs | 42.6 | 30.7 | −11.9 |
| furn | 55.2 | 44.0 | −11.2 |
| sofa | 63.8 | 53.3 | −10.6 |
| **bed** | 67.1 | **61.7** | **−5.4** |
| floor | 96.8 | 92.9 | −4.0 |
| **avg** | **60.2** | **44.0** | **−16.1** |

Bed and floor are the two most occlusion-robust classes in the benchmark. Wall
loses five times more IoU than bed does. Ceiling — the other large planar
surface — is worst of all.

## Recall by visibility

Same effect in recall terms, using our TSDF split. Test split:

| class | visible | occluded | drop |
| --- | ---: | ---: | ---: |
| ceiling | 84.5% | 14.6% | **−69.9** |
| chair | 73.3% | 11.5% | **−61.8** |
| table | 69.1% | 13.9% | **−55.3** |
| **wall** | **79.5%** | **25.8%** | **−53.7** |
| window | 68.3% | 21.3% | −46.9 |
| tvs | 64.3% | 29.1% | −35.2 |
| objs | 56.3% | 27.3% | −29.1 |
| furn | 68.9% | 50.6% | −18.4 |
| sofa | 77.3% | 71.1% | −6.1 |
| floor | 98.4% | 98.8% | +0.4 |
| **bed** | **77.3%** | **84.0%** | **+6.7** |

<details>
<summary>All 1449 frames (optimistic — includes memorised training frames)</summary>

| class | visible | occluded | drop |
| --- | ---: | ---: | ---: |
| ceiling | 91.6% | 29.6% | −62.0 |
| chair | 81.1% | 16.0% | −65.0 |
| table | 79.9% | 30.0% | −49.9 |
| wall | 83.1% | 35.0% | −48.2 |
| window | 81.0% | 39.7% | −41.3 |
| tvs | 77.3% | 58.6% | −18.7 |
| objs | 69.2% | 52.0% | −17.2 |
| furn | 82.0% | 70.9% | −11.1 |
| sofa | 84.9% | 80.2% | −4.6 |
| floor | 98.7% | 98.9% | +0.2 |
| bed | 84.6% | 89.3% | +4.7 |

</details>

### Where occluded wall goes instead

Test split, 233446 occluded GT-wall voxels:

| becomes | share |
| --- | ---: |
| **empty** | **59.0%** |
| wall (correct) | 25.8% |
| furn | 9.3% |
| window | 1.2% |
| floor | 1.2% |
| bed | 1.1% |

Nearly **three in five** occluded wall voxels are predicted as *nothing at all*.

### Per frame

Restricted to frames with more than 200 GT-wall voxels on each side:

| | test (320 frames) | all (693 frames) |
| --- | ---: | ---: |
| median visible wall recall | 84.4% | 87.7% |
| median occluded wall recall | **29.5%** | 39.8% |
| frames below 50% occluded recall | 237 (**74%**) | 426 (61%) |
| frames below 30% | 163 (**51%**) | 265 (38%) |

Half of all held-out frames recover under 30% of the wall they cannot see.
room03 is not an outlier; it is the median case.

### The asymmetry is the mechanism

Bed is the only class that gets *better* under occlusion, with sofa and floor
close behind. These are not two independent errors, they are one shape:

> **large volumetric furniture wins the occluded space; thin planar structure
> loses it.**

Which is exactly the room03 slice — `#` above the mattress line, `B` below it.

## Does the paper report this?

Partly, and not in a form that shows the problem.

| metric | region scored | per-class? |
| --- | --- | --- |
| **SC** (prec / recall / IoU) | occluded voxels only | **no** — occupied vs empty |
| **SSC mIoU** (the 12-class table) | surface **and** occluded, mixed | yes, but not split |

So the benchmark *does* have an occlusion metric — SC — inherited from SSCNet,
and CleanerS's headline claim of **+3.1% IoU** on scene completion is
substantially an occluded-voxel gain. But SC collapses every class into
"occupied", and the two classes at issue move in opposite directions
(−26.7 vs −5.4 IoU). Averaging a vanishing wall against a confidently
over-extruded bed can even *improve* the binary number, since the bed's extra
voxels partly fill the space the wall vacated.

The per-class table, meanwhile, mixes the surface and occluded halves. Its
wall IoU of 43.2 is a blend of 62.8 and 36.1.

The paper is *conceptually* about this — its abstract frames SSC as having to
"imagine what is behind the visible surface" and motivates the cleaner teacher
as focusing "more on the 'imagination' of unseen voxels". It just never reports
a number that separates *which* things get imagined well.

## The second artefact: our grid, not the model

room03's render also has a floor slab running up to **2.71 m past the back
wall** — 1008 voxels, 56% of all floor voxels shown, and 16.6% of the whole
`.ply` sits outside the room.

That one is ours. `FrameLoader.from_live_camera` hardcodes
`vox_origin = (-2.4, 0, -0.05)` and a 4.8 m-deep box anchored at the camera
plane. The room ends at 2.35 m, so **half the box is outside the room**, and
`--mask surface` (`occluded | surface`, where `occluded = tsdf < -0.5`) has no
concept of a room boundary — it renders everything behind the observed surface
out to the grid edge. NYU's own mask (`label_weight`) gets that boundary from
ground truth, which marks out-of-room voxels 255; we have no GT, so we cannot.

The sweep separates this cleanly. Across **18 722 958** test-split voxels in
`occluded & GT == 255` (NYU's out-of-room marker):

| predicted | share |
| --- | ---: |
| **empty** | **95.8%** |
| floor | 3.8% |
| wall | 0.1% |
| furn | 0.1% |

So the model does have a mild "keep laying floor" habit, but on NYU it has
almost nowhere to express it — the `.bin` header fits the box to each room. In
room03 that same 3.8% tendency gets roughly ten times the volume to fill.

## What this means

Two distinct problems, and it is worth not conflating them:

| symptom | cause | fixable here? |
| --- | --- | --- |
| bed extruded through the wall behind it | **the pretrained model** — 36.1 occluded wall IoU vs 61.7 for bed, reproduces on NYU | no |
| floor slab 2.7 m past the wall | **our grid placement** amplifying a 3.8% model bias | yes |

The first is the honest limit of single-view SSC on planar structure behind
furniture. It is *already inside* the published numbers — we reproduce Table 1
to 0.1 IoU, so nothing has regressed; we are simply looking at a failure the
aggregate mIoU averages away. Fixing it needs multiple views, or accepting that
occluded planar structure is unreliable and not treating it as a measurement.
(See [MAKING_GT.md](../../reconstruction_GT/MAKING_GT.md) on why SSC output is never a reference.)

The second is ours and is display-only. Options, none implemented yet:

1. **Clip the render at the observed back surface** — drop voxels more than
   ~0.2 m behind the furthest observed depth along their ray. Kills the floor
   slab, and incidentally hides most of the bed's overshoot, since both live in
   the same unbounded region.
2. **Fit `vox_origin`/`yaw` per capture** from the depth so the box covers the
   actual room, the way NYU's `.bin` header does per scene. Note room03's wall
   was at 36° yaw with `yaw = 0` recorded, so the box's far corner reaches
   deeper past the wall than its centre does.
3. Shoot from further back so the room fills more of the 4.8 m.

Option 1 is display-only and safe. Option 2 changes a model *input* and would
need care — a fitted `vox_origin` is an estimate, and estimates do not belong
in `meta.json` alongside measured values.

## Reproducing

```bash
python -m inference.nyu_occlusion_sweep --verify_encoder --out /tmp/nyu_sweep.npz
```

9.2 minutes for 1449 frames on a 3050 Ti (4 GB), at 2.6 frames/s. Prints every
table above — test split first, then test+train — and saves the raw confusion
matrices so the same run can be re-sliced without re-running:

| key (prefixed `test_` / `train_`) | shape | over which mask |
| --- | --- | --- |
| `vis` / `occ` | (12,12) | our TSDF split, GT × pred |
| `surf` / `sc` / `ssc` | (12,12) | the benchmark's masks, GT × pred |
| `scbin` | (2,2) | occupied-vs-empty on the SC mask — the published number |
| `p255` / `n255` | (12,) / scalar | predictions in `occluded & GT == 255` |
| `per` | (N,5) | per frame: `id, wall_vis_n, wall_vis_hit, wall_occ_n, wall_occ_hit` |
| `keys` | (N,) | frame ids, sorted |

To re-print from a saved npz without re-running the model:

```python
from inference.nyu_occlusion_sweep import report
report('/tmp/nyu_sweep.npz', split=('test',))
```

Pass `--tsdf_dir TSDF --mapping_dir Mapping` to run the same measurement on the
reference encoder instead of ours — useful if the encoder is ever suspected,
though it was cleared here.

## Open items

- Nothing from this has been implemented. The three fixes above are all
  unstarted.
- The sweep scores whole classes. It does not isolate "wall occluded
  **by a bed**" from "wall occluded by anything", which is the specific case
  room03 hits. Splitting occluded GT-wall by the class of the occluder in front
  of it along the ray would sharpen the 36.1 number considerably.
- Worth checking whether the **teacher** checkpoint (`Teacher_ckpt.pth`, clean
  TSDF-CAD input) shows the same asymmetry. If it does, the cause is the label
  distribution rather than depth noise, and no amount of better capture will
  help.
- `captures/room02/depth/live_000000.png` was overwritten at 2026-09-04 20:19
  with a 640×526 8-bit RGB image (looks like a `colorize_depth` output written
  over its own input). The real depth survives in `room02/native/`; the 640×480
  frame needs re-deriving via `match_nyu_fov` before room02 can be re-run.
