# Useful commands

Everything this fork adds on top of `fereenwong/CleanerS`, as commands. The
upstream repo had one: `test_NYU.py`. This one has a live camera, a
ground-truth pipeline for real rooms, a second dataset and the checks that
keep all of it honest, so it is worth having the list in one place.

New machine first: [SETUP.md](SETUP.md). What lives on Hugging Face instead of
git: [DATA.md](DATA.md). The two module references explain *why* each tool
exists and what its numbers mean — this page only says how to run them:
[inference/document/README.md](inference/document/README.md) and
[reconstruction_GT/document/README.md](reconstruction_GT/document/README.md).

## Contents

- [Conventions](#conventions)
- [Data in and out](#data-in-and-out)
- [Reproduce the paper](#reproduce-the-paper)
- [Live camera](#live-camera)
- [One frame at a time](#one-frame-at-a-time)
- [Looking at a result](#looking-at-a-result)
- [Making ground truth for a real room](#making-ground-truth-for-a-real-room)
- [Scoring](#scoring)
- [Checks and self-tests](#checks-and-self-tests)
- [Long jobs](#long-jobs)
- [Every command, in one table](#every-command-in-one-table)

## Conventions

```bash
conda activate CleanerS            # py3.7 env; everything below except hf_data.py
cd ~/CleanerS/CleanerS             # repo root: run from here
```

- **Run from the repo root.** `test_NYU.py` is the one exception — it is run
  from `examples/segmentation/`, because that is what its `--cfg` default
  assumes. Paths inside the cfgs resolve against the repo, not the working
  directory (`cleaner/utils/paths.py`), so nothing else cares.
- `--cfg ./cfgs/NYU/voxelSSC.yaml` is the default everywhere and can be
  dropped. `--pretrained_path` is **required** by `run_live`, `run_inference`
  and `run_scannet`, and defaults to `./checkpoint/CleanerS_ckpt.pth` in the
  evaluation tools.
- `tools/hf_data.py` is the one script that needs a **newer** python — run it
  as `python3` from a deactivated shell (any python ≥ 3.9).
- The `reconstruction_GT/*.sh` scripts find their own interpreter: an active
  `CleanerS` env, else `~/miniconda3/envs/CleanerS`, else `$CLEANERS_PY`. They
  take a bare room name (`room08` → `captures/room08`) or a path.
- Every tool takes `--help`, and every one of them has a module docstring that
  says why it exists. `python -m inference.run_live --help` is the authority on
  flags; this page is the authority on which command to reach for.

## Data in and out

Datasets, `.bag` recordings and checkpoints are too big for git and live in a
private Hugging Face dataset repo. [DATA.md](DATA.md) says what pairs with what.

```bash
python3 tools/hf_data.py pull                  # ~24 GB; only fetches what is missing
python3 tools/hf_data.py pull --skip-images     # ~9 GB: no ScanNet posed_images
python3 tools/hf_data.py push --dry-run         # what is stale?
python3 tools/hf_data.py push
python3 tools/hf_data.py push --repack NYU/Custom_TSDF    # globs ok: 'Scannet/gt_*'
```

**Run the dry-run after every commit that touched `captures/` or a dataset
folder.** Git and HF hold two halves of one dataset and the mismatch is silent:
re-fusing a scan rewrites both `trajectory.txt` (git) and `room.ply` (HF), and
a fresh `pull` then gives you an annotation that no longer sits on its mesh.

## Reproduce the paper

Upstream's own entry points, unchanged apart from path handling:

```bash
cd examples/segmentation
python test_NYU.py --cfg ../../cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ../../checkpoint/CleanerS_ckpt.pth --no_visualize
# SC 74.95, SSC mIoU 47.73   (654 frames, a few minutes)

bash run.sh                                     # training
```

## Live camera

`run_live` is the streaming runner: one process, model loaded once, a frame at
a time. `--source` takes a camera or any replay, and everything downstream is
identical either way — same alignment, same intrinsics, same depth scale.

```bash
# the camera
python -m inference.run_live \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source realsense --camera_height 1.42 --display

# no camera: a recorded sweep, with the accelerometer it was recorded with
python -m inference.run_live \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source bag:captures/room07/scan/scan.bag --display --colour all

# the accumulating room, which is what to show someone (DEMO.md)
python -m inference.run_live \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source bag:captures/room07/scan/scan.bag \
    --accumulate --display --zoom 2.2 --out_dir outputs/room07_world

# NYU or a folder of depth PNGs, as a replay
python -m inference.run_live --pretrained_path ... \
    --source nyu:./data/NYU/testset/depth --rgb_dir ./data/NYU/testset/RGB \
    --max_frames 20
```

| `--source` | |
| --- | --- |
| `realsense[:WxH@FPS]` | the camera |
| `bag:/path/scan.bag` | a recorded sweep; replays the IMU too |
| `nyu:/path` | NYU's bit-rotated depth encoding |
| `folder:/path` | plain uint16-millimetre depth PNGs |

The flags worth knowing: `--colour class|zbuffer|all`, `--save
ply|npy|both|none`, `--mask surface|occluded|frustum`, `--encoder
auto|torch|numpy|cuda`, `--amp auto|on|off`, `--accumulate` plus
`--world_colour photo|verdict|class`, `--loop`, `--max_frames`,
`--no_imu`, `--realtime`. In the window:

| key | |
| --- | --- |
| `space` / `f` | freeze / carry on |
| `m` | cubes → one surface → the sensor's own reconstruction |
| `c` | cycle `photo` → `verdict` → `class` |
| `+` `-` / `0` | zoom / fit what has been built |
| `s` | screenshot into `--out_dir` |

[DEMO.md](inference/document/DEMO.md) is the script for running this in front
of someone; [WORLD_BUILDING.md](inference/document/WORLD_BUILDING.md) is how
the camera knows where it is; [VIEWERS.md](inference/document/VIEWERS.md) is
what each colouring means.

## One frame at a time

Capture a still frame to disk, then predict on it. This is the path a scored
capture goes down, because the frame has to stay put to be annotated against.

```bash
# --preview: type the camera height in the window, SPACE saves the frame
python -m inference.capture --out_dir ./captures/room01 --filters --preview
# without --preview, measure it and pass it:
python -m inference.capture --out_dir ./captures/room01 --filters --camera_height 1.42

python -m inference.run_inference \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --live_dir ./captures/room01 --out_dir ./outputs/room01
```

`run_inference` also takes `--bin_path`/`--png_path` for a single NYU frame or
`--data_dir` for a folder of them, and skips frames that already have a
prediction unless given `--overwrite` — so an interrupted run resumes.

`capture --hole_fill` **invents geometry the sensor never saw**; what the other
capture settings do to the output, with measured numbers, is in
[CAMERA_TUNING.md](inference/document/CAMERA_TUNING.md).

## Looking at a result

Four viewers over the same `.ply`, in one family. The first three answer
different questions; the fourth puts two of them in one window.

```bash
python inference/display.py          ./outputs/room01/ply/live_000000.ply   # points
python inference/display_solid.py    ./outputs/room01/ply/live_000000.ply   # cubes + photo
python inference/reconstruct.py      ./outputs/room01/ply/live_000000.ply   # what the camera MEASURED
python inference/display_overlay.py  ./outputs/room01/ply/live_000000.ply   # both, overlaid
```

`reconstruct.py` is the only geometry in the repo that is pure measurement, so
it is the reference a completion gets checked against — never the other way
round. Useful variants:

```bash
python inference/display_solid.py <ply> --colour class        # flat label colours
python inference/display_solid.py <ply> --visibility all      # colour by the evidence reaching each voxel
python inference/display_solid.py <ply> --screenshot out.png  # offscreen
python inference/reconstruct.py   <ply> --surface label       # the model's labels on real geometry
python inference/reconstruct.py   <ply> --surface match       # filled vs left empty
python inference/display_overlay.py <ply> --show surface
```

Prediction next to ground truth, two windows framed identically:

```bash
inference/show_nyu.sh 0001                        # vs NYU's own Label, not ISO's
inference/show_nyu.sh 0001 --show voxels          # extra flags go to both windows
inference/show_scannet.sh scene0702_00            # the scene's best frame
inference/show_scannet.sh scene0702_00 worst
inference/show_scannet.sh scene0702_00 00004      # any frame; exported on demand
MASK=none inference/show_nyu.sh 0001              # the full prediction, unmasked
PRED_DIR=outputs/scannet PYTHON=python3 inference/show_scannet.sh scene0702_00
```

Depth you can actually read, the way the RealSense Viewer draws it:

```bash
python -m inference.colorize_depth captures/room01/depth --out captures/room01/depth_vis
python -m inference.colorize_depth data/NYU/depth/NYU0001_0000.png --encoding nyu
```

## Making ground truth for a real room

The runbook, with the reasoning, is
[MAKING_GT.md](reconstruction_GT/document/MAKING_GT.md). The short version —
capture, annotate, voxelise, predict, score:

```bash
./reconstruction_GT/1_check_camera.sh                       # USB 3? IMU?
./reconstruction_GT/record_room.sh      room08              # still + sweep + fuse
./reconstruction_GT/refine_height.sh    room08              # fitted floor -> camera_height
python -m reconstruction_GT.box_editor  captures/room08     # annotate
python -m reconstruction_GT.voxelize_gt captures/room08     # -> Label / TSDF / Mapping
python -m inference.run_inference --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --live_dir captures/room08 --out_dir ./outputs/room08
python -m reconstruction_GT.evaluate_gt captures/room08 --report
```

`record_room.sh` is steps 1, 3, 4 and 5 run in order, stopping at the first
failure. The steps are also separate, for when one of them needs redoing:

```bash
./reconstruction_GT/2_rig_test.sh                  # 15 s throwaway, then rm -rf captures/_rigtest
./reconstruction_GT/3_capture_still.sh room08      # the frame the model is scored on
./reconstruction_GT/4_record_sweep.sh  room08      # or --seconds 60
./reconstruction_GT/5_fuse_scan.sh     room08      # --min_weight 8 drops stragglers
```

Step 3 refuses to overwrite a capture that already has a `meta.json`: ground
truth may already be built against that frame. Step 5 exits 2 if any check
warns, because a warning means the mesh does not line up with the grid and
annotating against it would waste the session.

**Measuring the rig.** The grid is floor-anchored and has no pitch/roll term,
so camera height and tilt are the difference between ground truth that lines up
and ground truth that silently does not. Each of these writes a *measured*
value and records where it came from:

```bash
./reconstruction_GT/measure_lens_height.sh         # camera straight down at bare floor
./reconstruction_GT/measure_lens_height.sh room07  # and compare to that capture
./reconstruction_GT/measure_floor.sh room07        # one tilted frame; uses the IMU, tilt is fine
./reconstruction_GT/refine_height.sh room08        # floor fit -> camera_height, then re-fuses
./reconstruction_GT/refine_tilt.sh   room08        # accelerometer tilt -> meta.json, then re-fuses
./reconstruction_GT/refine_tilt.sh   room08 --from-floor   # better, once a sweep exists
```

**Annotating and voxelising.**

```bash
python -m reconstruction_GT.box_editor  captures/room08 --room_height 2.56
python -m reconstruction_GT.voxelize_gt captures/room08
python -m reconstruction_GT.voxelize_gt captures/room08 --frame live_000445   # one frame
python -m reconstruction_GT.show_gt     captures/room08 --cloud    # GT over the sweep
python -m reconstruction_GT.show_gt     captures/room08 --scored   # only what is scored
python -m reconstruction_GT.show_gt     captures/room08 --export   # write the .ply only
python -m reconstruction_GT.play_bag    captures/room08            # SPACE pause, Q quit
python -m reconstruction_GT.play_bag    captures/room08 --speed 2
```

The editor's keys are in [BOX_EDITOR.md](reconstruction_GT/document/BOX_EDITOR.md).

**Adding a frame from the sweep** so the annotation you already made scores it
too — more viewpoints out of one session's annotation:

```bash
python -m reconstruction_GT.export_frame captures/room08 --list
python -m reconstruction_GT.export_frame captures/room08 --list --rank floor
python -m reconstruction_GT.export_frame captures/room08 --frame 445
# then voxelize --frame, re-run run_inference, re-score
```

## Scoring

**A room you annotated**, with NYU's own protocol:

```bash
python -m reconstruction_GT.evaluate_gt captures/room08 --report       # -> gt/EVALUATION.md
python -m reconstruction_GT.evaluate_gt captures/room08 --fov all --report
python -m reconstruction_GT.evaluate_gt captures/room08 --frame live_000445 --report
```

Keep `room07` and `room08` in step — each is the other's control, so regenerate
both reports together.

**Every viewpoint of a sweep**, because one frame's number says as much about
the viewpoint as about the model:

```bash
python -m reconstruction_GT.sweep_eval captures/room08            # every 20th frame
python -m reconstruction_GT.sweep_eval captures/room08 --every 5
python -m reconstruction_GT.sweep_eval captures/room08 --keep 1   # best + worst onto disk in full
```

**NYU**, per class, as a markdown report:

```bash
python inference/evaluate.py                                  # tables -> stdout
python inference/evaluate.py --report inference/document/EVALUATION.md
python inference/evaluate.py --train-tables
python inference/evaluate.py --split test
```

**Occ-ScanNet**, the second dataset this fork added:

```bash
python -m inference.run_scannet --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --split val --encoder cuda
python -m inference.evaluate_scannet --pred_dir ./outputs/scannet \
    --report inference/document/EVALUATION_SCANNET.md
python -m inference.export_scannet_ply --pred_dir ./outputs/scannet
python inference/display_overlay.py outputs/scannet/ply/<scene>/<frame>_pred.ply
```

**Why SC differs between datasets**, when the scoring rule is the same — SC
moves with the difficulty of the frame, so it has to be read against its own
baseline:

```bash
python -m reconstruction_GT.sc_gap --all       # build the CSVs, then the tables
python -m reconstruction_GT.sc_gap --report    # just the tables, from the CSVs
```

**Why furniture gets extruded through the wall behind it**, measured over all
1449 NYU frames:

```bash
python -m inference.nyu_occlusion_sweep --verify_encoder --out /tmp/nyu_sweep.npz
```

What all of this established, and which numbers to quote:
[FINDINGS.md](reconstruction_GT/document/FINDINGS.md) and
[USING_THE_MODEL.md](inference/document/USING_THE_MODEL.md).

## Checks and self-tests

None of these need new ground truth, and none can be made to pass by agreeing
with themselves. Run the first two before trusting any annotation you made:

```bash
python -m reconstruction_GT.verify_voxelizer     # 9 checks, bit-exact against NYU + Occ-ScanNet
python -m reconstruction_GT.verify_voxelizer --nyu-only
python -m reconstruction_GT.evaluate_gt --nyu    # must print SC 75.0 / SSC 47.7
```

```bash
python -m inference.check_world_map algebra  captures/room07 \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth     # --accumulate vs run_inference's own output
python -m inference.check_world_map tracker  captures/room07 --pretrained_path ...
python -m inference.check_world_map measured captures/room07 --pretrained_path ...

python -m reconstruction_GT.drift_check captures/room08       # forward vs reverse
python -m reconstruction_GT.refine_frame_pose captures/room08          # report
python -m reconstruction_GT.refine_frame_pose captures/room08 --write  # apply
python -m reconstruction_GT.imu_check   captures/room08       # what the accelerometer is worth
python -m reconstruction_GT.bag_reader  <bag>                 # frame i is frame i, at any speed

python -m unittest discover -s tests -v     # pure numpy: no GPU, no bag, no checkpoint
```

After `refine_frame_pose --write`, rebuild and re-score what it touched:
`voxelize_gt --frame <stem>`, then `evaluate_gt --report`.
[DRIFT_AND_ALIGNMENT.md](reconstruction_GT/document/DRIFT_AND_ALIGNMENT.md) has
the numbers.

**Timings.** Every table in
[HARDWARE.md](inference/document/HARDWARE.md) comes out of this, so none of it
has to be taken on trust and all of it can be redone on another machine:

```bash
python -m inference.benchmark --pretrained_path ./checkpoint/CleanerS_ckpt.pth
python -m inference.benchmark --pretrained_path ... --what encode,agree
python -m inference.benchmark --pretrained_path ... --out inference/document/HARDWARE.md
#   --out writes the sections it RAN, so pair it with the full run, not with --what
```

## Long jobs

`sweep_eval` and friends take tens of minutes on the GPU, and suspending the
machine mid-run leaves the CUDA context wedged. `guard.sh` restarts a job whose
*output* goes quiet, and measures that quiet in `/proc/uptime`, which does not
advance while the machine is asleep — so closing the lid for an hour is not a
stall. It only helps because the jobs it wraps skip work already on disk.

```bash
./reconstruction_GT/guard.sh sweep8 \
    python -u -m reconstruction_GT.sweep_eval captures/room08 --every 5

touch outputs/guard/sweep8.pause     # pause; rm the file to continue
touch outputs/guard/sweep8.stop      # stop for good
tail -f outputs/guard/sweep8.log
```

## Every command, in one table

| command | what it does | writes |
| --- | --- | --- |
| `tools/hf_data.py pull \| push` | the half of the dataset git does not hold | `data/`, `checkpoint/`, `captures/*/scan/` |
| `examples/segmentation/test_NYU.py` | upstream's NYU test | `visual_pred/` |
| `examples/segmentation/run.sh` | upstream's training | `examples/segmentation/log/` |
| `inference.capture` | one still frame from the D455 | `captures/<room>/` |
| `inference.run_inference` | depth (+RGB) → TSDF → prediction → `.ply` | `outputs/<room>/{prediction,ply}` |
| `inference.run_live` | streaming camera or replay, optional accumulating room | `outputs/<run>/` |
| `inference.run_scannet` | CleanerS over Occ-ScanNet frames | `outputs/scannet/` |
| `inference.benchmark` | regenerates every timing in HARDWARE.md | stdout, `--out` |
| `inference.colorize_depth` | 16-bit depth → readable colour | `<src>_vis/` |
| `inference.evaluate` | per-class NYU tables | stdout, `--report` |
| `inference.evaluate_scannet` | per-class Occ-ScanNet tables + NYU reference rows | stdout, `--report`, `per_frame.csv` |
| `inference.export_scannet_ply` | best/worst frame per scene as `.ply` | `<pred_dir>/ply/` |
| `inference.nyu_occlusion_sweep` | how much worse the model gets once a class is hidden | `--out` npz |
| `inference.check_world_map algebra\|tracker\|measured` | keeps `--accumulate` honest against committed data | stdout |
| `inference/display.py` | voxels as points | — |
| `inference/display_solid.py` | voxels as cubes, photo-coloured | `--screenshot` |
| `inference/reconstruct.py` | the measured surface, nothing inferred | `--screenshot` |
| `inference/display_overlay.py` | prediction over measurement | — |
| `inference/show_nyu.sh` | prediction vs NYU GT, two framed windows | `custom_visual_pred/` |
| `inference/show_scannet.sh` | prediction vs Occ-ScanNet GT, two framed windows | `outputs/<run>/ply/` |
| `reconstruction_GT/1_check_camera.sh` | is the D455 there, and on USB 3 | — |
| `reconstruction_GT/2_rig_test.sh` | 15 s throwaway proving the rig works | `captures/_rigtest` (delete it) |
| `reconstruction_GT/3_capture_still.sh` | the still evaluation frame | `captures/<room>/` |
| `reconstruction_GT/4_record_sweep.sh` | the hand-held sweep | `captures/<room>/scan/scan.bag` |
| `reconstruction_GT/5_fuse_scan.sh` | sweep → one room mesh, then the checks | `scan/room.ply`, `scan/fuse_report.json` |
| `reconstruction_GT/record_room.sh` | steps 1, 3, 4, 5 in order | all of the above |
| `reconstruction_GT/measure_lens_height.sh` | lens height, camera straight down | stdout |
| `reconstruction_GT/measure_floor.sh` | lens height from one tilted frame | `<capture>/floor_check/` |
| `reconstruction_GT/refine_height.sh` | fitted floor → `camera_height`, re-fuses | `meta.json`, `scan/room.ply` |
| `reconstruction_GT/refine_tilt.sh` | measured tilt → `meta.json`, re-fuses | `meta.json`, `scan/room.ply` |
| `reconstruction_GT.record_scan` | the sweep, directly | `scan/scan.bag` |
| `reconstruction_GT.fuse_scan` | the sweep → mesh, directly | `scan/room.ply` |
| `reconstruction_GT.play_bag` | replay a sweep: photo, depth, room building up | — |
| `reconstruction_GT.bag_reader` | read a bag frame by frame, deterministically | — |
| `reconstruction_GT.box_editor` | place 3D boxes in the room cloud | `gt/solids.json` |
| `reconstruction_GT.voxelize_gt` | solids → the arrays the evaluator reads | `gt/{Label,TSDF,Mapping}/` |
| `reconstruction_GT.show_gt` | the ground truth, in the room it came from | `gt/ply/` |
| `reconstruction_GT.export_frame` | add a swept frame to the capture | `captures/<room>/{rgb,depth}`, `meta.json` |
| `reconstruction_GT.evaluate_gt` | score a capture with NYU's protocol | `gt/EVALUATION.md` |
| `reconstruction_GT.evaluate_gt --nyu` | self-check: must print SC 75.0 / 47.7 | stdout |
| `reconstruction_GT.sweep_eval` | score every viewpoint of a sweep | `outputs/<room>/sweep_eval.csv` |
| `reconstruction_GT.sc_gap` | why SC differs across datasets | `outputs/sc_gap.csv`, tables |
| `reconstruction_GT.verify_voxelizer` | 9 checks, bit-exact vs NYU + Occ-ScanNet | stdout |
| `reconstruction_GT.drift_check` | how far the trajectory drifts | `outputs/<room>/drift_check.json` |
| `reconstruction_GT.refine_frame_pose` | correct one swept frame's pose against the room | `meta.json` (`--write`) |
| `reconstruction_GT.imu_check` | the accelerometer against the tracked poses | `scan/imu_check.{json,png}` |
| `reconstruction_GT/guard.sh` | run a long job so a closed lid cannot lose it | `outputs/guard/<tag>.log` |
| `python -m unittest discover -s tests -v` | the pure-numpy rules as assertions | — |
