# Ground truth for a D455 capture — the runbook

Record a room, annotate it, score CleanerS on it. Every step has a command.

- **Why any of it works this way:** [MAKING_GT.md](MAKING_GT.md)
- **Every key in the annotation tool:** [BOX_EDITOR.md](BOX_EDITOR.md)
- **What it all established — the write-up:** [FINDINGS.md](FINDINGS.md)

Run everything from the repo root, in the `CleanerS` conda environment (the
shell scripts find the interpreter themselves, so `conda activate` is optional
for those).

---

## The whole thing

```bash
./reconstruction_GT/1_check_camera.sh                       # USB 3? IMU?
./reconstruction_GT/record_room.sh        room08            # still + sweep + fuse
./reconstruction_GT/refine_height.sh      room08            # floor -> camera_height
python -m reconstruction_GT.box_editor    captures/room08   # annotate
python -m reconstruction_GT.voxelize_gt   captures/room08   # -> Label / TSDF / Mapping
python -m inference.run_inference --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --live_dir captures/room08 --out_dir ./outputs/room08   # predict
python -m reconstruction_GT.evaluate_gt   captures/room08 --report
```

Budget: ~10 minutes with the camera, ~1–2 hours annotating, seconds for the
rest.

---

## 1. Check the camera

```bash
./reconstruction_GT/1_check_camera.sh
```

Wants `USB 3.2` and an accelerometer that opens. On USB 2 a D455 manages ~5 fps
at 848x480, which is untrackable. If the accelerometer reports `Permission
denied`, install the udev rules and replug:

```bash
sudo cp ~/CleanerS/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## 2. Rig test — first time only

```bash
./reconstruction_GT/2_rig_test.sh          # 15 s throwaway, then: rm -rf captures/_rigtest
```

Proves the previews open, the countdown runs and fusion works, before you
spend a real session on it.

## 3. Record the room

```bash
./reconstruction_GT/record_room.sh room08                # or --seconds 60
```

Three acts, with pauses between:

1. **Still frame** — level the camera, measure the lens height with a tape,
   type it in, check the DEPTH pane, press SPACE. This frame is what gets
   scored.
2. **Sweep** — hold still through the countdown, *then* lift the camera. 30–60 s,
   slowly, within ~3 m of surfaces, each object from several sides, floor in
   view at some point, and stay out of your own shot.
3. **Fuse** — automatic. It prints three checks and exits 2 if any warns.

Do the steps separately if you prefer:

```bash
./reconstruction_GT/3_capture_still.sh room08
./reconstruction_GT/4_record_sweep.sh  room08
./reconstruction_GT/5_fuse_scan.sh     room08            # --min_weight 8 to drop stragglers
```

**Aim for occlusion.** A small room shot from the corner gives a completion
metric that cannot fail — see [the first
room](MAKING_GT.md#the-first-room-and-what-it-taught). Shoot from a doorway, or
across a larger room, so real empty space is hidden behind furniture.

## 4. Fix the height

```bash
./reconstruction_GT/refine_height.sh room08              # then re-fuses for you
```

Sets `camera_height` from the fused floor so the floor lands at z = 0, keeping
your tape reading as `camera_height_tape`. On room07 the tape was out by
9.7 cm. To measure the floor from a single tilted frame instead:

```bash
./reconstruction_GT/measure_floor.sh room08
```

## 5. Watch the sweep (optional)

```bash
python -m reconstruction_GT.play_bag captures/room08           # SPACE pause, Q quit
python -m reconstruction_GT.play_bag captures/room08 --speed 2
```

Photo and depth side by side, with the room building up in 3D and the frames
`fuse_scan` would drop marked in red.

## 6. Annotate

```bash
python -m reconstruction_GT.box_editor captures/room08 --room_height 2.56
```

Place a solid per object. The usual loop for one: `N`, drive it over the object
with the arrows, `F` to fit it to the points, `G` to sit it on the floor,
`A`/`W` to push out the faces the camera never saw, then a digit for the class.
`ENTER` saves, `ESC` discards.

```
N . , TAB L         new / next / previous / list
arrows R V          move        Q E  rotate      K  rotation axis
A W S               face, grow, shrink           H  box / cylinder / wedge
F G I               fit / floor / into-wall      C  CAD mesh
M                   edit the room shell
1..9 U O            class: ..9 tvs, U furn, O objs
```

Floor, ceiling and walls come from the **room shell** (blue) — never box those.
Full reference: [BOX_EDITOR.md](BOX_EDITOR.md).

Sizes matter more than precision: a box hugs what the camera saw, so extend it
to the real furniture. `--no_ceiling` if the sweep never looked up.

## 7. Voxelize

```bash
python -m reconstruction_GT.voxelize_gt captures/room08
```

Writes `gt/Label/`, `gt/TSDF/` and `gt/Mapping/` for every frame in
`meta.json`. Re-run it after any edit to the solids — it is seconds.

## 8. Look at what you made

```bash
python -m reconstruction_GT.show_gt captures/room08 --cloud    # GT over the sweep
python -m reconstruction_GT.show_gt captures/room08 --scored   # only what is scored
python -m reconstruction_GT.show_gt captures/room08 --export   # write the .ply only
python -m reconstruction_GT.show_gt captures/room08 --frame live_000445 --fov frustum
```

Every coloured voxel should sit on or inside the grey surface it came from.
`--frame` picks one frame of a capture that holds several; `--fov frustum`
drops what that frame's camera could not see, which is what the evaluator
scores by default.

## 9. Predict

```bash
python -m inference.run_inference --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --live_dir captures/room08 --out_dir ./outputs/room08
```

Reads only `meta.json`, the depth and the RGB. It never sees the ground truth.

## 10. Score

```bash
python -m reconstruction_GT.evaluate_gt captures/room08 --report
python -m reconstruction_GT.evaluate_gt captures/room08 --fov all --report
```

Prints the per-class table and writes `captures/room08/gt/EVALUATION.md`.
`--fov frustum` (default) scores only what the camera could see; `--fov all`
scores everything annotated. They are not comparable — say which you used.

**Read the SC warning.** If the evaluator says the SC set is over 90% occupied,
that number is meaningless for that frame: a model predicting "occupied"
everywhere would beat it. SSC still stands.

## 11. Score more frames from the same sweep (optional)

The still frame is one viewpoint. The sweep holds a thousand more, already
tracked, and the annotation describes the **room**, so every one of them can be
scored against it — no extra boxing.

```bash
python -m reconstruction_GT.export_frame captures/room08 --list
python -m reconstruction_GT.export_frame captures/room08 --list --rank floor
python -m reconstruction_GT.export_frame captures/room08 --frame 445
```

`--list` walks the bag and scores every fifth tracked frame on the two things
that make a frame worth scoring: how much **floor** it sees (the class a level
camera never gets) and how much its near and far halves **overlap** (which is
where occlusion, and therefore anything for SC to measure, lives). Each is
capped and normalised, so `--rank both` means both; `--rank floor` or
`--rank occlusion` sorts on one. On room08 it puts frame 445 first — 14.2%
floor, 48/3 near/far — which is the frame used below. `--frame N` adds the frame **to the room**, under the same
`live_<six digits>` stem `capture.py` uses — bag frame 445 becomes
`live_000445` — as `depth/live_000445.png`, `rgb/live_000445.png`, plus its
camera under `frame_meta` in `meta.json`: the height and tilt of the tracked
pose, and `world_from_room`, the 4x4 that carries the room's solids into that
frame's world. There is still one annotation and one folder; from here every
command takes `--frame`:

```bash
python -m reconstruction_GT.voxelize_gt captures/room08 --frame live_000445
python -m inference.run_inference --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --live_dir captures/room08 --out_dir ./outputs/room08
python -m reconstruction_GT.evaluate_gt captures/room08 --frame live_000445 --report
```

Leave `--frame` off and each of them does every frame the capture has.

To see the whole sweep at once rather than picking one frame, score every
tracked frame against the same annotation:

```bash
python -m reconstruction_GT.sweep_eval captures/room08              # every 20th
python -m reconstruction_GT.sweep_eval captures/room08 --every 10
python -m reconstruction_GT.sweep_eval captures/room08 --keep 1     # add the
                                            # best and worst to the capture
```

One pass over the bag: ground truth, prediction and score for each frame, in
memory, written to `outputs/room08/sweep_eval.csv`. Read the `occ%` column
next to `SC` — it is what "predict occupied everywhere" would score on that
frame, so `margin` (SC minus it) is what the model actually added. `--rank
margin|sc|ssc` chooses which end the best/worst come from.

Two caveats, both in `export_frame.py`: the bag holds raw frames, so an
exported one keeps its lens distortion (up to ~3 cm at the corners), and its
pose comes from tracking rather than a tripod. A tilted frame also loses a lot
of grid — frame 445 looks 22.7° down, and 73% of its gravity-aligned grid falls
outside the view, which is why it scores 1684 voxels where the room holds 9798.

## 12. Compare prediction with ground truth

```bash
python inference/display_overlay.py outputs/room08/ply/live_000000.ply
python inference/display_overlay.py outputs/room08/ply/live_000000_gt.ply
```

Add `--frame-grid` to both so the same voxel sits at the same place on screen,
`--show surface|voxels|both`, `--colour class|photo`, or `--screenshot out.png`.

---

## Where everything lands

One room, one folder — and one annotation, however many frames it holds.

```
captures/room08/
  meta.json                 cam_K, camera_height (+ _tape, _source), yaw,
                            "frames", and "frame_meta" for any frame lifted
                            out of the sweep (its own height, tilt, cam_K and
                            `world_from_room`)
  depth/live_000000.png     640x480 uint16 mm  <- the model's input
  depth/live_000445.png     bag frame 445 of the sweep, same contract
  rgb/…                     one per frame
  scan/scan.bag             the raw sweep (~30 MB/s; delete once fused)
  scan/room.ply             fused mesh          room_cloud.ply  for annotating
  scan/trajectory.txt       bag frame index + 4x4 pose, per tracked frame
  scan/fuse_report.json     frames fused/dropped + the three checks
  gt/solids.json            YOUR ANNOTATION — the only irreplaceable file,
                            in the ROOM's world, shared by every frame
  gt/Label/<frame>.npz      label3d, 60x36x60, 255 retained
  gt/TSDF/<frame>.npz       arr_0 model-input TSDF, arr_1 label_weight
  gt/Mapping/<frame>.npz    which voxels a depth pixel reached (SC needs it)
  gt/ply/<frame>_gt.ply     the ground truth, viewable (_gt_fov: this view only)
  gt/EVALUATION.md          the score
outputs/room08/
  prediction/<frame>.npy    the model's 60x36x60 output
  ply/<frame>.ply           prediction, and a copy of the GT for comparison
```

Back up `gt/solids.json`. Everything else regenerates from the bag and it.

`scan/scan.bag` and `scan/room*.ply` are git-ignored and live on Hugging Face,
so anything that re-fuses the mesh -- `5_fuse_scan.sh`, `refine_tilt.sh`,
`refine_height.sh` -- leaves git holding the new `trajectory.txt` and
`solids.json` while HF still has the mesh they were built against. Push after
committing, or a fresh `pull` elsewhere pairs the annotation with the wrong
geometry:

```bash
python3 tools/hf_data.py push --dry-run      # DATA.md: what moves with what
python3 tools/hf_data.py push
```

## Checks worth re-running

```bash
python -m reconstruction_GT.verify_voxelizer          # 9 checks vs NYU + Occ-ScanNet
python -m reconstruction_GT.evaluate_gt --nyu         # must print SC 75.0 / SSC 47.7
python -m reconstruction_GT.bag_reader <bag>          # frame i is frame i, at any speed
python -m reconstruction_GT.imu_check captures/room08 # what the accelerometer is worth
```

The first proves the voxelizer still reproduces NYU's shipped files bit-exactly;
the second proves the evaluator still measures what the repo measures. Run them
after touching either. The third proves a bag still reads the same frames
whether the loop is fast or slow, which is what lets a frame index address a
pose -- MAKING_GT.md, gotcha 10.

The fourth measures the accelerometer against the only independent thing that
knows which way is down, the fused floor. On this D455 it reads 9.02 m/s^2 at
rest instead of 9.81 -- an 8% scale error, harmless because only the DIRECTION
is used -- with 0.04 deg of noise but **2.5 deg of direction error**. That is
why `refine_tilt.sh --from-floor` exists and why the grid should follow the
floor rather than the IMU. It writes a plot of the whole bag to
`captures/<room>/scan/imu_check.png`.

## Comparing against the benchmarks

```bash
python -m reconstruction_GT.sc_gap --all        # build the CSVs, then the tables
python -m reconstruction_GT.sc_gap --report     # just the tables, from the CSVs
```

`sc_gap.py` measures the same quantities on NYU test, a 1500-frame sample of
Occ-ScanNet val, and both rooms, so a correlation measured here can be read
against the benchmarks instead of being mistaken for a fact about SC. It writes
four CSVs, each a stage that is skipped if the file already exists:

| file | what it holds |
| --- | --- |
| `outputs/nyu_perframe.csv` | SC, its trivial baseline, the margin and SSC per NYU test frame |
| `outputs/nyu_viewpoints.csv` | the above plus camera height, median depth and hidden extent |
| `outputs/scannet_viewpoints.csv` | the same on Occ-ScanNet, plus GT floor voxel count and tilt |
| `outputs/sc_gap.csv` | precision, recall and the baseline per frame, all four datasets |

The first three are what `evaluate_gt.py --report` reads for its NYU and
ScanNet reference columns, so build them before regenerating a room report that
should carry those columns. The fourth answers why SC is ~79 on NYU and ~55 on
Occ-ScanNet: matched on the baseline, recall is comparable and precision is not,
which is `document/EVALUATION_SCANNET.md` section 3.

## Long jobs, and a laptop that sleeps

A sweep is tens of minutes on the GPU. Run it under the guard and a closed lid
cannot lose it:

```bash
./reconstruction_GT/guard.sh sweep8 \
    python -u -m reconstruction_GT.sweep_eval captures/room08 --every 5

touch  outputs/guard/sweep8.pause     # pause          rm the file to continue
touch  outputs/guard/sweep8.stop      # stop for good
tail -f outputs/guard/sweep8.log
```

It restarts the job if its output goes quiet for five minutes of AWAKE time --
quiet measured in `/proc/uptime`, which does not advance while the machine is
asleep, so sleeping is not mistaken for a wedged GPU. Restarting is cheap
because the jobs resume: `run_inference` skips frames that already have a
prediction (`--overwrite` to redo) and `sweep_eval` keeps its CSV and skips
frames already in it (`--restart` to ignore).

## When something looks wrong

| symptom | cause | what to do |
| --- | --- | --- |
| `USB 2.1` in step 1 | charge-only cable, or a USB-2 port | different cable; blue/SS port |
| accelerometer `Permission denied` | udev rules missing | the `sudo cp` in step 1, then replug |
| `still_match` warns | the camera moved between still and sweep | re-record both |
| `floor` warns | `camera_height` wrong, or the sweep never saw the floor | `refine_height.sh`, or pass `--room_height` |
| `imu_tilt` warns | camera not level; the grid has no roll term | re-level and re-record; 2.45° costs 13 cm at 3 m |
| tracking drops many frames | swung too fast, or a blank wall | `play_bag.py` shows where; sweep slower |
| the shell overshoots a wall | IR goes through glass | `M`, pick that wall, `S` to pull it in |
| a ceiling that was never seen | level camera in a small room | `--no_ceiling`, or `--room_height` |
| CloudCompare shows "175729776 point clouds" | the Flatpak read the wrong file | `/usr/bin/CloudCompare -O <path>` |
| an exported frame's GT floats off its surfaces | the capture was fused before `bag_reader` | re-fuse, then re-export the frame |
| `No block is touched in TSDF volume` | depth scale in the wrong convention | `bag_metadata()['depth_scale_o3d']` |
| SC IoU near 1.0 | the SC set has no empty space | capture with occlusion; quote SSC only |
