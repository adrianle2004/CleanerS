# Setting up on a new machine

From nothing to a reproduced NYU score: GitHub for the code, a private Hugging Face
dataset repo for everything else (datasets, RealSense recordings, checkpoints).
[DATA.md](DATA.md) says what lives where and why.

| | |
| --- | --- |
| Code | https://github.com/adrianle2004/CleanerS (public fork of `fereenwong/CleanerS`) |
| Data | https://huggingface.co/datasets/adrianle2004/cleaners-data (**private**) |
| Verified on | Ubuntu, RTX 3050 Ti laptop, driver 595, CUDA 12.0 `nvcc`, conda env `CleanerS` (py3.7) |

## 1. Workspace and code

The code expects `data/` **next to** the repo, so make a parent folder:

```bash
mkdir -p ~/CleanerS && cd ~/CleanerS
git clone git@github.com:adrianle2004/CleanerS.git
cd CleanerS
git remote add upstream https://github.com/fereenwong/CleanerS.git   # the original, for updates
```

## 2. Python environment (`CleanerS`, py3.7)

The model stack is old and pinned: torch 1.10.1 + CUDA 11.1 + mmcv-full 1.5.0. The
conda torch ships its own CUDA runtime, so it runs on a newer driver.

```bash
conda create -n CleanerS python=3.7 -y
conda activate CleanerS
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install -r requirements.txt
pip install pickle5            # Occ-ScanNet GT pickles are protocol 5
pip install open3d==0.17.0     # only for the viewers and the GT tools (box_editor, fuse_scan, ...)
```

Check:

```bash
python -c "import torch, mmcv, mmseg; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__)"
# 1.10.1 True 1.5.0
```

## 3. Data from Hugging Face

`tools/hf_data.py` needs a **newer** python than the env above; the base conda
python (or any python >= 3.9) is fine. Use a token from
https://huggingface.co/settings/tokens: *read* is enough to pull, *write* to push.

```bash
conda deactivate                         # back to base python
pip install huggingface_hub
hf auth login
cd ~/CleanerS/CleanerS
python tools/hf_data.py pull             # ~24 GB
# python tools/hf_data.py pull --skip-images   # ~9 GB: leaves out ScanNet posed_images,
#                                              # needed only for run_scannet.py
```

It fills in:

```
~/CleanerS/data/                         NYU, NYUCAD, Scannet, utils, symlink trees
~/CleanerS/CleanerS/checkpoint/          CleanerS_ckpt.pth, Teacher_ckpt.pth, mit_b2.pth
~/CleanerS/CleanerS/captures/room0{6,7,8}/scan/      scan.bag, room*.ply
```

It only fetches what is missing, so rerun it if the connection drops.

## 4. Check it: reproduce NYU test

```bash
conda activate CleanerS
cd ~/CleanerS/CleanerS/examples/segmentation
python test_NYU.py --cfg ../../cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ../../checkpoint/CleanerS_ckpt.pth --no_visualize
```

Expected: **SC 74.95, SSC mIoU 47.73** (654 frames, a few minutes). Run it from
`examples/segmentation/`: the config's paths (`../../checkpoint`, `../../../data/NYU`)
are relative to that folder.

## 5. Optional: the CUDA TSDF encoder

`inference.frame_loader.FrameLoader(encoder='cuda')` is ~4.5x faster than the numpy
encoder (285 vs 1317 ms/frame). The prebuilt `data/utils/DataProcess*.so` and
`data/utils/build/libdatautil.so` come down with the data and work as-is on a
machine like the one above. Check:

```bash
python -c "from inference.frame_loader import load_cuda_encoder as l; print(l())"
# <module 'DataProcess' ...>   -> usable;   None -> rebuild
```

`encoder='auto'` falls back to numpy on its own, so this only matters for speed.
To rebuild (needs a system CUDA toolkit and gcc-12; `CMakeLists.txt` hardcodes
`/usr/bin/gcc-12`):

```bash
sudo apt install nvidia-cuda-toolkit gcc-12 g++-12 cmake
conda activate CleanerS
cd ~/CleanerS/data/utils
mkdir -p build && (cd build && cmake .. && make)      # -> build/libdatautil.so
python setup_datautil.py build_ext --inplace          # -> DataProcess.cpython-37m-*.so
```

## 6. Optional: the D455 camera

`pyrealsense2` from `requirements.txt` bundles its own librealsense (libusb backend),
so no kernel module is needed. If the camera is not found without `sudo`, install
librealsense's udev rules (`config/99-realsense-libusb.rules` in
https://github.com/IntelRealSense/librealsense into `/etc/udev/rules.d/`, then
`sudo udevadm control --reload-rules && sudo udevadm trigger`) and replug.

```bash
./reconstruction_GT/1_check_camera.sh      # USB 3? IMU?
```

Then [reconstruction_GT/README.md](reconstruction_GT/README.md) is the capture → GT →
evaluation runbook, and [inference/document/README.md](inference/document/README.md)
the module reference.

## Day-to-day

```bash
# code
git pull                                 # your fork
git fetch upstream                       # the original authors' repo, if it ever changes

# data you added or recorded (new capture, new dataset folder)
python tools/hf_data.py push --dry-run   # what would go up
python tools/hf_data.py push             # only new paths are sent
```

`captures/*/scan/scan.bag`, `captures/*/scan/room*.ply` and `checkpoint/*.pth` are
git-ignored; everything else under `captures/` (photos, depth, `meta.json`, `gt/`)
is committed normally. Regenerable outputs (`outputs/scannet*`, `custom_visual_pred/`,
`visual_pred/`) are stored nowhere; rebuild them with the scripts that made them.
