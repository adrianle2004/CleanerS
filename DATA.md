# Data and large files

This repo is code, docs, reports and the small parts of the captures. Everything
else lives in a **private** Hugging Face dataset repo and is moved with
`tools/hf_data.py`.

## Workspace layout

The code expects `data/` to sit next to the repo, not inside it:

```
<workspace>/
  CleanerS/      this repo
  data/          NYU/, NYUCAD/, Scannet/, utils/ (DataProcess CUDA extension), ...
```

Paths are resolved from that layout (`cfgs/default.yaml`, `cleaner/dataset/NYU/NYU.py`,
`inference/`, `reconstruction_GT/`), so the workspace can live anywhere.

## Getting the data

```bash
pip install huggingface_hub            # any python >= 3.9, not the py3.7 CleanerS env
hf auth login                          # once, with a token that can read the repo
python tools/hf_data.py pull           # ~24 GB; --skip-images leaves out ScanNet's 15 GB of posed_images
```

The repo defaults to `<your HF user>/cleaners-data`; use `--repo user/name` or
`CLEANERS_HF_REPO` for another one. `pull` only fetches what is missing, so it is
safe to rerun.

## What is where

| Local | On HF | Size |
| --- | --- | ---: |
| `data/**` | same path | 8.5 GB, 35k files |
| `data/Scannet/posed_images/` | `Scannet/posed_images_shards/*.tar`, 50 scenes each | 15 GB |
| symlinks under `data/` (`Scannet_subset654/`, `NYU/test/RGB/`) | `symlinks.tsv`, recreated as relative links | |
| `captures/room0{6,7,8}/scan/scan.bag` | `repo/captures/...` | 5.1 GB |
| `captures/*/scan/room*.ply` (room meshes) | `repo/captures/...` | 0.13 GB |
| `checkpoint/*.pth` | `repo/checkpoint/...` | 0.6 GB |

Not stored anywhere, regenerate them: `outputs/scannet*/` (`inference/run_scannet.py`),
`custom_visual_pred/`, `visual_pred/` (`examples/segmentation/test_NYU.py`).

After `pull`, rebuild the CUDA extension if `import DataProcess` fails on the new
machine (`data/utils/setup_datautil.py`, `data/utils/CMakeLists.txt`).

## Why the data is private

ScanNet's terms of use do not allow redistributing the data, and the Occ-ScanNet
ground truth in `data/Scannet/<scene>/*.npz` is derived from it. The room captures
are 3D scans of real rooms. Keep the HF repo private; do not move any of it into
this public repo.

## Adding data

Put it in place locally, then `python tools/hf_data.py push`. Push only uploads
paths HF does not have yet; `--dry-run` lists them first. A file changed in place
keeps its old version on HF until it is removed there.
