"""Make paths in the cfgs independent of the working directory.

The two entry points disagree about where you stand: `test_NYU.py` defaults to
`--cfg ../../cfgs/NYU/voxelSSC.yaml` and is run from `examples/segmentation`,
while `run_inference.py`, `run_live.py`, `run_scannet.py` and `sweep_eval.py`
all document `--pretrained_path ./checkpoint/...` and are run from the repo
root. A plain relative path in a cfg can only satisfy one of them.

So a cfg path is read as relative to the REPO, not to the caller:

    pretrained: checkpoint/mit_b2.pth     -> <repo>/checkpoint/mit_b2.pth
    data_root:  ../data/NYU              -> <repo>/../data/NYU, the sibling
                                            data folder (DATA.md)

A path that already resolves from the working directory is left alone, so
anything typed on the command line keeps working exactly as before.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve(path):
    """Absolute path for `path`, trying the working directory then the repo.

    Returns `path` untouched when it is empty, absolute, or found in neither
    place -- an unresolvable path is left for the caller to report, so the
    error still names what was actually asked for.
    """
    if not path or not isinstance(path, str) or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    from_repo = os.path.normpath(os.path.join(REPO_ROOT, path))
    return from_repo if os.path.exists(from_repo) else path
