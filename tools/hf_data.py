"""Move the files git does not hold to and from a private Hugging Face dataset repo.

The workspace this expects (see DATA.md):

    <ws>/CleanerS/   this repo
    <ws>/data/       NYU, NYUCAD, Scannet, utils (the DataProcess CUDA extension), ...

What lives on HF, and where:

    data/<file> and data/<dir>/<file>     -> same path, as is
    data/<dir>/<subdir>/...               -> packs/<dir>/<subdir>.tar
    data/Scannet/<scene>/...              -> packs/Scannet/gt_NNN.tar            50 scenes each
    data/Scannet/posed_images/<scene>/... -> packs/Scannet/posed_images_NNN.tar  50 scenes each
    every symlink under data/             -> symlinks.tsv, remade as relative links
                                             (Scannet_subset654/ and NYU/test/RGB/ are link trees)
    CleanerS/<file matched by REPO_FILES> -> repo/<same path>

Packing is not optional: the Hub rate-limits per-file API calls, and 35k small
files uploaded one by one ran at 10 files a minute.

    python tools/hf_data.py push --dry-run    what would be uploaded, nothing sent
    python tools/hf_data.py push              upload whatever HF does not have yet (resumable)
    python tools/hf_data.py pull              download into place and unpack

A pack is skipped when its name is already on HF, so a folder that changed after
it was pushed needs `push --repack NYU/depth` (or a glob: 'Scannet/gt_*').

Needs huggingface_hub >= 1.0 and `hf auth login` once. It runs in any python >= 3.9;
it does not need the CleanerS env (py3.7 is too old for huggingface_hub).
"""
import argparse
import collections
import fnmatch
import glob
import hashlib
import os
import shutil
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(REPO), 'data')
TMP = os.path.join(os.path.dirname(REPO), '.hf_tmp')
DEFAULT_NAME = 'cleaners-data'

# the files in this repo that .gitignore keeps out of git
REPO_FILES = ['captures/*/scan/scan.bag', 'captures/*/scan/room*.ply', 'checkpoint/*.pth']
DATA_IGNORE = ['*/__pycache__/*', '*.pyc', '*.log', '.cache/*', '.hf_packs/*', 'symlinks.tsv']
LINKS = 'symlinks.tsv'
DONE = '.hf_packs'          # data/.hf_packs/<pack>.done once a pack is unpacked
SCENES_PER_PACK = 50
# loose files per commit
MAX_FILES, MAX_BYTES = 500, 4 << 30


def repo_id(api, name):
    return name if '/' in name else '%s/%s' % (api.whoami()['name'], name)


def walk_data():
    """(files, links): real files as (local path, rel), and every symlink as
    (rel, target). A target inside data/ is kept relative to data/ ('data:...'),
    so the links survive a move to another machine."""
    out, links = [], []
    real = os.path.realpath(DATA)
    for root, dirs, files in os.walk(DATA):
        dirs.sort()
        for f in sorted(dirs + files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, DATA)
            if any(fnmatch.fnmatch(rel, pat) for pat in DATA_IGNORE):
                continue
            if os.path.islink(p):
                t = os.path.normpath(os.path.join(root, os.readlink(p)))
                links.append((rel, 'data:' + os.path.relpath(t, real)
                              if t.startswith(real + os.sep) else t))
            elif f in files:
                out.append((p, rel))
    return out, links


def scene_index(parent):
    """scene -> pack number, from the sorted scene folders under data/<parent>."""
    d = os.path.join(DATA, parent)
    scenes = sorted(s for s in os.listdir(d) if s.startswith('scene')
                    and os.path.isdir(os.path.join(d, s)) and not os.path.islink(os.path.join(d, s)))
    return {s: i // SCENES_PER_PACK for i, s in enumerate(scenes)}


def plan():
    """(loose, packs, links): loose [(local path, repo path)], packs
    {pack name: [(local path, rel)]}, links [(rel, target)]."""
    files, links = walk_data()
    idx = {'Scannet': {}, 'Scannet/posed_images': {}}
    if os.path.isdir(os.path.join(DATA, 'Scannet')):
        idx = {k: scene_index(k) if os.path.isdir(os.path.join(DATA, k)) else {} for k in idx}
    loose, packs = [], collections.defaultdict(list)
    for p, rel in files:
        parts = rel.split('/')
        if len(parts) <= 2:
            loose.append((p, rel))
            continue
        if parts[0] == 'Scannet' and parts[1] == 'posed_images':
            key = 'Scannet/posed_images_%03d' % idx['Scannet/posed_images'][parts[2]]
        elif parts[0] == 'Scannet' and parts[1].startswith('scene'):
            key = 'Scannet/gt_%03d' % idx['Scannet'][parts[1]]
        else:
            key = parts[0] + '/' + parts[1]
        packs[key].append((p, rel))
    for p in REPO_FILES:
        for f in sorted(glob.glob(os.path.join(REPO, p))):
            loose.append((f, 'repo/' + os.path.relpath(f, REPO)))
    return loose, dict(packs), links


def pack_path(key):
    return 'packs/%s.tar' % key


def _digest(path, algo, header=b''):
    h = hashlib.new(algo)
    h.update(header)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def differs(path, info):
    """True when the local file is not what HF already holds.

    Matching on the path alone would let an edited file sit on HF forever --
    push would keep reporting nothing to do. HF gives a git blob SHA-1 for a
    small file and an LFS SHA-256 for a large one; size is checked first
    because it settles most cases without reading the file.
    """
    if info is None:
        return True
    size = getattr(info, 'size', None)
    if size is not None and size != os.path.getsize(path):
        return True
    lfs = getattr(info, 'lfs', None)
    if lfs is not None:
        return lfs.sha256 != _digest(path, 'sha256')
    return info.blob_id != _digest(
        path, 'sha1', b'blob %d\0' % os.path.getsize(path))


def stale_on_hf(api, rid, pairs):
    """The subset of (local, rel) already on HF whose content has changed."""
    info = {}
    rels = [rel for _, rel in pairs]
    for i in range(0, len(rels), 256):
        for x in api.get_paths_info(rid, rels[i:i + 256], expand=True,
                                    repo_type='dataset'):
            info[x.path] = x
    return [(p, rel) for p, rel in pairs if differs(p, info.get(rel))]


def batches(items):
    batch, size = [], 0
    for p, rel in items:
        n = os.path.getsize(p)
        if batch and (len(batch) >= MAX_FILES or size + n > MAX_BYTES):
            yield batch
            batch, size = [], 0
        batch.append((p, rel))
        size += n
    if batch:
        yield batch


def gb(n):
    return '%.2f GB' % (n / 1e9)


def push(args):
    from huggingface_hub import HfApi, CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi()
    loose, packs, links = plan()
    rid = repo_id(api, args.repo)
    if not args.dry_run:
        api.create_repo(rid, repo_type='dataset', private=True, exist_ok=True)
    try:
        remote = set(api.list_repo_files(rid, repo_type='dataset'))
    except RepositoryNotFoundError:             # dry run before the first push
        remote = set()

    fresh = [(p, rel) for p, rel in loose if rel not in remote]
    known = [(p, rel) for p, rel in loose if rel in remote]
    # a known path still has to be checked: its content may have changed
    changed = stale_on_hf(api, rid, known) if known else []
    loose = fresh + changed
    if changed:
        print('%d file(s) changed since they were uploaded:' % len(changed))
        for _, rel in changed:
            print('  ' + rel)
    repack = lambda k: any(fnmatch.fnmatch(k, pat) for pat in args.repack)
    todo = sorted(k for k in packs if pack_path(k) not in remote or repack(k))
    size = lambda fs: sum(os.path.getsize(p) for p, _ in fs)
    print('to upload: %d loose files (%s), %d packs (%d files, %s), %d symlinks'
          % (len(loose), gb(size(loose)), len(todo), sum(len(packs[k]) for k in todo),
             gb(sum(size(packs[k]) for k in todo)), len(links)))
    if args.dry_run:
        for p, rel in loose:
            print('  %9s  %s' % (gb(os.path.getsize(p)), rel))
        for k in todo:
            print('  %9s  %s  (%d files)' % (gb(size(packs[k])), pack_path(k), len(packs[k])))
        return

    tsv = ''.join('%s\t%s\n' % l for l in links).encode()
    api.upload_file(path_or_fileobj=tsv, path_in_repo=LINKS, repo_id=rid,
                    repo_type='dataset', commit_message='update ' + LINKS)
    for batch in batches(loose):
        print('loose: %d files, %s' % (len(batch), gb(size(batch))))
        api.create_commit(rid, repo_type='dataset',
                          operations=[CommitOperationAdd(rel, p) for p, rel in batch],
                          commit_message='add %d files (%s ...)' % (len(batch), batch[0][1]))

    os.makedirs(TMP, exist_ok=True)
    for n, k in enumerate(todo):
        # one at a time: the disk has no room for a second copy of the data
        tar = os.path.join(TMP, 'pack.tar')
        with tarfile.open(tar, 'w') as t:
            for p, rel in packs[k]:
                t.add(p, arcname=rel, recursive=False)
        print('[%d/%d] %s: %d files, %s' % (n + 1, len(todo), pack_path(k), len(packs[k]),
                                            gb(os.path.getsize(tar))))
        api.upload_file(path_or_fileobj=tar, path_in_repo=pack_path(k), repo_id=rid,
                        repo_type='dataset', commit_message='add ' + pack_path(k))
        os.remove(tar)
    shutil.rmtree(TMP, ignore_errors=True)
    print('done: https://huggingface.co/datasets/' + rid)


def pull(args):
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    api = HfApi()
    rid = repo_id(api, args.repo)
    os.makedirs(DATA, exist_ok=True)
    print('loose files ...')
    snapshot_download(rid, repo_type='dataset', local_dir=DATA,
                      ignore_patterns=['repo/*', 'packs/*'])

    files = api.list_repo_files(rid, repo_type='dataset')
    for rel in [f for f in files if f.startswith('repo/')]:
        dst = os.path.join(REPO, rel[len('repo/'):])
        if os.path.exists(dst):
            continue
        print(rel[len('repo/'):])
        got = hf_hub_download(rid, rel, repo_type='dataset', local_dir=TMP)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(got, dst)

    for rel in sorted(f for f in files if f.startswith('packs/')):
        key = rel[len('packs/'):-len('.tar')]
        if args.skip_images and key.startswith('Scannet/posed_images_'):
            continue
        done = os.path.join(DATA, DONE, key + '.done')
        if os.path.exists(done):
            continue
        print(rel)
        got = hf_hub_download(rid, rel, repo_type='dataset', local_dir=TMP)
        with tarfile.open(got) as t:
            t.extractall(DATA)
        os.remove(got)
        os.makedirs(os.path.dirname(done), exist_ok=True)
        open(done, 'w').close()

    # last: link trees point into the unpacked folders
    links = os.path.join(DATA, LINKS)
    if os.path.exists(links):
        for line in open(links):
            rel, target = line.rstrip('\n').split('\t')
            p = os.path.join(DATA, rel)
            if os.path.lexists(p):
                continue
            if target.startswith('data:'):
                target = os.path.relpath(os.path.join(DATA, target[5:]), os.path.dirname(p))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            os.symlink(target, p)
    shutil.rmtree(TMP, ignore_errors=True)
    print('done')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cmd', choices=['push', 'pull'])
    p.add_argument('--repo', default=os.environ.get('CLEANERS_HF_REPO', DEFAULT_NAME),
                   help='HF dataset repo, "name" (under your account) or "user/name"; '
                        'default $CLEANERS_HF_REPO or %s' % DEFAULT_NAME)
    p.add_argument('--dry-run', action='store_true', help='push: list, send nothing')
    p.add_argument('--repack', nargs='*', default=[], metavar='PACK',
                   help='push: re-upload these packs even if HF has them (globs ok)')
    p.add_argument('--skip-images', action='store_true',
                   help='pull: leave out the 15 GB of ScanNet posed_images')
    args = p.parse_args()
    if not os.path.isdir(DATA) and args.cmd == 'push':
        sys.exit('no data folder at %s' % DATA)
    {'push': push, 'pull': pull}[args.cmd](args)


if __name__ == '__main__':
    main()
