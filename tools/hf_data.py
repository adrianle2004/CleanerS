"""Move the files git does not hold to and from a private Hugging Face dataset repo.

The workspace this expects (see DATA.md):

    <ws>/CleanerS/   this repo
    <ws>/data/       NYU, NYUCAD, Scannet, utils (the DataProcess CUDA extension), ...

What lives on HF, and where:

    <data file>                               -> <same relative path>
    data/Scannet/posed_images/<scene>/...     -> Scannet/posed_images_shards/posed_images_NNN.tar
                                                 (205k files packed 50 scenes a tar; HF wants
                                                 repos under ~100k files)
    CleanerS/<file matched by REPO_FILES>     -> repo/<same relative path>
    every symlink under data/                 -> symlinks.tsv (Scannet_subset654 and
                                                 NYU/test/RGB are link trees; pull remakes
                                                 them as relative links)

    python tools/hf_data.py push --dry-run    what would be uploaded, nothing sent
    python tools/hf_data.py push              upload whatever HF does not have yet (resumable)
    python tools/hf_data.py pull              download into place, unpack the image shards

Needs huggingface_hub >= 1.0 and `hf auth login` once. It runs in any python >= 3.9;
it does not need the CleanerS env (py3.7 is too old for huggingface_hub).
"""
import argparse
import fnmatch
import glob
import os
import shutil
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(REPO), 'data')
DEFAULT_NAME = 'cleaners-data'

# the files in this repo that .gitignore keeps out of git
REPO_FILES = ['captures/*/scan/scan.bag', 'captures/*/scan/room*.ply', 'checkpoint/*.pth']
DATA_IGNORE = ['*/__pycache__/*', '*.pyc', '*.log', '.cache/*', 'Scannet/posed_images/*',
               'symlinks.tsv']

LINKS = 'symlinks.tsv'
SHARD_DIR = 'Scannet/posed_images_shards'
SCENES_PER_SHARD = 50
# per commit; big enough to be quick, small enough that a dropped push loses little
MAX_FILES, MAX_BYTES = 2000, 4 << 30


def repo_id(api, name):
    return name if '/' in name else '%s/%s' % (api.whoami()['name'], name)


def walk_data():
    """(files, links): real files as (local path, path in repo), and every symlink
    as (path in repo, target). A target inside data/ is kept relative to data/,
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
                t = os.readlink(p)
                t = os.path.join(root, t) if not os.path.isabs(t) else t
                t = os.path.normpath(t)
                links.append((rel, 'data:' + os.path.relpath(t, real)
                              if t.startswith(real + os.sep) else t))
            elif f in files:
                out.append((p, rel))
    return out, links


def local_files():
    """(local path, path in the HF repo) for everything except the image shards."""
    out = walk_data()[0]
    for pat in REPO_FILES:
        for p in sorted(glob.glob(os.path.join(REPO, pat))):
            out.append((p, 'repo/' + os.path.relpath(p, REPO)))
    return out


def shards():
    """[(shard path in repo, [scene, ...])], a fixed split so reruns name them the same."""
    src = os.path.join(DATA, 'Scannet', 'posed_images')
    if not os.path.isdir(src):
        return []
    scenes = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    return [('%s/posed_images_%03d.tar' % (SHARD_DIR, i // SCENES_PER_SHARD),
             scenes[i:i + SCENES_PER_SHARD])
            for i in range(0, len(scenes), SCENES_PER_SHARD)]


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
    api = HfApi()
    rid = None if args.dry_run else repo_id(api, args.repo)
    if rid:
        api.create_repo(rid, repo_type='dataset', private=True, exist_ok=True)
        remote = set(api.list_repo_files(rid, repo_type='dataset'))
    else:
        remote = set()

    todo = [(p, rel) for p, rel in local_files() if rel not in remote]
    sh = [(rel, sc) for rel, sc in shards() if rel not in remote]
    print('to upload: %d files, %s; %d image shards (%d scenes)'
          % (len(todo), gb(sum(os.path.getsize(p) for p, _ in todo)),
             len(sh), sum(len(sc) for _, sc in sh)))
    if args.dry_run:
        for p, rel in todo:
            if os.path.getsize(p) > 50e6 or rel.startswith('repo/'):
                print('  %9s  %s' % (gb(os.path.getsize(p)), rel))
        return

    links = walk_data()[1]
    print('%d symlinks -> %s' % (len(links), LINKS))
    tsv = ''.join('%s\t%s\n' % l for l in links).encode()
    api.upload_file(path_or_fileobj=tsv, path_in_repo=LINKS, repo_id=rid,
                    repo_type='dataset', commit_message='update ' + LINKS)

    for i, batch in enumerate(batches(todo)):
        print('commit %d: %d files, %s' % (i + 1, len(batch),
                                           gb(sum(os.path.getsize(p) for p, _ in batch))))
        api.create_commit(rid, repo_type='dataset',
                          operations=[CommitOperationAdd(rel, p) for p, rel in batch],
                          commit_message='add %d files (%s ...)' % (len(batch), batch[0][1]))

    src = os.path.join(DATA, 'Scannet')
    tmp = os.path.join(os.path.dirname(REPO), '.hf_tmp')
    os.makedirs(tmp, exist_ok=True)
    for rel, scenes in sh:
        # one at a time: the disk cannot hold a second copy of 15 GB
        tar = os.path.join(tmp, os.path.basename(rel))
        with tarfile.open(tar, 'w') as t:
            for s in scenes:
                t.add(os.path.join(src, 'posed_images', s), arcname='posed_images/' + s)
        print('%s: %d scenes, %s' % (rel, len(scenes), gb(os.path.getsize(tar))))
        api.upload_file(path_or_fileobj=tar, path_in_repo=rel, repo_id=rid,
                        repo_type='dataset', commit_message='add ' + rel)
        os.remove(tar)
    shutil.rmtree(tmp, ignore_errors=True)
    print('done: https://huggingface.co/datasets/' + rid)


def pull(args):
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    api = HfApi()
    rid = repo_id(api, args.repo)
    os.makedirs(DATA, exist_ok=True)
    print('data/ ...')
    snapshot_download(rid, repo_type='dataset', local_dir=DATA,
                      ignore_patterns=['repo/*', SHARD_DIR + '/*'])

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

    files = api.list_repo_files(rid, repo_type='dataset')
    tmp = os.path.join(os.path.dirname(REPO), '.hf_tmp')
    for rel in [f for f in files if f.startswith('repo/')]:
        dst = os.path.join(REPO, rel[len('repo/'):])
        if os.path.exists(dst):
            continue
        print(rel[len('repo/'):])
        got = hf_hub_download(rid, rel, repo_type='dataset', local_dir=tmp)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(got, dst)

    if not args.skip_images:
        dst = os.path.join(DATA, 'Scannet')
        for rel in sorted(f for f in files if f.startswith(SHARD_DIR + '/')):
            done = os.path.join(dst, 'posed_images', '.%s.done' % os.path.basename(rel))
            if os.path.exists(done):
                continue
            print(rel)
            got = hf_hub_download(rid, rel, repo_type='dataset', local_dir=tmp)
            with tarfile.open(got) as t:
                t.extractall(dst)
            os.remove(got)
            open(done, 'w').close()
    shutil.rmtree(tmp, ignore_errors=True)
    print('done')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cmd', choices=['push', 'pull'])
    p.add_argument('--repo', default=os.environ.get('CLEANERS_HF_REPO', DEFAULT_NAME),
                   help='HF dataset repo, "name" (under your account) or "user/name"; '
                        'default $CLEANERS_HF_REPO or %s' % DEFAULT_NAME)
    p.add_argument('--dry-run', action='store_true', help='push: list, send nothing')
    p.add_argument('--skip-images', action='store_true',
                   help='pull: leave out the 15 GB of ScanNet posed_images')
    args = p.parse_args()
    if not os.path.isdir(DATA) and args.cmd == 'push':
        sys.exit('no data folder at %s' % DATA)
    {'push': push, 'pull': pull}[args.cmd](args)


if __name__ == '__main__':
    main()
