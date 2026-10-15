"""Regenerate the measurements in document/HARDWARE.md.

Every table on that page comes out of this script, so none of it has to be
taken on trust, and all of it can be redone on another machine. One section
per table:

    frame    the end-to-end table -- encode / infer / ply per frame for each
             (encoder, amp) pair. This is where the 4.2x comes from.
    encode   the encode stage on its own, per arithmetic and per torch
             configuration, each with its peak VRAM.
    agree    whether the fast path answers the same thing: slabbed vs whole,
             fp16 vs fp32, torch vs numpy, AMP vs fp32.
    colour   what the live window's three colour modes cost to paint.
    vram     how small a card this fits on, by capping the allocator and
             running a frame in a child process until it OOMs.

What is NOT here, because it needs the dataset rather than a frame: the NYU
654-frame scores. Those come from run_inference.py and evaluate.py, and
HARDWARE.md names the three runs that produced them.

    python -m inference.benchmark \
        --pretrained_path ./checkpoint/CleanerS_ckpt.pth
    python -m inference.benchmark --pretrained_path ... --what encode,agree
    python -m inference.benchmark --pretrained_path ... --device cpu
    python -m inference.benchmark --pretrained_path ... --out /tmp/hw.md

Every timing is the MEDIAN of --frames frames after --warmup warm-up frames,
which is what the page quotes; a median rather than a mean because the first
frames after a configuration change are not representative of a steady loop.

The default source is the five depth frames tracked in git, replayed on
--loop. Cost per frame here is independent of the scene -- fixed grid, fixed
model, a distance transform over the whole volume either way -- so cycling
five frames measures what five hundred would. Point --source at
bag:captures/room07/scan/scan.bag for distinct frames.
"""

import argparse
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from .camera import open_source
    from .frame_loader import FrameLoader, VOX_SIZE_HI, load_cuda_encoder
    from . import model as model_mod
    from . import tsdf_torch
    from .run_live import build_mask, load_cfg
    from .visualize import save_prediction_ply
    from .utils import get_device, setup_logging
except ImportError:
    from camera import open_source
    from frame_loader import FrameLoader, VOX_SIZE_HI, load_cuda_encoder
    import model as model_mod
    import tsdf_torch
    from run_live import build_mask, load_cfg
    from visualize import save_prediction_ply
    from utils import get_device, setup_logging

log = logging.getLogger('benchmark')

SECTIONS = ('frame', 'encode', 'agree', 'colour', 'vram')
VRAM_CAPS = (None, 1536, 1280, 1152, 1024, 896, 768)   # MB, allocator cap
# float32 carries ~7 digits, so a TSDF in [-1, 1] rounds at ~1e-7. Anything
# at or below this is the two encoders agreeing, in different arithmetic.
ULP = 1e-6


# ---------------------------------------------------------------- plumbing --

def parse_args():
    p = argparse.ArgumentParser('regenerate document/HARDWARE.md')
    p.add_argument('--cfg', default='./cfgs/NYU/voxelSSC.yaml')
    p.add_argument('--pretrained_path',
                   default='./checkpoint/CleanerS_ckpt.pth')
    p.add_argument('--source', default='folder:captures/room07/depth',
                   help='any run_live source spec; the default is the five '
                        'depth frames tracked in git')
    p.add_argument('--rgb_dir', default=None)
    p.add_argument('--no_loop', dest='loop', action='store_false',
                   help='stop at the end of the source instead of cycling it')
    p.add_argument('--what', default='all',
                   help='comma-separated: %s, or all' % ', '.join(SECTIONS))
    p.add_argument('--frames', type=int, default=20)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--colour_frames', type=int, default=12,
                   help='the colour section is slower per frame and its cost '
                        'varies with how much of the room is predicted')
    p.add_argument('--device', default=None, choices=['cuda', 'cpu'],
                   help='default: cuda when there is one')
    p.add_argument('--reach', type=float, default=0.75,
                   help="live_view's default, for the colour section")
    p.add_argument('--out', default=None, help='write the markdown here too')
    # the vram section re-runs this file in a child with the allocator capped
    p.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--_cap', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--_amp', default='auto', help=argparse.SUPPRESS)
    return p.parse_args()


def md(xs):
    """Median of a list of seconds, in ms."""
    return 1000.0 * float(np.median(xs))


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def collect_frames(args, n):
    """Read n frames into memory up front, so file I/O is not in any timing."""
    src = open_source(args.source, rgb_dir=args.rgb_dir, loop=args.loop,
                      fov='nyu')
    out = []
    try:
        for frame in src.frames():
            out.append(frame)
            if len(out) >= n:
                break
    finally:
        src.close()
    if not out:
        raise SystemExit('no frames from %r' % args.source)
    if len(out) < n:
        log.warning('asked for %d frames, got %d -- pass --loop or a longer '
                    'source', n, len(out))
    return out


def make_loader(frame, encoder, device):
    """The same choice run_live makes: a replayed NYU frame brings its own
    grid placement, a live/captured one gets a floor-anchored grid."""
    if frame.vox_origin is not None:
        return FrameLoader(frame.depth, frame.vox_origin, frame.cam_pose,
                           rgb=frame.rgb, cam_K=frame.cam_K,
                           encoder=encoder, device=device)
    height = frame.camera_height if frame.camera_height is not None else 1.25
    yaw = frame.yaw if frame.yaw is not None else 0.0
    return FrameLoader.from_live_camera(
        frame.depth, rgb=frame.rgb, cam_K=frame.cam_K, camera_height=height,
        yaw=yaw, up_camera=frame.up_camera, encoder=encoder, device=device)


def have_cuda_encoder():
    try:
        return load_cuda_encoder() is not None
    except Exception:
        return False


def table(header, rows, align=None):
    """A markdown table. align is a string of '<' and '>' per column."""
    align = align or '<' * len(header)
    bar = ['---' if a == '<' else '---:' for a in align]
    out = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(bar) + ' |']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


# ------------------------------------------------------------- provenance --

def machine_block(args, device):
    lines = ['## The machine', '']
    gpu = 'none'
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = '%s, %d MB, sm_%d%d' % (p.name, p.total_memory // (1024 ** 2),
                                      p.major, p.minor)
    cpu = platform.processor() or platform.machine()
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    rows = [
        ('GPU', gpu),
        ('CPU', cpu),
        ('torch', torch.__version__),
        ('CUDA runtime', torch.version.cuda or 'n/a'),
        ('arch list', ' '.join(torch.cuda.get_arch_list())
                      if torch.cuda.is_available() else 'n/a'),
        ('device used', str(device)),
        ('source', args.source + (' --loop' if args.loop else '')),
        ('frames / warm-up', '%d / %d' % (args.frames, args.warmup)),
        ('cuda encoder built', 'yes' if have_cuda_encoder() else 'no'),
    ]
    lines.append(table(('what', 'value'), rows))
    return '\n'.join(lines)


# ------------------------------------------------------------ the sections --

def section_frame(args, frames, model, device):
    """The end-to-end table: what a whole frame costs, per configuration."""
    cuda = torch.device(device).type == 'cuda'
    configs = [('numpy', 'off', False), ('torch', 'off', False)]
    if cuda and have_cuda_encoder():
        configs.insert(1, ('cuda', 'off', False))
    if torch.device(device).type == 'cuda':
        configs += [('torch', 'auto', False), ('torch', 'auto', True)]

    tmp = tempfile.mkdtemp(prefix='bench_ply_')
    rows, vram = [], []
    for encoder, amp, ply in configs:
        enc, inf, wr = [], [], []
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for i, frame in enumerate(frames):
            t0 = time.time()
            loader = make_loader(frame, encoder, device)
            sample = loader.build_tsdf_and_mapping()
            sync(device)
            t1 = time.time()
            pred, _ = model_mod.predict(model, sample, device=device, amp=amp)
            sync(device)
            t2 = time.time()
            if ply:
                mask = build_mask(sample, 'surface')
                save_prediction_ply(pred, mask,
                                    os.path.join(tmp, 'bench.ply'))
            t3 = time.time()
            if i < args.warmup:
                continue
            enc.append(t1 - t0)
            inf.append(t2 - t1)
            wr.append(t3 - t2)
        total = md(enc) + md(inf) + md(wr)
        if cuda:
            vram.append(('`%s` / `amp %s`%s'
                         % (encoder, amp, ' + `.ply`' if ply else ''),
                         '%d MB' % (torch.cuda.max_memory_allocated()
                                    // (1024 ** 2))))
        rows.append(('`%s` / `amp %s`%s' % (encoder, amp,
                                            ' + `.ply`' if ply else ''),
                     '%.0f' % md(enc), '%.0f' % md(inf),
                     '%.0f' % md(wr) if ply else '0',
                     '%.0f ms' % total, '%.2f' % (1000.0 / total)))
    try:
        os.remove(os.path.join(tmp, 'bench.ply'))
        os.rmdir(tmp)
    except OSError:
        pass
    out = ('## End to end\n\n'
           + table(('configuration', 'encode', 'infer', 'ply', 'frame',
                    'fps'), rows, '<>>>>>')
           + '\n\nEach row is one configuration of the same loop: build the '
             'TSDF, run the\nforward pass, optionally write a `.ply`. '
             'Columns are medians, so the frame\ncolumn is their sum rather '
             'than a measured total.')
    if vram:
        out += ('\n\nPeak VRAM while doing it, model included:\n\n'
                + table(('configuration', 'allocated'), vram, '<>')
                + '\n\nWhat the tensors need. What the allocator RESERVES on '
                  'top is not a\nper-row number -- it keeps whatever the row '
                  'above it left -- so the\nfigure a card actually has to '
                  'clear is the `vram` section, which runs\neach case in a '
                  'process of its own.')
    return out


def section_encode(args, frames, device):
    """The encode stage on its own, per arithmetic -- and, for torch, per
    slab size and transform dtype, which is where the VRAM goes."""
    cuda = torch.device(device).type == 'cuda'
    variants = [('numpy', 'scipy, float64, CPU', dict(encoder='numpy'))]
    if cuda and have_cuda_encoder():
        variants.append(('cuda', 'datautil.cu kernels',
                         dict(encoder='cuda')))
    variants.append(('torch', 'separable transform', dict(encoder='torch')))

    rows = []
    for name, note, kw in variants:
        stage, peak = [], 0
        for i, frame in enumerate(frames):
            base = 0
            if cuda:
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            loader = make_loader(frame, kw['encoder'], device)
            t0 = time.time()
            loader.build_tsdf_and_mapping()
            sync(device)
            if i >= args.warmup:
                stage.append(time.time() - t0)
                if cuda:
                    peak = max(peak,
                               torch.cuda.max_memory_allocated() - base)
        rows.append(('`%s`' % name, note, '%.0f' % md(stage),
                     '%d MB' % (peak // (1024 ** 2)) if peak > 0 else '-'))
    out = ['## The encode stage', '',
           table(('encoder', 'what it is', 'ms', 'peak VRAM'), rows,
                 '<<>>'),
           '',
           'The stage is the whole of `build_tsdf_and_mapping`: the distance',
           'transform, the projection, the block mean and the two mapping',
           'tables. The mapping tables stay in numpy on every path, and the',
           'peak is what the stage allocates ABOVE whatever is already',
           'resident, so a loaded model does not land in this column.']

    # the torch encoder's own knobs: slab size and transform dtype
    if cuda:
        inner = []
        depth_frames = frames[:max(1, args.warmup + 5)]
        for slab, dtype, label in ((24, torch.float16, 'slab 24, fp16'),
                                   (24, torch.float32, 'slab 24, fp32'),
                                   (240, torch.float16, 'whole, fp16'),
                                   (240, torch.float32, 'whole, fp32')):
            ts, peak = [], 0
            for i, frame in enumerate(depth_frames):
                loader = make_loader(frame, 'torch', device)
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                t0 = time.time()
                tsdf_torch.encode_low(
                    loader.depth, loader.cam_K, loader.cam_pose,
                    loader.vox_origin,
                    drop_invalid_depth=loader.drop_invalid_depth,
                    device=device, slab=slab, edt_dtype=dtype)
                sync(device)
                if i >= args.warmup:
                    ts.append(time.time() - t0)
                    peak = max(peak,
                               torch.cuda.max_memory_allocated() - base)
            inner.append((label + (' *(the default)*' if slab == 24 and
                                   dtype == torch.float16 else ''),
                          '%.0f' % md(ts),
                          '%d MB' % (peak // (1024 ** 2))))
        out += ['', 'Inside the torch encoder, timed without the mapping '
                    'tables:', '',
                table(('configuration', 'ms', 'peak VRAM'), inner, '<>>'),
                '',
                'fp16 is exact here, not a trade: every value in a truncated',
                'squared-distance field is a whole number <= 3 * 12^2 = 432, '
                'and fp16',
                'holds every integer to 2048. Slabbing costs no time and is '
                'what makes',
                'the encoder fit beside the model on a small card.']
    return '\n'.join(out)


def section_agree(args, frames, model, device):
    """Does the fast path answer the same thing? Compare the TSDF each
    configuration builds, and the labels AMP predicts."""
    cuda = torch.device(device).type == 'cuda'
    use = frames[:min(len(frames), max(3, args.frames // 4))]
    rows = []

    def tsdf_of(encoder, frame):
        return np.asarray(make_loader(frame, encoder, device)
                          .build_tsdf_and_mapping()['tsdf'], np.float64)

    def compare(pairs):
        """-> (any, real, worst, total). `any` counts a difference in the
        last bit of a float32, which float32-vs-float64 arithmetic produces
        in bulk and which means nothing; `real` counts the ones bigger than
        that, which is the number worth quoting."""
        anyd, real, worst, total = 0, 0, 0.0, 0
        for a, b in pairs:
            d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
            anyd += int((d > 0).sum())
            real += int((d > ULP).sum())
            worst = max(worst, float(d.max()))
            total += d.size
        return anyd, real, worst, total

    anyd, real, worst, total = compare(
        (tsdf_of('numpy', f), tsdf_of('torch', f)) for f in use)
    rows.append(('torch vs numpy TSDF', '{:,}'.format(total),
                 '%d' % anyd, '%d' % real, '%.3g' % worst))

    if cuda:
        for slab, dtype, label in ((240, torch.float16, 'whole vs slabbed'),
                                   (24, torch.float32, 'fp32 vs fp16')):
            pairs = []
            for frame in use:
                loader = make_loader(frame, 'torch', device)
                ref, _ = tsdf_torch.encode_low(
                    loader.depth, loader.cam_K, loader.cam_pose,
                    loader.vox_origin,
                    drop_invalid_depth=loader.drop_invalid_depth,
                    device=device)
                alt, _ = tsdf_torch.encode_low(
                    loader.depth, loader.cam_K, loader.cam_pose,
                    loader.vox_origin,
                    drop_invalid_depth=loader.drop_invalid_depth,
                    device=device, slab=slab, edt_dtype=dtype)
                pairs.append((ref, alt))
            anyd, real, worst, total = compare(pairs)
            rows.append((label, '{:,}'.format(total), '%d' % anyd,
                         '%d' % real, '%.3g' % worst))

        # the forward pass needs the colour frame; a depth-only source
        # (nyu: without --rgb_dir) can still do the encoder rows above
        with_rgb = [f for f in use if f.rgb is not None]
        if with_rgb:
            diff, total = 0, 0
            for frame in with_rgb:
                sample = make_loader(frame, 'torch',
                                     device).build_tsdf_and_mapping()
                ref, _ = model_mod.predict(model, sample, device=device,
                                           amp='off')
                alt, _ = model_mod.predict(model, sample, device=device,
                                           amp='on')
                diff += int((ref != alt).sum())
                total += ref.size
            rows.append(('AMP vs fp32 labels', '{:,}'.format(total),
                         '%d' % diff, '%d' % diff,
                         '%.2f%%' % (100.0 * diff / total)))
        else:
            rows.append(('AMP vs fp32 labels', '-', '-', '-',
                         'no colour frame'))

    return ('## Does it change the answer?\n\n'
            + table(('comparison', 'voxels', 'any diff', '> %g' % ULP,
                     'worst'), rows, '<>>>>')
            + '\n\nOver %d frames. `any diff` counts a difference in the last '
              'bit of a\nfloat32, which float32-vs-float64 arithmetic '
              'produces in bulk and which\nmeans nothing; the `> %g` column '
              'is the one worth quoting. Zero in both\nis bit-identical. The '
              'last row is the only change that reaches the\noutput, and the '
              'NYU 654-frame scores in HARDWARE.md say what it does to\na '
              'score: nothing above the fourth decimal.' % (len(use), ULP))


def section_colour(args, frames, model, device):
    """What the live window pays to paint a frame, per --colour mode."""
    try:
        from . import display_solid as ds
        from .live_view import occupied
    except ImportError:
        import display_solid as ds
        from live_view import occupied

    use = frames[:min(len(frames), args.colour_frames + args.warmup)]
    paints = {m: [] for m in ('class', 'zbuffer', 'all')}
    builds, counts = [], []
    for i, frame in enumerate(use):
        if frame.rgb is None:          # nothing to paint with, and no forward
            continue
        loader = make_loader(frame, 'auto' if torch.cuda.is_available()
                             else 'numpy', device)
        sample = loader.build_tsdf_and_mapping()
        pred, _ = model_mod.predict(model, sample, device=device, amp='auto')
        mask = build_mask(sample, 'surface')
        idx, cls = occupied(pred, mask)
        if len(idx) == 0:
            continue
        class_rgb = ds.DEFAULT_COLORMAP[cls]
        image = np.ascontiguousarray(
            np.asarray(frame.rgb)[:, :, ::-1]).astype(np.uint8) \
            if frame.rgb is not None else None
        for mode in ('class', 'zbuffer', 'all'):
            t0 = time.time()
            if mode == 'class' or image is None:
                _ = class_rgb
            else:
                photo, visible, _, _ = ds.photo_colours(
                    idx, loader.vox_origin, loader.cam_pose, loader.cam_K,
                    image, None if mode == 'zbuffer' else loader.depth,
                    mode=mode, reach=args.reach)
                _, _ = ds.fill_hidden(photo, visible, class_rgb,
                                      ds.to_view_index(idx),
                                      'nearest' if mode == 'all' else 'class')
            if i >= args.warmup:
                paints[mode].append(time.time() - t0)
        t0 = time.time()
        ds.build_cube_mesh(ds.to_view_index(idx), class_rgb)
        if i >= args.warmup:
            builds.append(time.time() - t0)
            counts.append(len(idx))

    rows = [('`class`', 'the semantic colour', '%.0f' % md(paints['class'])),
            ('`zbuffer`', 'the photo pixel each voxel is front-most on',
             '%.0f' % md(paints['zbuffer'])),
            ('`all`', 'the three photo rules stacked',
             '%.0f' % md(paints['all']))]
    return ('## The live window\n\n'
            + table(('`--colour`', 'what it paints with', '+ms'), rows, '<<>')
            + '\n\nPaint only, which is what varies between the modes. The '
              'mesh on top of it\nis %.0f ms for a median %d voxels, and the '
              'window redraw a further few.\nA run with `--display` prints '
              'the same split on exit (`display: ... colour\n... mesh+draw '
              '...`).' % (md(builds), int(np.median(counts))))


def section_vram(args):
    """How small a card this fits on. Runs one frame in a child process with
    the allocator capped, because an OOM is not survivable in-process."""
    if not torch.cuda.is_available():
        return '## Hardware gates\n\nNo CUDA device: nothing to cap.'
    total_mb = (torch.cuda.get_device_properties(0).total_memory
                // (1024 ** 2))
    held_mb = torch.cuda.memory_reserved() // (1024 ** 2)
    if held_mb > 64:
        log.warning('this process still holds %d MB of the card; the capped '
                    'children have that much less to fit in', held_mb)
    rows = []
    for cap in VRAM_CAPS:
        cells = []
        for amp in ('auto', 'off'):
            cmd = [sys.executable, '-m', 'inference.benchmark', '--_child',
                   '--cfg', args.cfg,
                   '--pretrained_path', args.pretrained_path,
                   '--source', args.source, '--_amp', amp]
            if cap is not None:
                cmd += ['--_cap', str(cap)]
            if not args.loop:
                cmd.append('--no_loop')
            env = dict(os.environ, PYTHONPATH=_REPO_ROOT)
            r = subprocess.run(cmd, cwd=_REPO_ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = r.stdout.decode('utf-8', 'replace').strip().splitlines()
            last = out[-1].strip() if out else ''
            if last == 'OOM' or last.startswith('OK'):
                cells.append(last)
            elif 'out of memory' in r.stderr.decode('utf-8', 'replace'):
                cells.append('OOM')          # died before it could say so
            else:
                cells.append('error')
                log.warning('cap %s amp %s: %s', cap, amp,
                            r.stderr.decode('utf-8', 'replace')[-400:])
        rows.append(('uncapped (%d MB)' % total_mb if cap is None
                     else '%d MB' % cap, cells[0], cells[1]))
    return ('## Hardware gates\n\n'
            + table(('usable VRAM', 'AMP', 'fp32'), rows, '><<')
            + '\n\nOne frame per cell, the allocator held to that much by\n'
              '`set_per_process_memory_fraction` in a child process -- an OOM '
              'is not\nsurvivable in the process that hits it. A card that '
              'only just clears the\nfloor pays for it in time: the allocator '
              'can cache less and churns.%s'
            % ('\n\n**This run is not trustworthy:** the parent process still '
               'held %d MB of\nthe card, so the children had that much less '
               'to fit in. Run `--what vram`\non its own.' % held_mb
               if held_mb > 64 else ''))


def run_capped_child(args):
    """The child of section_vram: one line, 'OK <ms>' or 'OOM'."""
    device = get_device()
    if args._cap is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args._cap * 1024.0 * 1024.0 / total))
    try:
        cfg = load_cfg(args, [])
        model = model_mod.load_model(cfg, device=device)
        # a cold first frame pays for cudnn's own setup, which is not what
        # the time column is about: warm up, then take the median
        frames = collect_frames(args, 4)
        ts = []
        for i, frame in enumerate(frames):
            t0 = time.time()
            loader = make_loader(frame, 'auto', device)
            sample = loader.build_tsdf_and_mapping()
            model_mod.predict(model, sample, device=device, amp=args._amp)
            sync(device)
            if i > 0:
                ts.append(time.time() - t0)
        print('OK %.0f ms' % md(ts))
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('OOM')
        else:
            raise
    return 0


# -------------------------------------------------------------------- main --

def main():
    args = parse_args()
    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)

    if args._child:
        return run_capped_child(args)

    want = SECTIONS if args.what == 'all' else tuple(
        w.strip() for w in args.what.split(','))
    for w in want:
        if w not in SECTIONS:
            raise SystemExit('unknown section %r; pick from %s'
                             % (w, ', '.join(SECTIONS)))

    device = torch.device(args.device) if args.device else get_device()
    frames = collect_frames(args, args.frames + args.warmup)

    model = None
    if {'frame', 'agree', 'colour'} & set(want):
        cfg = load_cfg(args, [])
        model = model_mod.load_model(cfg, device=device)

    blocks = ['# Hardware, measured', '',
              'Generated by `python -m inference.benchmark`, %s.'
              % time.strftime('%Y-%m-%d'), '',
              machine_block(args, device)]
    for w in want:
        log.warning('running section %r ...', w)
        if w == 'frame':
            blocks.append(section_frame(args, frames, model, device))
        elif w == 'encode':
            blocks.append(section_encode(args, frames, device))
        elif w == 'agree':
            blocks.append(section_agree(args, frames, model, device))
        elif w == 'colour':
            blocks.append(section_colour(args, frames, model, device))
        elif w == 'vram':
            # the children need the card. A parent still holding the model,
            # and the allocator still holding every block it cached for the
            # sections above, is enough to OOM every one of them.
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            blocks.append(section_vram(args))

    text = '\n\n'.join(blocks) + '\n'
    sys.stdout.write(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text)
        log.warning('written to %s', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
