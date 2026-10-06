"""
reconstruction_GT/bag_reader.py

Read a recorded .bag frame by frame, deterministically.

    for bf in iter_bag('captures/room08/scan/scan.bag'):
        bf.index, bf.depth, bf.depth_raw, bf.rgb, bf.K, bf.depth_scale
    bf = read_frame('captures/room08/scan/scan.bag', 445)
    md = bag_metadata('captures/room08/scan/scan.bag')

*** WHY THIS EXISTS ***

`o3d.t.io.RSBagReader` is a PLAYBACK device: it runs the bag at its recorded
frame rate and drops frames when the consumer is slower than the recording.
Measured on room08's bag -- two reads of "frame 150" in the same process:

    read 0: 368598 valid depth pixels
    read 1: 338168 valid depth pixels        two reads identical: False

So "frame i" is not a property of the file, it is a property of how fast the
loop that read it happened to run. A loop that only decodes frames lands on one
alignment; a loop that runs a network between frames lands on another, and
every frame index it reports is then paired with the wrong pose.

librealsense has the switch for this -- `playback.set_real_time(False)` makes
the pipeline deliver every frame, waiting for the consumer instead of dropping.
That is what this module uses, so an index means the same frame every time, at
any speed, which is what pairing frames with `trajectory.txt` requires.

Verified on room08: two full passes agree bit-exactly on every frame, and a
slow pass (0.2 s of work per frame) agrees with a fast one.
"""

import logging
import collections

import numpy as np

log = logging.getLogger(__name__)

BagFrame = collections.namedtuple(
    'BagFrame', 'index depth depth_raw rgb K depth_scale K_color color_from_depth')
BagFrame.__new__.__defaults__ = (None, None)        # unset unless align=False
BagFrame.__doc__ = '''One frame of a bag.

index       frames delivered before it, counting from 0 -- the same counting
            fuse_scan.py writes into trajectory.txt
depth       metres, float32            depth_raw  the sensor's own uint16
rgb         HxWx3 uint8; registered to the depth camera unless align=False,
            in which case it is the colour camera's own untouched frame
K           3x3 depth intrinsics       depth_scale  metres per raw unit
K_color, color_from_depth   set only when align=False: the colour camera's
            intrinsics and the 4x4 that takes a point from the depth camera's
            frame into the colour camera's. Together they let a consumer
            register the two per point instead of resampling either image
            (inference/frame_loader.py, _build_mapping2d).
'''


def _start(bag_path, want_color=True, align='depth'):
    """-> (pipeline, profile, K, depth_scale, align). Playback, never real
    time, so every frame is delivered however slow the consumer is."""
    import pyrealsense2 as rs
    pipe = rs.pipeline(rs.context())
    cfg = rs.config()
    cfg.enable_device_from_file(bag_path, repeat_playback=False)
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)
    stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = stream.get_intrinsics()
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], np.float64)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    extras = {}
    if want_color:
        cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ci = cs.get_intrinsics()
        extras['K_color'] = np.array([[ci.fx, 0.0, ci.ppx],
                                      [0.0, ci.fy, ci.ppy],
                                      [0.0, 0.0, 1.0]], np.float64)
        e = profile.get_stream(rs.stream.depth).get_extrinsics_to(cs)
        T = np.eye(4)
        T[:3, :3] = np.asarray(e.rotation, np.float64).reshape(3, 3).T
        T[:3, 3] = np.asarray(e.translation, np.float64)
        extras['color_from_depth'] = T
    al = None
    if want_color and align:
        target = rs.stream.color if align == 'color' else rs.stream.depth
        al = rs.align(target)
        if align == 'color':
            # the depth now lives in the colour camera's frame, so the colour
            # intrinsics are the ones that unproject it -- this is NYU's
            # arrangement, and inference/camera.py's
            K = extras['K_color']
    return pipe, profile, K, scale, al, extras


def bag_metadata(bag_path):
    """What the header says, without reading the frames."""
    import pyrealsense2 as rs
    pipe, profile, K, scale, _, _ = _start(bag_path, want_color=False)
    try:
        stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        dev = profile.get_device()
        return {
            'width': stream.width(), 'height': stream.height(),
            'fps': float(stream.fps()),
            # two conventions, and mixing them silently fuses nothing:
            # librealsense gives metres per raw unit (0.001), Open3D's
            # integrate()/track_frame_to_model() want raw units per metre.
            'depth_scale': float(scale),
            'depth_scale_o3d': 1.0 / float(scale),
            'K': K,
            'device_name': dev.get_info(rs.camera_info.name)
            if dev.supports(rs.camera_info.name) else 'unknown',
            'duration_s': float(dev.as_playback().get_duration().total_seconds()),
        }
    finally:
        pipe.stop()


def iter_bag(bag_path, want_color=True, align='depth'):
    """Yield a BagFrame per frame, in order, every time.

    align='depth'  warps the colour into the depth camera: a colour for every
                   depth pixel, which is what fusion wants, at the cost of the
                   colour image wherever the depth is missing.
    align='color'  warps the DEPTH into the colour camera and leaves the colour
                   untouched. This is what NYU is -- its depth was projected
                   into the RGB frame, which is why SSCNet and CleanerS
                   unproject it with the RGB intrinsics -- and what
                   inference/camera.py does for the still frame. `K` comes back
                   as the colour intrinsics, because that is the frame the
                   depth now lives in.
    align=False    neither is touched; K_color and color_from_depth come back
                   so the caller can relate them per point instead.
    """
    pipe, _, K, scale, aligner, extras = _start(bag_path, want_color, align)
    try:
        i = -1
        while True:
            ok, frames = pipe.try_wait_for_frames(5000)
            if not ok:
                break
            i += 1
            if aligner is not None:
                frames = aligner.process(frames)
            d = frames.get_depth_frame()
            c = frames.get_color_frame() if want_color else None
            if not d or (want_color and not c):
                continue
            raw = np.asanyarray(d.get_data())
            yield BagFrame(i, raw.astype(np.float32) * scale, raw,
                           np.asanyarray(c.get_data()) if c else None,
                           K, float(scale), extras.get('K_color'),
                           extras.get('color_from_depth'))
    finally:
        try:
            pipe.stop()
        except RuntimeError:
            pass


def read_frame(bag_path, index, want_color=True, align='depth'):
    """-> the BagFrame at `index`."""
    for bf in iter_bag(bag_path, want_color, align):
        if bf.index == index:
            return bf
    raise SystemExit('frame %d is past the end of %s' % (index, bag_path))


def main():
    """Self-check: two passes must agree, fast and slow."""
    import sys
    import time
    from inference.utils import setup_logging
    setup_logging()
    bag = sys.argv[1] if len(sys.argv) > 1 else 'captures/room08/scan/scan.bag'
    every = 100

    def digest(delay=0.0):
        out = {}
        for bf in iter_bag(bag):
            if bf.index % every == 0:
                out[bf.index] = (float(bf.depth.sum()), int((bf.depth > 0).sum()))
                if delay:
                    time.sleep(delay)
        return out

    t = time.time()
    a = digest()
    log.info('pass 1: %d frames sampled in %.0f s', len(a), time.time() - t)
    b = digest(0.2)                      # a slow consumer: the real test
    same = all(a[k] == b.get(k) for k in a) and len(a) == len(b)
    print('\nfast pass and slow pass agree on every sampled frame: %s' % same)
    if not same:
        for k in sorted(a):
            if a[k] != b.get(k):
                print('  frame %d: %s vs %s' % (k, a[k], b.get(k)))
    return 0 if same else 1


if __name__ == '__main__':
    raise SystemExit(main())
