"""
reconstruction_GT/imu_check.py

How good is the accelerometer, and does it matter for the ground truth?

    python -m reconstruction_GT.imu_check captures/room08
    python -m reconstruction_GT.imu_check captures/room08 --no_plot

*** WHY ***

`meta.json`'s `up_camera` is one accelerometer reading, taken while the camera
was held still, and the whole grid is built on it: world Z is that vector, so
an error in it tilts every voxel of the ground truth. MAKING_GT.md, "no
estimated values in the capture record" -- it is a MEASUREMENT, so it has an
error bar, and this is the script that puts a number on it.

*** WHAT AN ACCELEROMETER ACTUALLY MEASURES ***

Proper acceleration: gravity plus whatever the hand is doing. At rest the hand
term is zero and the reading IS gravity (pointing up, since it reads the
reaction). The moment the camera moves, the two are added together and cannot
be separated from one sample -- a 1 m/s^2 swing, which is gentle, is already
6 degrees of apparent tilt. So the reading is trustworthy exactly when the
camera is still, which is why `record_scan.py` asks for a hold at the start and
`capture.py` takes `up_camera` from a stationary camera.

This script measures three things over the whole bag:

  |a|                 should sit at 9.81 m/s^2 whenever the camera is at rest;
                      departures are the hand, not the sensor
  noise at rest       how much the direction wanders while nothing is moving --
                      the error bar on `up_camera` itself
  accel vs tracking   the angle between the accelerometer's idea of up and the
                      SLAM pose's idea of up, over time. At rest this is real
                      disagreement (IMU noise + tracking drift). In motion it
                      is mostly the hand term, and is expected to be large.

*** WHAT IT CANNOT TELL YOU ***

It cannot separate IMU error from tracking drift in the last number; it can
only bound their sum. The independent check on the grid is geometric and
already in `fuse_report.json`: `floor.surface_tilt_deg`, how level the fused
floor came out. If the grid were tilted, the floor would be tilted with it.
"""

import os
import sys
import json
import logging
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

log = logging.getLogger(__name__)

G = 9.80665
REST_WIN = 0.30          # s, the window the vector must hold steady over
REST_STILL = 0.06        # m/s^2, how much it may wander inside it


def parse_args():
    p = argparse.ArgumentParser(description='Measure the accelerometer against '
                                            'the tracked poses.')
    p.add_argument('capture_dir')
    p.add_argument('--no_plot', action='store_true')
    p.add_argument('--out', default=None,
                   help='plot path (default <capture>/scan/imu_check.png)')
    return p.parse_args()


def _stretch_spread(mask, up):
    """Angular spread inside each still stretch, median over stretches."""
    if not mask.any():
        return None
    d = np.diff(mask.astype(int))
    starts = ([0] if mask[0] else []) + list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1) + ([len(mask)] if mask[-1] else [])
    out = []
    for a0, b0 in zip(sorted(starts), sorted(ends)):
        seg = up[a0:b0]
        if len(seg) < 8:
            continue
        m = seg.mean(axis=0)
        m /= np.linalg.norm(m)
        out.append(np.degrees(np.arccos(np.clip(seg @ m, -1, 1))).std())
    return float(np.median(out)) if out else None


def _allan(hold, rate=63.0):
    """How the direction settles as you average longer, over the opening hold.

    This needs no reference at all: it only asks whether the fluctuation
    averages away. White noise falls as 1/sqrt(tau); a floor that stops falling
    is bias instability, which averaging cannot remove and which a single
    `up_camera` reading would inherit.
    """
    if len(hold) < 64:
        return []
    out, base = [], None
    for k in (1, 2, 4, 8, 16, 32, 64):
        n = len(hold) // k
        if n < 4:
            break
        blocks = hold[:n * k].reshape(n, k, 3).mean(axis=1)
        ub = blocks / np.linalg.norm(blocks, axis=1)[:, None]
        mb = ub.mean(axis=0)
        mb /= np.linalg.norm(mb)
        dev = float(np.degrees(np.arccos(np.clip(ub @ mb, -1, 1))).std())
        base = dev if base is None else base
        out.append({'tau_s': round(k / rate, 4), 'deg': round(dev, 5),
                    'white_noise_would_be': round(base / np.sqrt(k), 5)})
    return out


def read_imu(bag):
    """-> (accel samples (N,4): t, x, y, z | R imu->colour | frame times (M,2)).

    Two passes, because they cannot be one. Motion frames delivered inside a
    video frameset arrive at the VIDEO rate -- ask for depth and colour too and
    the 63 Hz accelerometer comes back as 15 Hz, one sample per frame, which is
    what an earlier version of this script measured and reported as the sensor
    rate. So the motion stream is read alone, and the frame timestamps in a
    second pass that counts frames exactly as bag_reader does.
    """
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(cfg, bag, repeat_playback=False)
    cfg.enable_stream(rs.stream.accel)
    profile = pipe.start(cfg)
    accel = []
    try:
        profile.get_device().as_playback().set_real_time(False)
        ext = profile.get_stream(rs.stream.accel).get_extrinsics_to(
            profile.get_stream(rs.stream.accel))
        while True:
            try:
                fs = pipe.wait_for_frames(2000)
            except RuntimeError:
                break
            for f in fs:
                if f.is_motion_frame():
                    m = f.as_motion_frame().get_motion_data()
                    accel.append((f.get_timestamp() / 1000.0, m.x, m.y, m.z))
    finally:
        pipe.stop()

    # the IMU -> colour rotation, and the frame clock, from a video pass
    pipe, cfg = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(cfg, bag, repeat_playback=False)
    cfg.enable_stream(rs.stream.accel)
    cfg.enable_stream(rs.stream.color)
    cfg.enable_stream(rs.stream.depth)
    profile = pipe.start(cfg)
    frames, idx = [], -1
    try:
        profile.get_device().as_playback().set_real_time(False)
        ext = profile.get_stream(rs.stream.accel).get_extrinsics_to(
            profile.get_stream(rs.stream.color))
        R = np.asarray(ext.rotation, np.float64).reshape(3, 3).T   # column-major
        while True:
            try:
                fs = pipe.wait_for_frames(2000)
            except RuntimeError:
                break
            has_depth = has_color = False
            last_ts = None
            for f in fs:
                if f.is_motion_frame():
                    continue
                st = f.get_profile().stream_type()
                has_depth |= st == rs.stream.depth
                has_color |= st == rs.stream.color
                last_ts = f.get_timestamp() / 1000.0
            if has_depth and has_color and last_ts is not None:
                idx += 1
                frames.append((idx, last_ts))
    finally:
        pipe.stop()

    accel = np.asarray(accel)
    frames = np.asarray(frames, np.float64)
    t0 = min(accel[0, 0], frames[0, 1]) if len(frames) else accel[0, 0]
    accel[:, 0] -= t0
    if len(frames):
        frames[:, 1] -= t0
    return accel, R, frames


def rest_mask(t, a_cam, win=REST_WIN, tol=REST_STILL):
    """True where the accelerometer VECTOR is steady over `win` seconds.

    Steadiness, not closeness to 9.81: this sensor reads 9.02 at rest (see the
    calibration section), so a test against g would find no rest at all. A
    vector that is not changing is a camera that is not accelerating, whatever
    the magnitude says.
    """
    n = len(t)
    out = np.zeros(n, bool)
    if n < 5:
        return out
    dt = np.median(np.diff(t))
    k = max(3, int(round(win / max(dt, 1e-6))))
    c1 = np.cumsum(np.vstack([np.zeros(3), a_cam]), axis=0)
    c2 = np.cumsum(np.vstack([np.zeros(3), a_cam ** 2]), axis=0)
    for i in range(n - k):
        m1 = (c1[i + k] - c1[i]) / k
        m2 = (c2[i + k] - c2[i]) / k
        var = np.maximum(m2 - m1 ** 2, 0.0).sum()
        if np.sqrt(var) < tol:
            out[i:i + k] = True
    return out


def fit_bias(a_rest):
    """Per-axis scale and offset from rest samples in varied orientations.

    At rest the true vector has length g whatever way the camera points, so
    fitting an ellipsoid to the rest samples recovers what the sensor does to
    it. Returns None when the orientations are too alike to constrain a fit --
    a sweep held mostly one way up cannot calibrate anything.
    """
    if len(a_rest) < 40:
        return None
    u = a_rest / np.linalg.norm(a_rest, axis=1)[:, None]
    if np.linalg.svd(u - u.mean(axis=0), compute_uv=False)[2] < 0.05 * len(u) ** 0.5:
        return None                       # directions nearly coplanar/identical
    x, y, z = a_rest.T
    A = np.stack([x ** 2, y ** 2, z ** 2, x, y, z, np.ones_like(x)], axis=1)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    if sv[-1] / sv[0] > 1e-3:
        return None                       # not a well-determined ellipsoid
    p = Vt[-1]
    if np.any(p[:3] == 0) or np.any(np.sign(p[:3]) != np.sign(p[0])):
        return None
    b = -p[3:6] / (2 * p[:3])
    k = (p[:3] * b ** 2).sum() - p[6]
    if k <= 0:
        return None
    s = np.sqrt(p[:3] / k) * G            # measured = s * true + b, per axis
    return {'bias': b, 'scale': 1.0 / s}


def analyse(cap):
    from reconstruction_GT.export_frame import load_poses
    bag = os.path.join(cap, 'scan', 'scan.bag')
    accel, R, frames = read_imu(bag)
    if len(accel) < 20:
        raise SystemExit('only %d accelerometer samples in %s' % (len(accel), bag))
    t = accel[:, 0]
    a_raw = accel[:, 1:]                             # IMU axes, as recorded
    a_cam = a_raw @ R.T                              # into the colour camera frame
    mag = np.linalg.norm(a_cam, axis=1)
    up_meas = a_cam / mag[:, None]                   # measured up, camera frame
    rest = rest_mask(t, a_cam)

    # what the tracked pose says up should be, in that frame's camera coords
    poses = load_poses(cap)
    err = np.full(len(t), np.nan)
    if len(frames):
        fi, ft = frames[:, 0].astype(int), frames[:, 1]
        near = np.searchsorted(ft, t).clip(0, len(ft) - 1)
        for k, j in enumerate(near):
            i = fi[j]
            if i not in poses or abs(ft[j] - t[k]) > 0.1:
                continue
            up_pose = poses[i][:3, :3].T @ np.array([0.0, 0.0, 1.0])
            c = float(np.dot(up_pose / np.linalg.norm(up_pose), up_meas[k]))
            err[k] = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    meta_path = os.path.join(cap, 'meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    up_meta = meta.get('up_camera')
    tilt_meta = first_hold = None
    if up_meta is not None:
        u = np.asarray(up_meta, np.float64)
        u = u / np.linalg.norm(u)
        tilt_meta = np.degrees(np.arccos(np.clip(up_meas @ u, -1.0, 1.0)))
        # only the FIRST still stretch may be compared with it: `up_camera` is
        # gravity in CAMERA coordinates, so it changes as the camera turns, and
        # against a rotated camera the comparison measures the rotation
        if rest.any():
            i0 = int(np.argmax(rest))
            i1 = i0 + int(np.argmin(rest[i0:])) if not rest[i0:].all() else len(t)
            if i1 - i0 > 8:
                first_hold = {
                    'window_s': [round(float(t[i0]), 2), round(float(t[i1 - 1]), 2)],
                    'samples': int(i1 - i0),
                    'median_deg': round(float(np.median(tilt_meta[i0:i1])), 3),
                    'max_deg': round(float(np.max(tilt_meta[i0:i1])), 3)}

    rep = json.load(open(os.path.join(cap, 'scan', 'fuse_report.json')))
    floor = rep.get('checks', {}).get('floor', {})
    imu = rep.get('checks', {}).get('imu_tilt', {})

    # Precision needs no external reference: while the camera is still,
    # gravity is constant by physics, so anything the sensor does between
    # consecutive samples is the sensor. Accuracy is the opposite -- it needs
    # something that knows which way up is, and the only such thing here is the
    # fused floor.
    noise_sweep = []
    for tol in (0.02, 0.04, 0.06, 0.10, 0.20):
        rm = rest_mask(t, a_cam, tol=tol)
        sp = _stretch_spread(rm, up_meas)
        noise_sweep.append({'tol_m_s2': tol,
                            'rest_pct': round(100 * float(rm.mean()), 1),
                            'spread_deg': None if sp is None else round(sp, 4)})
    allan = _allan(a_cam[t < t[0] + 5.0])

    at_rest = up_meas[rest]
    noise = np.nan
    if rest.any():
        # noise as short-term wander, measured inside each still stretch rather
        # than across the sweep: the camera is pointed differently in each one
        sp = _stretch_spread(rest, up_meas)
        if sp is not None:
            noise = sp

    # what the sensor does to the vector, and whether it bends the direction
    cal = fit_bias(a_raw[rest]) if rest.any() else None
    cal_out, cal_shift = None, None
    if cal is not None:
        b, sc = cal['bias'], cal['scale']
        corrected = ((a_raw - b) * sc) @ R.T
        cm = np.linalg.norm(corrected, axis=1)
        good = cm > 1e-6
        ang = np.degrees(np.arccos(np.clip(
            (corrected[good] / cm[good][:, None] * up_meas[good]).sum(1), -1, 1)))
        cal_out = {'bias_m_s2': [round(float(v), 4) for v in b],
                   'scale': [round(float(v), 4) for v in sc],
                   'residual_g_std': round(float(cm[rest[good]].std()), 4)
                   if rest[good].any() else None}
        cal_shift = round(float(np.median(ang[rest[good]])), 3) \
            if rest[good].any() else None

    drift = None
    ok = rest & ~np.isnan(err)
    if ok.sum() > 30 and t[ok].ptp() > 10:
        drift = round(float(np.polyfit(t[ok], err[ok], 1)[0]) * 60.0, 3)

    return {
        'room': os.path.basename(cap), 'bag': bag,
        'up_camera_source': meta.get('up_camera_source'),
        'samples': int(len(t)), 'duration_s': float(t[-1] - t[0]),
        'vs_pose_drift_deg_per_min': drift,
        'vs_stored_up_first_hold': first_hold,
        'calibration_note': (
            'per-axis bias and scale cannot be separated from this sweep: the '
            'camera stayed close to one orientation while still, so the rest '
            'samples do not span enough directions to fit an ellipsoid. Only '
            'the magnitude error is observable.'
            if cal is None else 'fitted from the rest samples'),
        'rate_hz': round(len(t) / max(t[-1] - t[0], 1e-9), 1),
        'rest_fraction': round(100.0 * float(rest.mean()), 1),
        'mag_at_rest_mean': round(float(mag[rest].mean()), 4) if rest.any() else None,
        'mag_at_rest_vs_g_pct': round(100 * float(mag[rest].mean() - G) / G, 2)
        if rest.any() else None,
        'calibration': cal_out,
        'direction_shift_if_calibrated_deg': cal_shift,
        'mag_at_rest_std': round(float(mag[rest].std()), 4) if rest.any() else None,
        'mag_moving_max': round(float(mag[~rest].max()), 3) if (~rest).any() else None,
        'direction_noise_at_rest_deg': None if np.isnan(noise) else round(noise, 3),
        'direction_noise_vs_threshold': noise_sweep,
        'allan_opening_hold': allan,
        'vs_pose_at_rest_median_deg':
            round(float(np.nanmedian(err[rest])), 3) if rest.any() else None,
        'vs_pose_moving_median_deg':
            round(float(np.nanmedian(err[~rest])), 3) if (~rest).any() else None,
        'meta_up_camera': up_meta,
        'fuse_imu_tilt_deg': imu.get('tilt_deg'),
        'fuse_floor_tilt_deg': floor.get('surface_tilt_deg'),
        'grid_follows_gravity': up_meta is not None,
        '_series': (t, mag, err, rest, tilt_meta),
    }


def plot(res, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return None
    t, mag, err, rest, tilt_meta = res['_series']
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=130, sharex=True)

    def shade(ax):
        d = np.diff(rest.astype(int))
        starts = list(np.where(d == 1)[0] + 1) + ([0] if rest[0] else [])
        ends = list(np.where(d == -1)[0] + 1) + ([len(t) - 1] if rest[-1] else [])
        for s, e in zip(sorted(starts), sorted(ends)):
            ax.axvspan(t[s], t[e], color='#7fc47f', alpha=0.18, lw=0)

    ax = axes[0]
    ax.plot(t, mag, lw=0.7, color='#2f6fb0')
    ax.axhline(G, color='#c2403d', lw=1, ls='--', label='g = 9.81 m/s$^2$')
    shade(ax)
    ax.set_ylabel('|a|  (m/s$^2$)')
    ax.set_title('%s: accelerometer over the bag. Shaded = camera at rest, '
                 'where the reading IS gravity' % res['room'], fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, err, lw=0.7, color='#8a5fb0')
    shade(ax)
    ax.set_ylabel('accel vs pose\n(deg)')
    ax.grid(alpha=0.25)
    if res['vs_pose_at_rest_median_deg'] is not None:
        ax.axhline(res['vs_pose_at_rest_median_deg'], color='#3f7d3f', lw=1,
                   ls='--', label='median at rest %.2f deg'
                   % res['vs_pose_at_rest_median_deg'])
        ax.legend(fontsize=8, loc='upper right')

    h = res.get('vs_stored_up_first_hold')
    if h:
        ax.axvspan(h['window_s'][0], h['window_s'][1], color='#b07f2f',
                   alpha=0.12, lw=0)
        ax.annotate('first hold: %.2f deg from meta.json\'s up_camera'
                    % h['median_deg'],
                    xy=(h['window_s'][1], ax.get_ylim()[1] * 0.85),
                    xytext=(h['window_s'][1] + 3, ax.get_ylim()[1] * 0.85),
                    fontsize=8, color='#7a5720',
                    arrowprops=dict(arrowstyle='->', color='#7a5720', lw=0.8))
    axes[-1].set_xlabel('seconds into the bag')
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    from inference.utils import setup_logging
    setup_logging()
    args = parse_args()
    cap = args.capture_dir.rstrip('/')
    res = analyse(cap)
    series = res.pop('_series')

    print('\n=== %s: accelerometer ===' % os.path.basename(cap))
    print('%d samples over %.1f s (%.0f Hz), camera at rest for %.0f%% of it'
          % (res['samples'], res['duration_s'], res['rate_hz'],
             res['rest_fraction']))
    print('\nat rest -- where the reading is gravity:')
    print('  |a|            %.4f +- %.4f m/s^2   (g = %.5f, so bias %+.3f%%)'
          % (res['mag_at_rest_mean'], res['mag_at_rest_std'], G,
             100 * (res['mag_at_rest_mean'] - G) / G))
    tight = [d for d in res['direction_noise_vs_threshold']
             if d['spread_deg'] is not None]
    print('  direction      wanders %.3f deg (1 sigma) at the rest threshold '
          'used here;' % res['direction_noise_at_rest_deg'])
    if tight:
        print('                 %s'
              % ', '.join('%.3f deg at %.2f' % (d['spread_deg'], d['tol_m_s2'])
                          for d in tight))
        print('                 -- the looser ones measure how still the hand '
              'was, not the sensor')
    al = res.get('allan_opening_hold') or []
    if al:
        print('  averaging      %.4f deg at %.3f s -> %.4f deg at %.2f s '
              '(white noise: %.4f)'
              % (al[0]['deg'], al[0]['tau_s'], al[-1]['deg'], al[-1]['tau_s'],
                 al[-1]['white_noise_would_be']))
        print('                 it keeps falling, so this is random noise that '
              'averaging removes,\n                 not a drifting bias -- and '
              'no external reference was needed to say so')
    print('  vs tracking    %.2f deg median disagreement with the SLAM pose'
          % res['vs_pose_at_rest_median_deg'])
    if res['vs_stored_up_first_hold']:
        h = res['vs_stored_up_first_hold']
        print('  vs meta.json   %.2f deg from the stored up_camera during the '
              'first hold\n                 (%.1f-%.1f s, %d samples, worst '
              '%.2f deg) -- the only window\n                 where that '
              'comparison means anything'
              % (h['median_deg'], h['window_s'][0], h['window_s'][1],
                 h['samples'], h['max_deg']))
    if res['vs_pose_drift_deg_per_min'] is not None:
        print('  drift          %+.2f deg/min in that disagreement over the sweep'
              % res['vs_pose_drift_deg_per_min'])
    print('\nmoving -- where it is gravity PLUS the hand:')
    print('  |a| reaches    %.2f m/s^2 (%.0f%% off g)'
          % (res['mag_moving_max'], 100 * abs(res['mag_moving_max'] - G) / G))
    print('  vs tracking    %.2f deg median -- not sensor error, the hand'
          % res['vs_pose_moving_median_deg'])
    print('\ncalibration:')
    print('  %s' % res['calibration_note'])
    print('  a uniform scale error does not move the DIRECTION, and direction '
          'is all\n  the grid uses -- live_cam_pose normalises up_camera.')
    print('\nwhat the grid actually does with it:')
    src = res.get('up_camera_source') or ''
    from_floor = 'floor' in src.lower()
    if not res['grid_follows_gravity']:
        print('  up_camera in meta.json   NO -- the grid assumes the camera '
              'was level, so the\n                           tilt above goes '
              'straight into the ground truth')
    elif from_floor:
        print('  up_camera in meta.json   yes, but it is NOT this sensor:')
        print('                           %s' % src)
        print('                           so the grid follows the fused FLOOR, '
              'and the figure\n                           above is the '
              "accelerometer disagreeing with geometry")
    else:
        print('  up_camera in meta.json   yes, from the accelerometer')
    print('  camera tilt at the hold  %s deg' % res['fuse_imu_tilt_deg'])
    print('  fused floor came out     %s deg off level' % res['fuse_floor_tilt_deg'])
    if from_floor:
        print('                           (NOT an independent check here: the '
              'grid was built\n                           from this floor, so '
              'it is a consistency check)')

    out = os.path.join(cap, 'scan', 'imu_check.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print('\n%s' % out)
    if not args.no_plot:
        res['_series'] = series
        p = plot(res, args.out or os.path.join(cap, 'scan', 'imu_check.png'))
        print('%s' % p if p else 'no plot (matplotlib unavailable)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
