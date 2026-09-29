#!/usr/bin/env bash
# Step 1: is the D455 there, and on USB 3?
#
#   ./reconstruction_GT/1_check_camera.sh
#
# A charge-only or USB 2 cable enumerates at 2.1 and the streams fail to start
# later, with an error that does not mention the cable. Check it here instead.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

say "RealSense devices"
"$PY" - <<'PY'
import sys
import pyrealsense2 as rs
devs = list(rs.context().query_devices())
if not devs:
    print('none found')
    print('  - plug the D455 into a USB 3 port (blue, or marked SS)')
    print('  - check the udev rules: 99-realsense-libusb.rules')
    sys.exit(1)
bad = False
for d in devs:
    usb = d.get_info(rs.camera_info.usb_type_descriptor)
    print('%s  fw %s  serial %s  USB %s' % (
        d.get_info(rs.camera_info.name),
        d.get_info(rs.camera_info.firmware_version),
        d.get_info(rs.camera_info.serial_number), usb))
    sensors = [s.get_info(rs.camera_info.name) for s in d.query_sensors()]
    print('  sensors:', ', '.join(sensors))
    if not usb.startswith('3'):
        print('  USB %s -- too slow for 848x480 @ 15 fps. Change the cable or '
              'the port.' % usb)
        bad = True
    if not any('Motion' in s for s in sensors):
        print('  no motion module: fuse_scan will skip its IMU tilt check')
sys.exit(2 if bad else 0)
PY
say "accelerometer permissions"
"$PY" - <<'PY' || true
import pyrealsense2 as rs
cfg = rs.config(); cfg.enable_stream(rs.stream.accel)
pipe = rs.pipeline()
try:
    pipe.start(cfg); pipe.stop()
    print('accelerometer opens -- fuse_scan can run its tilt check')
except RuntimeError as e:
    msg = str(e)
    print('CANNOT OPEN THE ACCELEROMETER:')
    print('  %s' % msg)
    if 'Permission denied' in msg or 'scan_element' in msg:
        print('')
        print('The udev rules are missing. Install them, then REPLUG the camera:')
        print('  sudo cp ~/CleanerS/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/')
        print('  sudo udevadm control --reload-rules && sudo udevadm trigger')
    print('')
    print('Without it, recordings carry no gravity and fuse_scan skips its tilt check.')
PY

say "camera OK -- next: ./reconstruction_GT/2_rig_test.sh"
