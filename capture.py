"""Record one labeled motion trial. Original timing and bad rows are retained."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
import uuid

FIELDS = ['seq', 'device_us', 'valid', 'ax_g', 'ay_g', 'az_g',
          'gx_dps', 'gy_dps', 'gz_dps', 'accel_age_us', 'gyro_age_us', 'usb_drops']


def parse_sample(line):
    parts = line.strip().split(',')
    if not parts or parts[0] != 'M5IMU1':
        return None
    if len(parts) != 13:
        raise ValueError('Wrong number of columns')
    row = dict(zip(FIELDS, parts[1:]))
    for key in FIELDS:
        row[key] = float(row[key]) if key in FIELDS[3:9] else int(row[key])
    if row['valid'] not in (0, 1):
        raise ValueError('Bad validity flag')
    if not all(math.isfinite(row[key]) for key in FIELDS[3:9]):
        row['valid'] = 0
    if row['accel_age_us'] != 0 or row['gyro_age_us'] != 0:
        row['valid'] = 0
    return row


def counter_delta(new, old):
    """Unsigned 32-bit counter difference; rollover is not a reset."""
    return (new - old) & 0xFFFFFFFF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-ports', action='store_true')
    parser.add_argument('--port', help='USB serial port, e.g. COM5')
    parser.add_argument('--label', choices=['still', 'shaking', 'rotating'])
    parser.add_argument('--session', help='Collection session, e.g. day1')
    parser.add_argument('--seconds', type=int, default=20)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'data')
    args = parser.parse_args()
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        parser.exit(1, 'Install dependencies: python -m pip install -r requirements.txt\n')
    if args.list_ports:
        for port in list_ports.comports():
            print(f'{port.device}: {port.description}')
        return
    if not args.port or not args.label or not args.session:
        parser.error('--port, --label and --session are required')
    if not 5 <= args.seconds <= 120:
        parser.error('--seconds must be between 5 and 120')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    trial = f'{stamp}_{args.label}_{uuid.uuid4().hex[:8]}'
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'{trial}.csv'
    meta_path = args.out / f'{trial}.json'
    meta = dict(protocol='M5IMU1', session=args.session, trial=trial,
                label=args.label, requested_seconds=args.seconds,
                target_hz=50, timestamp_utc=stamp, status='incomplete',
                units={'accel': 'g', 'gyro': 'degrees/second', 'device_time': 'microseconds'})
    # Exclusive creation prevents accidental replacement of earlier recordings.
    with meta_path.open('x', encoding='utf-8') as file:
        json.dump(meta, file, indent=2)
    count = invalid = malformed = missing = 0
    intervals = []
    previous = first = last = None
    started = None
    error = None
    try:
        with serial.Serial(args.port, 115200, timeout=0.2) as port, path.open('x', newline='', encoding='utf-8') as file:
            print('Waiting for recorder firmware (up to 15 seconds)...', flush=True)
            deadline = time.monotonic() + 15
            ready = False
            while time.monotonic() < deadline:
                try:
                    ready = parse_sample(port.readline().decode('ascii', errors='replace')) is not None
                except ValueError:
                    continue
                if ready:
                    break
            if not ready:
                raise RuntimeError('No M5IMU1 stream. Check firmware, USB port, and close Serial Monitor.')
            print(f'Prepare: {args.label}. Keep this action going throughout the recording.', flush=True)
            for remaining in (3, 2, 1):
                print(remaining, flush=True)
                time.sleep(1)
            port.reset_input_buffer()
            # Ignore a potentially partial line following the buffer flush.
            port.readline()
            writer = csv.DictWriter(file, fieldnames=['session', 'trial', 'label', 'host_elapsed_s'] + FIELDS)
            writer.writeheader()
            started = time.monotonic()
            last_received = started
            print('RECORDING', flush=True)
            while time.monotonic() - started < args.seconds:
                if time.monotonic() - last_received > 3:
                    raise RuntimeError('Motion stream stopped for over 3 seconds')
                try:
                    row = parse_sample(port.readline().decode('ascii', errors='replace'))
                except ValueError:
                    malformed += 1
                    continue
                if row is None:
                    continue
                if previous is not None:
                    delta = counter_delta(row['seq'], previous['seq'])
                    dt = counter_delta(row['device_us'], previous['device_us'])
                    if delta == 0 or delta > 10000 or dt > 3000000:
                        raise RuntimeError('Device reset or discontinuity: repeat this trial')
                    missing += delta - 1
                    intervals.append(dt)
                first = first or row
                last = previous = row
                last_received = time.monotonic()
                writer.writerow(dict(session=args.session, trial=trial, label=args.label,
                                     host_elapsed_s=round(last_received-started, 6), **row))
                count += 1
                invalid += not row['valid']
            meta['status'] = 'complete'
    except (serial.SerialException, RuntimeError, OSError, KeyboardInterrupt) as exc:
        error = str(exc) or 'Interrupted by user'
        meta['error'] = error
    finally:
        elapsed = time.monotonic() - started if started is not None else 0
        median_us = statistics.median(intervals) if intervals else None
        warnings = []
        if count < args.seconds * 45:
            warnings.append('Fewer than 45 received samples/second; inspect timing before training')
        if invalid or missing or malformed:
            warnings.append('Invalid/missing/malformed samples present; inspect before training')
        if intervals and max(intervals) > 40000:
            warnings.append('Sampling gaps exceed 40 ms; inspect before training')
        meta.update(samples=count, invalid_samples=invalid, missing_sequences=missing,
                    malformed_rows=malformed, elapsed_s=round(elapsed, 3),
                    median_interval_us=median_us, max_interval_us=max(intervals, default=None),
                    median_rate_hz=round(1e6/median_us, 2) if median_us else None,
                    usb_drops_during_trial=counter_delta(last['usb_drops'], first['usb_drops']) if last else 0,
                    warnings=warnings)
        with meta_path.open('w', encoding='utf-8') as file:
            json.dump(meta, file, indent=2)
    print(f"{meta['status'].upper()}: {count} samples; {invalid} invalid; {missing} missing")
    print(f'CSV: {path}\nQuality report: {meta_path}')
    for warning in meta['warnings']:
        print(f'WARNING: {warning}')
    if error:
        raise SystemExit(error)


if __name__ == '__main__':
    main()
