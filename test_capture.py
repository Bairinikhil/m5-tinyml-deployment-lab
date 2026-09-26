import unittest
from unittest.mock import patch
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import csv
import io
from contextlib import redirect_stdout
import capture
from capture import parse_sample, counter_delta


class ParserTests(unittest.TestCase):
    SAMPLE = 'M5IMU1,10,200000,1,0.1,-0.2,1.0,2.0,3.0,4.0,0,0,2'

    def test_valid(self):
        row = parse_sample(self.SAMPLE)
        self.assertEqual(row['seq'], 10)
        self.assertEqual(row['valid'], 1)
        self.assertEqual(row['az_g'], 1.0)

    def test_boot_log(self):
        self.assertIsNone(parse_sample('ESP-ROM: startup'))

    def test_short(self):
        with self.assertRaises(ValueError):
            parse_sample('M5IMU1,1,2')

    def test_nonfinite(self):
        self.assertEqual(parse_sample(self.SAMPLE.replace('0.1,', 'nan,'))['valid'], 0)

    def test_stale(self):
        self.assertEqual(parse_sample(self.SAMPLE[:-5] + '20,0,2')['valid'], 0)

    def test_counter_wrap(self):
        self.assertEqual(counter_delta(5, 0xFFFFFFFE), 7)


class CaptureTests(unittest.TestCase):
    def run_capture(self, out, disconnect=False):
        clock = [0.0]

        class Disconnected(Exception):
            pass

        class Port:
            seq = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def reset_input_buffer(self):
                pass

            def readline(self):
                self.seq += 1
                clock[0] += 0.02
                if disconnect and self.seq > 10:
                    raise Disconnected('Simulated USB disconnect')
                return f'M5IMU1,{self.seq},{self.seq*20000},1,0,0,1,0,0,0,0,0,0\n'.encode()

        fake_serial = SimpleNamespace(Serial=lambda *a, **k: Port(), SerialException=Disconnected)
        modules = {'serial': fake_serial, 'serial.tools': SimpleNamespace(list_ports=None)}
        args = ['capture.py', '--port', 'FAKE', '--label', 'still', '--session', 'test',
                '--seconds', '5', '--out', str(out)]
        with patch.dict('sys.modules', modules), patch('sys.argv', args), \
             patch('capture.time.monotonic', side_effect=lambda: clock[0]), \
             patch('capture.time.sleep', side_effect=lambda n: clock.__setitem__(0, clock[0]+n)), \
             redirect_stdout(io.StringIO()):
            capture.main()

    def test_complete_capture_and_unique_files(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            self.run_capture(out)
            self.run_capture(out)
            self.assertEqual(len(list(out.glob('*.csv'))), 2)
            for report in out.glob('*.json'):
                meta = json.loads(report.read_text())
                self.assertEqual(meta['status'], 'complete')
                self.assertEqual(meta['warnings'], [])
                self.assertEqual(meta['median_rate_hz'], 50)
                with report.with_suffix('.csv').open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), meta['samples'])
                self.assertTrue(all(row['label'] == 'still' for row in rows))

    def test_disconnect_marks_trial_incomplete(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                self.run_capture(Path(directory), disconnect=True)
            meta = json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(meta['status'], 'incomplete')
            self.assertIn('disconnect', meta['error'])


if __name__ == '__main__':
    unittest.main()
