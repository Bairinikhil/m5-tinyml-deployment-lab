"""Train a leakage-safe first classifier from clean recorder trials."""
import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

AXES = ['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps']
LABELS = ['still', 'shaking', 'rotating']


def load_clean_trials(data_dir):
    trials = []
    for report_path in sorted(Path(data_dir).glob('*.json')):
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report.get('status') != 'complete' or report.get('warnings'):
            continue
        csv_path = report_path.with_suffix('.csv')
        if report.get('label') not in LABELS or not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open(encoding='utf-8')))
        values = np.array([[float(row[key]) for key in AXES] for row in rows], dtype=np.float32)
        if len(values) >= 50:
            trials.append((report['trial'], report['label'], values))
    return trials


def features(window):
    # Keep features interpretable and reproducible on-device later. Magnitudes
    # remove orientation dependence; first differences capture rapid shaking.
    accel_mag = np.linalg.norm(window[:, 0:3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1, keepdims=True)
    signals = np.concatenate([window, accel_mag, gyro_mag], axis=1)
    means = signals.mean(axis=0)
    stds = signals.std(axis=0)
    mins = signals.min(axis=0)
    maxs = signals.max(axis=0)
    rms = np.sqrt(np.mean(signals * signals, axis=0))
    diffs = np.diff(signals, axis=0)
    mean_abs_diff = np.mean(np.abs(diffs), axis=0)
    return np.concatenate([means, stds, mins, maxs, rms, mean_abs_diff])


def make_windows(trials, window=50, step=25):
    x, y, groups = [], [], []
    for trial, label, values in trials:
        for start in range(0, len(values) - window + 1, step):
            x.append(features(values[start:start + window]))
            y.append(label)
            groups.append(trial)
    return np.array(x, dtype=np.float32), np.array(y), np.array(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'model')
    args = parser.parse_args()
    trials = load_clean_trials(args.data)
    counts = {label: sum(item[1] == label for item in trials) for label in LABELS}
    if any(counts[label] < 2 for label in LABELS):
        raise SystemExit(f'Need at least two clean trials per label: {counts}')
    x, y, groups = make_windows(trials)
    # Deterministic holdout: one whole trial per label, so every class is
    # represented and overlapping windows from one trial never cross the split.
    test_groups = {label: sorted(trial for trial, trial_label, _ in trials
                                 if trial_label == label)[-1] for label in LABELS}
    test_idx = np.array([i for i, group in enumerate(groups) if group in test_groups.values()])
    train_idx = np.array([i for i, group in enumerate(groups) if group not in test_groups.values()])
    model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=7, n_jobs=-1)
    model.fit(x[train_idx], y[train_idx])
    predicted = model.predict(x[test_idx])
    labels = LABELS
    cm = confusion_matrix(y[test_idx], predicted, labels=labels).tolist()
    metrics = {
        'clean_trials': counts,
        'window_samples': 50,
        'step_samples': 25,
        'window_seconds': 1.0,
        'train_windows': int(len(train_idx)),
        'test_windows': int(len(test_idx)),
        'test_trials': sorted(set(groups[test_idx])),
        'split': 'one complete trial held out per label',
        'accuracy': float(accuracy_score(y[test_idx], predicted)),
        'labels': labels,
        'confusion_matrix_rows_actual_columns_predicted': cm,
        'classification_report': classification_report(y[test_idx], predicted, labels=labels, zero_division=0, output_dict=True),
        'feature_count': int(x.shape[1]),
        'features': 'six axes plus acceleration/gyro magnitude: mean, standard deviation, minimum, maximum, RMS, mean absolute first difference',
        'note': 'Baseline only; one day and a small dataset. Do not treat this as general accuracy.'
    }
    args.out.mkdir(exist_ok=True)
    with (args.out / 'motion_baseline.pkl').open('wb') as file:
        pickle.dump({'model': model, 'labels': labels, 'feature_count': x.shape[1]}, file)
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({'clean_trials': counts, 'train_windows': len(train_idx),
                      'test_windows': len(test_idx), 'test_trials': metrics['test_trials'],
                      'accuracy': metrics['accuracy'], 'confusion_matrix': cm}, indent=2))


if __name__ == '__main__':
    main()
