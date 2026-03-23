"""
BCI Competition IV Dataset 2b — Automatic Downloader and CSV Exporter

Downloads the BCI Competition IV Dataset 2b using MOABB (Mother of All BCI Benchmarks)
and exports each subject's data to CSV files compatible with the main.py pipeline.

Dataset: BCI Competition IV 2b
- 9 subjects
- 2 classes: Left Hand (class 1) vs Right Hand (class 2)
- 3 EEG channels: C3, Cz, C4
- 5 sessions per subject (sessions 1-2: training with feedback, 3-5: evaluation)
- Sampling rate: 250 Hz
- Trial duration: 4s post-cue

Usage:
    python download_dataset.py
    python download_dataset.py --output-dir data/raw/patients_2b
    python download_dataset.py --subjects 1 2 3
    python download_dataset.py --sessions eval   # only evaluation sessions (3,4,5)
    python download_dataset.py --sessions all    # all 5 sessions (default)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Check dependencies
# ============================================================================
def check_dependencies():
    """Check that required packages are installed, with helpful install instructions."""
    missing = []
    try:
        import mne
    except ImportError:
        missing.append("mne")
    try:
        import moabb
    except ImportError:
        missing.append("moabb")

    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("\nInstall all at once:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


# ============================================================================
# Download and export
# ============================================================================
def download_and_export(output_dir: Path, subjects: list, sessions: str, sfreq_target: float = 250.0):
    """
    Download BCI Competition IV 2b via MOABB and export to CSV.

    Parameters
    ----------
    output_dir : Path
        Directory where CSV files will be saved.
    subjects : list of int
        Subject IDs to download (1–9).
    sessions : str
        'all' for all 5 sessions, 'eval' for evaluation sessions (3,4,5) only,
        'train' for training sessions (1,2) only.
    sfreq_target : float
        Target sampling frequency. Dataset native is 250 Hz.
    """
    import mne
    from moabb.datasets import BNCI2014_004  # This is BCI Competition IV 2b

    mne.set_log_level("WARNING")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BCI Competition IV Dataset 2b")
    print(f"  Subjects: {subjects}")
    print(f"  Sessions: {sessions}")
    print(f"  Output:   {output_dir.resolve()}")
    print()

    dataset = BNCI2014_004()

    # Session filter
    session_map = {
        'all':   [0, 1, 2, 3, 4],  # moabb uses 0-indexed sessions
        'eval':  [2, 3, 4],
        'train': [0, 1],
    }
    if sessions not in session_map:
        raise ValueError(f"sessions must be 'all', 'eval', or 'train'. Got: {sessions}")

    label_map = {1: 'left', 2: 'right'}  # BCI 2b class codes
    # Note: MOABB/MNE uses event ids. BNCI2014_004 events: 1=Left, 2=Right

    for subject_id in subjects:
        print(f"Processing subject {subject_id}...")

        try:
            # get_data returns {subject: {session: {run: raw}}}
            data = dataset.get_data(subjects=[subject_id])
        except Exception as e:
            print(f"  ERROR downloading subject {subject_id}: {e}")
            continue

        all_rows = []
        epoch_counter = 0

        subj_sessions = data[subject_id]
        session_keys = sorted(subj_sessions.keys())  # e.g. ['session_0', 'session_1', ...]

        # Filter by requested sessions
        target_indices = session_map[sessions]
        session_keys_filtered = [k for i, k in enumerate(session_keys) if i in target_indices]

        for sess_key in session_keys_filtered:
            runs = subj_sessions[sess_key]
            for run_key, raw in runs.items():
                # Resample if needed
                if abs(raw.info['sfreq'] - sfreq_target) > 1:
                    raw = raw.resample(sfreq_target, npad='auto')

                # Get EEG channels only
                raw_eeg = raw.copy().pick_types(eeg=True, stim=False, eog=False)
                ch_names = raw_eeg.ch_names

                # Get events
                events, event_id = mne.events_from_annotations(raw_eeg, verbose=False)

                # Map class labels
                # BNCI2014_004 uses event IDs: typically 1=769 (Left), 2=770 (Right) in MNE
                # MOABB normalizes these; check what event_id contains
                left_id = None
                right_id = None
                for name, code in event_id.items():
                    name_lower = name.lower()
                    if 'left' in name_lower or '769' in name or name == '1':
                        left_id = code
                    elif 'right' in name_lower or '770' in name or name == '2':
                        right_id = code

                if left_id is None or right_id is None:
                    # Fallback: try numeric keys
                    keys = sorted(event_id.values())
                    if len(keys) >= 2:
                        left_id, right_id = keys[0], keys[1]
                    else:
                        print(f"    Warning: could not identify left/right events in {sess_key}/{run_key}. Skipping.")
                        continue

                # Epoch: 0 to 4 seconds post-cue
                tmin, tmax = 0.0, 4.0
                target_ids = {left_id, right_id}
                target_events = np.array([e for e in events if e[2] in target_ids])

                if len(target_events) == 0:
                    print(f"    Warning: no target events found in {sess_key}/{run_key}. Skipping.")
                    continue

                data_array, times = raw_eeg[:]
                # data_array shape: (n_channels, n_times_total)
                sfreq = raw_eeg.info['sfreq']

                for event in target_events:
                    onset_sample = event[0]
                    event_code = event[2]

                    if event_code == left_id:
                        label = 'left'
                    elif event_code == right_id:
                        label = 'right'
                    else:
                        continue

                    start = onset_sample
                    end = onset_sample + int(tmax * sfreq)

                    if end > data_array.shape[1]:
                        continue  # skip incomplete trials at end of recording

                    trial_data = data_array[:, start:end]  # (n_ch, n_times)
                    n_times_trial = trial_data.shape[1]
                    time_arr = np.arange(n_times_trial) / sfreq

                    for t_idx in range(n_times_trial):
                        row = {'time': time_arr[t_idx], 'epoch': epoch_counter, 'label': label}
                        for ch_i, ch_name in enumerate(ch_names):
                            col_name = f"EEG-{ch_name}" if not ch_name.startswith("EEG-") else ch_name
                            row[col_name] = float(trial_data[ch_i, t_idx])
                        all_rows.append(row)

                    epoch_counter += 1

        if not all_rows:
            print(f"  WARNING: No data rows for subject {subject_id}. Skipping CSV save.")
            continue

        df = pd.DataFrame(all_rows)

        # Reorder columns: time, epoch, label, EEG-*
        eeg_cols = sorted([c for c in df.columns if c.startswith("EEG-")])
        df = df[['time', 'epoch', 'label'] + eeg_cols]

        out_path = output_dir / f"BCICIV_2b_{subject_id}.csv"
        df.to_csv(out_path, index=False)

        n_trials = df['epoch'].nunique()
        n_left = (df.drop_duplicates('epoch').label == 'left').sum()
        n_right = (df.drop_duplicates('epoch').label == 'right').sum()
        print(f"  Subject {subject_id}: {n_trials} trials ({n_left} left, {n_right} right) → {out_path}")

    print(f"\nDone. CSV files saved to: {output_dir.resolve()}")
    print(f"\nTo use 2b data in main.py, update load_patient_csv() to use 'BCICIV_2b_{{id}}.csv'")
    print(f"or set the DATA_DIR env variable:")
    print(f"  set DATA_DIR={output_dir.resolve()}")


# ============================================================================
# Entry point
# ============================================================================
def main():
    check_dependencies()

    parser = argparse.ArgumentParser(
        description="Download BCI Competition IV Dataset 2b and export to CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/patients_2b"),
        help="Directory to save CSV files (default: data/raw/patients_2b).",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, 10)),
        help="Subject IDs to download (1–9). Default: all 9.",
    )
    parser.add_argument(
        "--sessions",
        choices=["all", "eval", "train"],
        default="all",
        help="Which sessions to include: all (default), eval (sessions 3-5), train (sessions 1-2).",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=250.0,
        help="Target sampling frequency in Hz (default: 250).",
    )

    args = parser.parse_args()

    download_and_export(
        output_dir=args.output_dir,
        subjects=args.subjects,
        sessions=args.sessions,
        sfreq_target=args.sfreq,
    )


if __name__ == "__main__":
    main()
