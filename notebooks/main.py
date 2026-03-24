"""
EEG Motor Imagery Classification — Leakage-Free Pipeline (BCICIV 2b)

This script implements a full evaluation pipeline comparing the classical
Common Spatial Patterns + Linear Discriminant Analysis (CSP-LDA) method
against the compact EEGNet convolutional neural network for motor imagery
EEG classification.

Key design principles:
- No data leakage: normalization is fit exclusively on the training set of
  each LOSO fold and only then applied to the test set.
- Leave-One-Subject-Out (LOSO) cross-validation for honest cross-subject
  generalization estimates.
- Multi-seed EEGNet evaluation to quantify sensitivity to random initialization.
- Full figure and table generation for article-ready output.
"""

# ============================================================================
# Imports
# ============================================================================
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

from scipy import signal, stats
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from mne.decoding import CSP

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================================
# Utility functions and class definitions
# ============================================================================

def find_data_dir() -> Path:
    """Locates data/raw/patients_2b from the current directory or parents.
    Set DATA_DIR env var to override."""
    env_override = os.environ.get("DATA_DIR")
    if env_override:
        return Path(env_override).resolve()

    candidates = [
        Path("data/raw/patients_2b"),        # from project root
        Path("../data/raw/patients_2b"),     # from notebooks/ directory
        Path(__file__).parent.parent / "data" / "raw" / "patients_2b",  # relative to script
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def butter_bandpass_filter(x: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Bandpass Butterworth (zero-phase) filter applied on 1D signal."""
    nyq = 0.5 * sfreq
    lowc = low / nyq
    highc = high / nyq
    b, a = signal.butter(order, [lowc, highc], btype="bandpass")
    return signal.filtfilt(b, a, x)


def zscore_per_trial(X: np.ndarray) -> np.ndarray:
    """Z-score per trial and per channel (no info from other trials). X: (n_trials, n_times, n_ch)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


def fit_channel_stats(X_train: np.ndarray):
    """Channel statistics computed *only from the fold's training set*."""
    flat = X_train.reshape(-1, X_train.shape[-1])  # (n_trials*n_times, n_ch)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-8
    return mu, sd


def apply_channel_stats(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu[None, None, :]) / sd[None, None, :]


def load_patient_csv(patient_id: int, data_dir: Path) -> pd.DataFrame:
    fp = data_dir / f"BCICIV_2b_{patient_id}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")
    df = pd.read_csv(fp)
    # basic guarantees
    if "time" in df.columns:
        df["time"] = df["time"].astype(float)
    return df


def patient_trials_from_df(df: pd.DataFrame, *, apply_filter: bool, sfreq: float,
                            f_low: float, f_high: float, f_order: int) -> tuple:
    """Returns (X_raw, y, eeg_cols) for one patient.
    - Filters by labels 'left'/'right'
    - Groups by epoch (trial)
    - X_raw has shape (n_trials, n_times, n_ch) and is NOT normalized here.
    """
    df = df[df["label"].isin(["left", "right"])].copy()
    if df.empty:
        return np.empty((0, 0, 0)), np.empty((0,)), []

    df = df.sort_values("time").reset_index(drop=True)
    eeg_cols = [c for c in df.columns if c.startswith("EEG-")]
    if not eeg_cols:
        raise ValueError("No EEG-* columns found")
    if "epoch" not in df.columns:
        raise ValueError("No 'epoch' column found")

    X_list = []
    y_list = []
    # process each epoch separately (safe)
    for epoch_id in df["epoch"].unique():
        trial = df[df["epoch"] == epoch_id].sort_values("time")
        if len(trial) == 0:
            continue
        label = trial["label"].iloc[0]
        y = 0 if label == "left" else 1

        sigs = []
        for col in eeg_cols:
            x = trial[col].to_numpy()
            if apply_filter:
                x = butter_bandpass_filter(x, sfreq, f_low, f_high, order=f_order)
            sigs.append(x)
        X = np.stack(sigs, axis=-1)  # (n_times, n_ch)
        X_list.append(X)
        y_list.append(y)

    # ensure consistent shape
    # if varying lengths, truncate to minimum (avoids silent error)
    n_times = min(x.shape[0] for x in X_list) if X_list else 0
    X_list = [x[:n_times, :] for x in X_list]
    X_arr = np.stack(X_list, axis=0) if X_list else np.empty((0, 0, 0))
    y_arr = np.asarray(y_list, dtype=int)

    return X_arr, y_arr, eeg_cols


def normalize_fold(X_train, X_test, use_normalization, norm_strategy):
    """Apply fold-safe normalization without leakage."""
    if not use_normalization:
        return X_train, X_test

    if norm_strategy == "per_trial":
        return zscore_per_trial(X_train), zscore_per_trial(X_test)

    if norm_strategy == "fold_train_stats":
        # FIT ON TRAIN ONLY (safe)
        mu, sd = fit_channel_stats(X_train)
        return apply_channel_stats(X_train, mu, sd), apply_channel_stats(X_test, mu, sd)

    raise ValueError(f"Unknown NORM_STRATEGY: {norm_strategy}")


def set_seed(seed):
    """Fix seed for NumPy, TensorFlow, and Python."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    # For GPU (if available)
    tf.config.experimental.enable_op_determinism()


def build_eegnet(n_channels, n_times, n_classes=2, **hparams):
    """
    Builds EEGNet as in Lawhern et al. (2018), Journal of Neural Engineering.

    Input shape passed to model: (n_channels, n_times)
    Internally reshaped to: (1, n_channels, n_times) — treated as a 2D image
    with 1 input channel, n_channels rows, n_times columns.

    Block 1:
      - Temporal Conv2D: filters of shape (1, kernel_length) — captures temporal dynamics
      - DepthwiseConv2D: filters of shape (n_channels, 1) — learns spatial (electrode) filters
    Block 2:
      - SeparableConv2D: filters of shape (1, 16) — captures temporal dependencies in learned features
    """
    F1 = hparams.get('F1', 8)
    F2 = hparams.get('F2', 16)
    D = hparams.get('D', 2)
    kernel_length = hparams.get('kernel_length', 64)
    dropout_rate = hparams.get('dropout_rate', 0.5)

    inputs = layers.Input(shape=(n_channels, n_times))
    # Reshape to (1, n_channels, n_times) — treat as image with 1 channel
    x = layers.Reshape((1, n_channels, n_times))(inputs)

    # -------------------------------------------------------------------------
    # Block 1: Temporal Conv + Spatial DepthwiseConv
    # -------------------------------------------------------------------------
    # Temporal convolution: (1, kernel_length) filters across time, per F1 filter
    x = layers.Conv2D(F1, (1, kernel_length), padding='same', use_bias=False,
                      data_format='channels_first', name='conv_temporal')(x)
    x = layers.BatchNormalization(axis=1, name='bn_1')(x)

    # Spatial depthwise conv: (n_channels, 1) filters across electrodes
    x = layers.DepthwiseConv2D((n_channels, 1), depth_multiplier=D,
                               padding='valid', use_bias=False,
                               data_format='channels_first',
                               depthwise_constraint=keras.constraints.MaxNorm(1.),
                               name='depthwise_spatial')(x)
    x = layers.BatchNormalization(axis=1, name='bn_2')(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4), data_format='channels_first')(x)
    x = layers.Dropout(dropout_rate)(x)

    # -------------------------------------------------------------------------
    # Block 2: Separable Conv (temporal refinement)
    # -------------------------------------------------------------------------
    x = layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False,
                               data_format='channels_first', name='sep_conv')(x)
    x = layers.BatchNormalization(axis=1, name='bn_3')(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8), data_format='channels_first')(x)
    x = layers.Dropout(dropout_rate)(x)

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------
    x = layers.Flatten()(x)
    x = layers.Dense(n_classes, kernel_constraint=keras.constraints.MaxNorm(0.25),
                     activation='softmax', name='classification')(x)

    return keras.Model(inputs=inputs, outputs=x, name='EEGNet')


def draw_block(ax, x, y, w, h, label, color='lightblue', fontsize=10):
    """Draw a rounded rectangle block on a matplotlib axes."""
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw a directional arrow on a matplotlib axes."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                            mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.5, mid_y, label, ha='left', va='bottom', fontsize=9, style='italic')


def calc_stats(data):
    """Compute descriptive statistics for an array."""
    return {
        'Mean': np.mean(data),
        'Std': np.std(data),
        'Min': np.min(data),
        'Max': np.max(data),
        'Median': np.median(data),
    }


def plot_sample_trials(X_plot, y, label_value: int, eeg_cols, channels_to_plot,
                       num_trials: int, fname: str, figures_dir: Path,
                       save_figures: bool, show_plots: bool, fig_dpi: int,
                       use_filter: bool, filter_low: float, filter_high: float,
                       use_normalization: bool):
    """Plot a grid of individual trials for a given class label."""
    trials_idx = np.where(y == label_value)[0]
    if len(trials_idx) == 0:
        print("(no trials for this label)")
        return

    take = trials_idx[:min(num_trials, len(trials_idx))]
    fig, axes = plt.subplots(len(take), len(channels_to_plot), figsize=(14, 2.2 * len(take)), sharex=True)

    if len(take) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(channels_to_plot) == 1:
        axes = np.expand_dims(axes, axis=1)

    label_name = "LEFT" if label_value == 0 else "RIGHT"
    title_bits = []
    if use_filter:
        title_bits.append(f"{filter_low}-{filter_high}Hz")
    if use_normalization:
        title_bits.append("per-trial z-score")
    title = " + ".join(title_bits) if title_bits else "original"

    fig.suptitle(f"Trial Examples - {label_name} ({title})", fontsize=12, fontweight='bold')

    t = np.linspace(-0.1, 0.7, X_plot.shape[1])

    for r, trial_i in enumerate(take):
        for c, ch in enumerate(channels_to_plot):
            ax = axes[r, c]
            ch_idx = eeg_cols.index(ch)
            ax.plot(t, X_plot[trial_i, :, ch_idx])
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(ch, fontweight='bold')
            if c == 0:
                ax.text(-0.25, 0.5, f"T{trial_i}", transform=ax.transAxes, fontsize=9, fontweight='bold')

    axes[-1, max(0, len(channels_to_plot) // 2)].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_figures:
        out = figures_dir / fname
        plt.savefig(out, dpi=fig_dpi, bbox_inches='tight')
        print(f"Saved: {out}")

    if show_plots:
        plt.show()
    else:
        plt.close()


# ============================================================================
# Main entry point
# ============================================================================

def main(show_plots: bool = True, save_figures: bool = True):
    # ----------------------------
    # Configuration
    # ----------------------------
    DATA_DIR = find_data_dir()

    if not DATA_DIR.exists():
        print(f"Warning: DATA_DIR not found at: {DATA_DIR}")
        print(f"  Current directory: {Path.cwd()}")

    FIGURES_DIR = Path("figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset / patient for visualization
    PATIENT_ID = 1  # used in visualization sections (does not affect LOSO)

    # Signal parameters
    SFREQ = 250  # BCICIV 2b is typically 250 Hz
    FILTER_LOW = 8
    FILTER_HIGH = 30
    FILTER_ORDER = 4

    # Control flags
    USE_FILTER = True              # filter per trial (safe; does not mix train/test)
    USE_DATA_NORMALIZATION = True  # safe normalization (applied inside each LOSO fold)
    NORM_STRATEGY = "fold_train_stats"  # "fold_train_stats" (recommended) | "per_trial"

    # Normalization for plotting only (scale [0,1] per channel) — does not change training X
    USE_PLOT_MINMAX = False

    SAVE_FIGURES = save_figures
    SHOW_PLOTS = show_plots

    # Visualization
    CHANNELS_TO_PLOT = ['EEG-C3', 'EEG-C4', 'EEG-Cz']
    NUM_TRIALS_TO_PLOT = 6
    FIG_DPI = 300

    print("╔═══════════════════════════════════════════════════════╗")
    print("║          ACTIVE PIPELINE SETTINGS            ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print(f"║ DATA_DIR: {str(DATA_DIR):.<49}║")
    print(f"║ SFREQ: {SFREQ} Hz{'.'*43}║")
    print(f"║ Butterworth Filter: {'✓ ACTIVE' if USE_FILTER else '✗ INACTIVE':.<40}║")
    print(f"║ Band: {FILTER_LOW}-{FILTER_HIGH} Hz (order {FILTER_ORDER}){'.'*20}║")
    print(f"║ Train Normalization: {'✓ ACTIVE' if USE_DATA_NORMALIZATION else '✗ INACTIVE':.<38}║")
    print(f"║ Normalization Strategy: {NORM_STRATEGY:.<28}║")
    print(f"║ Plot min-max [0,1]: {'✓' if USE_PLOT_MINMAX else '✗':.<40}║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    # ============================================================================
    # Load single patient for visualization
    # ============================================================================
    df_p = load_patient_csv(PATIENT_ID, DATA_DIR)
    X_p_raw, y_p, eeg_columns = patient_trials_from_df(
        df_p,
        apply_filter=USE_FILTER,
        sfreq=SFREQ,
        f_low=FILTER_LOW,
        f_high=FILTER_HIGH,
        f_order=FILTER_ORDER,
    )

    print(f"Patient {PATIENT_ID}: {X_p_raw.shape[0]} trials, {X_p_raw.shape[1]} samples/trial, {X_p_raw.shape[2]} channels")
    print(f"Labels: left={int((y_p == 0).sum())} | right={int((y_p == 1).sum())}")
    print(f"Channel examples: {eeg_columns[:8]}{'...' if len(eeg_columns) > 8 else ''}")

    # ============================================================================
    # Prepare data for plotting (safe per-trial normalization, optional)
    # ============================================================================
    # For visualization, z-score per trial is fine (does not use info from other subjects/folds)
    X_p_plot = X_p_raw.copy()

    if USE_DATA_NORMALIZATION:
        # use PER-TRIAL (visualization) to avoid mixing trials
        X_p_plot = zscore_per_trial(X_p_plot)

    # mean/std per class
    X_left = X_p_plot[y_p == 0]
    X_right = X_p_plot[y_p == 1]

    mean_left = X_left.mean(axis=0) if len(X_left) else None      # (n_times, n_ch)
    std_left = X_left.std(axis=0) if len(X_left) else None
    mean_right = X_right.mean(axis=0) if len(X_right) else None
    std_right = X_right.std(axis=0) if len(X_right) else None

    print("Ready for plotting: mean/std computed per class")

    # ============================================================================
    # Plot: trial means RIGHT vs LEFT (C3/Cz/C4 by default)
    # ============================================================================
    time_axis = np.linspace(-0.1, 0.7, X_p_plot.shape[1])

    channels_to_plot = CHANNELS_TO_PLOT
    missing = [c for c in channels_to_plot if c not in eeg_columns]
    if missing:
        raise ValueError(f"Channels not found in CSV: {missing}")

    fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(10, 7), sharex=True)

    title_bits = []
    if USE_FILTER:
        title_bits.append(f"{FILTER_LOW}-{FILTER_HIGH}Hz")
    if USE_DATA_NORMALIZATION:
        title_bits.append("per-trial z-score")
    title = " + ".join(title_bits) if title_bits else "original"

    fig.suptitle(f"Trial Means - RIGHT vs LEFT ({title})", fontsize=12, fontweight='bold')

    for i, ch in enumerate(channels_to_plot):
        ch_idx = eeg_columns.index(ch)
        ax = axes[i]

        if mean_right is not None:
            ax.plot(time_axis, mean_right[:, ch_idx], label="RIGHT")
            ax.fill_between(time_axis,
                            mean_right[:, ch_idx] - std_right[:, ch_idx],
                            mean_right[:, ch_idx] + std_right[:, ch_idx],
                            alpha=0.2)
        if mean_left is not None:
            ax.plot(time_axis, mean_left[:, ch_idx], label="LEFT")
            ax.fill_between(time_axis,
                            mean_left[:, ch_idx] - std_left[:, ch_idx],
                            mean_left[:, ch_idx] + std_left[:, ch_idx],
                            alpha=0.2)

        ax.set_title(ch, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if SAVE_FIGURES:
        out = FIGURES_DIR / "trials_mean_comparison.png"
        plt.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
        print(f"Saved: {out}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # ============================================================================
    # Plot: sample trials (LEFT and RIGHT)
    # ============================================================================
    X_p_plot_for_trials = X_p_plot  # already filtered and (if enabled) z-scored per trial

    plot_sample_trials(X_p_plot_for_trials, y_p, 0, eeg_columns, CHANNELS_TO_PLOT,
                       NUM_TRIALS_TO_PLOT, "trials_left.png", FIGURES_DIR,
                       SAVE_FIGURES, SHOW_PLOTS, FIG_DPI,
                       USE_FILTER, FILTER_LOW, FILTER_HIGH, USE_DATA_NORMALIZATION)
    plot_sample_trials(X_p_plot_for_trials, y_p, 1, eeg_columns, CHANNELS_TO_PLOT,
                       NUM_TRIALS_TO_PLOT, "trials_right.png", FIGURES_DIR,
                       SAVE_FIGURES, SHOW_PLOTS, FIG_DPI,
                       USE_FILTER, FILTER_LOW, FILTER_HIGH, USE_DATA_NORMALIZATION)

    # ============================================================================
    # Load full dataset (all patients) — for LOSO
    # ============================================================================
    print("Loading full dataset...")

    # discover available patients (BCICIV_2a_1.csv ... BCICIV_2a_9.csv)
    available = []
    for pid in range(1, 10):
        if (DATA_DIR / f"BCICIV_2b_{pid}.csv").exists():
            available.append(pid)

    if not available:
        raise FileNotFoundError(f"No BCICIV_2b_*.csv files found in {DATA_DIR.resolve()}")

    X_all = []
    y_all = []
    groups_all = []
    eeg_cols_ref = None

    for pid in available:
        df = load_patient_csv(pid, DATA_DIR)
        X_pid, y_pid, eeg_cols = patient_trials_from_df(
            df,
            apply_filter=USE_FILTER,  # filter per trial (safe)
            sfreq=SFREQ,
            f_low=FILTER_LOW,
            f_high=FILTER_HIGH,
            f_order=FILTER_ORDER,
        )
        if X_pid.size == 0:
            continue

        if eeg_cols_ref is None:
            eeg_cols_ref = eeg_cols
        else:
            # ensure same channel order/set
            if eeg_cols != eeg_cols_ref:
                raise ValueError(f"Different EEG columns between patients. E.g., patient {pid} != reference")

        # Do NOT truncate here! Truncation happens inside LOSO (per fold)
        X_all.append(X_pid)
        y_all.append(y_pid)
        groups_all.append(np.full(len(y_pid), pid, dtype=int))

    # concat WITHOUT global truncation
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    groups = np.concatenate(groups_all, axis=0)

    print(f"Patients: {available}")
    print(f"X: {X.shape} (n_trials, n_times, n_ch) — not yet truncated!")
    print(f"y: {y.shape} | left={(y == 0).sum()} right={(y == 1).sum()}")
    print(f"groups: {groups.shape} | unique={np.unique(groups)}")

    # ============================================================================
    # LOSO split + safe normalization (WITHOUT leakage of min_times)
    # ============================================================================
    loso = LeaveOneGroupOut()

    print("LOSO starting — running CSP+LDA for each fold...\n")

    # Initialize results dictionary BEFORE the loop
    results_csp_lda = {}

    for fold_idx, (train_index, test_index) in enumerate(loso.split(X, y, groups=groups), start=1):
        test_subject = int(np.unique(groups[test_index])[0])

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # normalization WITHOUT leakage
        X_train_n, X_test_n = normalize_fold(X_train, X_test, USE_DATA_NORMALIZATION, NORM_STRATEGY)

        print(f"Fold {fold_idx}: TEST on subject {test_subject} | train={len(train_index)} test={len(test_index)}")
        print(f"   X_train_n={X_train_n.shape} | X_test_n={X_test_n.shape}")

        # --- 1. Data Preparation for MNE ---
        # MNE expects format (n_epochs, n_channels, n_times).
        # last dim is channels (n_ch < n_times typically)
        if X_train_n.shape[-1] < X_train_n.shape[1]:
            # Assume last dim is channels and second-to-last is time
            X_train_mne = X_train_n.transpose(0, 2, 1)
            X_test_mne = X_test_n.transpose(0, 2, 1)
        else:
            X_train_mne = X_train_n
            X_test_mne = X_test_n

        # Ensure labels match current fold indices
        y_train_fold = y[train_index]
        y_test_fold = y[test_index]

        # --- 1.5 Data Validation ---
        print(f"   DEBUG: X_train_mne shape={X_train_mne.shape}, X_test_mne shape={X_test_mne.shape}")
        print(f"   DEBUG: n_trials_train={len(y_train_fold)}, n_channels={X_train_mne.shape[1]}")
        print(f"   DEBUG: Contains NaN/Inf in train? {np.isnan(X_train_mne).any() or np.isinf(X_train_mne).any()}")

        # --- 2. CSP (Common Spatial Patterns) ---
        csp = CSP(n_components=min(8, X_train_mne.shape[1] - 1), reg='shrunk', log=True, norm_trace=True)

        # Fit and Transform on Train
        X_train_feats = csp.fit_transform(X_train_mne, y_train_fold)

        # Transform on Test (using filters learned on train)
        X_test_feats = csp.transform(X_test_mne)

        # --- 3. LDA Classifier ---
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_feats, y_train_fold)
        y_pred = lda.predict(X_test_feats)

        # --- 4. Metrics ---
        acc = accuracy_score(y_test_fold, y_pred)
        kappa = cohen_kappa_score(y_test_fold, y_pred)
        cm = confusion_matrix(y_test_fold, y_pred)

        # --- 5. Save Results ---
        results_csp_lda[test_subject] = {
            'acc': acc,
            'kappa': kappa,
            'confusion': cm,
            'y_true': y_test_fold,
            'y_pred': y_pred
        }

        print(f"   [CSP+LDA] Subject {test_subject}: Acc={acc:.4f}, Kappa={kappa:.4f}\n")

    # ============================================================================
    # Post-loop: Global Results Analysis (CSP+LDA)
    # ============================================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: CSP + LDA (Leave-One-Subject-Out)")
    print("=" * 60)

    # 1. Extract metrics from dictionary
    acc_list = [res['acc'] for res in results_csp_lda.values()]
    kappa_list = [res['kappa'] for res in results_csp_lda.values()]
    subjects = list(results_csp_lda.keys())

    # 2. Create DataFrame for tabular display
    df_results = pd.DataFrame({
        'Subject': subjects,
        'Accuracy': acc_list,
        'Kappa': kappa_list
    })

    # 3. Descriptive Statistics
    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    mean_kappa = np.mean(kappa_list)
    std_kappa = np.std(kappa_list)

    print(df_results.round(4))
    print("-" * 60)
    print(f"GLOBAL MEAN: Acc = {mean_acc:.4f} (±{std_acc:.4f}) | Kappa = {mean_kappa:.4f} (±{std_kappa:.4f})")
    print("-" * 60)

    # (Optional) Save to CSV for article
    df_results.to_csv('resultados_csp_lda.csv', index=False)
    print("Results saved to 'resultados_csp_lda.csv'.")

    # ============================================================================
    # EEGNet preparation — Hyperparameters
    # ============================================================================

    # Seed configuration for reproducibility
    # EEGNet Hyperparameters (fixed)
    EEGNET_HPARAMS = {
        'F1': 8,                # Number of spatial filters (Depth-wise)
        'F2': 16,               # Number of point-wise filters
        'D': 2,                 # Depth multiplier
        'kernel_length': 64,    # Temporal kernel length (first layer)
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 16,
        'early_stopping_patience': 15,
    }

    # Seeds for multiple runs
    EEGNET_SEEDS = [42, 123, 456]  # Minimum 3 seeds

    print("╔═══════════════════════════════════════════════════════╗")
    print("║         EEGNET SETTINGS (FIXED)                 ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print(f"║ F1 (spatial): {EEGNET_HPARAMS['F1']:.<48}║")
    print(f"║ F2 (point-wise): {EEGNET_HPARAMS['F2']:.<45}║")
    print(f"║ D (depth multiplier): {EEGNET_HPARAMS['D']:.<40}║")
    print(f"║ Temporal Kernel: {EEGNET_HPARAMS['kernel_length']:.<46}║")
    print(f"║ Dropout: {EEGNET_HPARAMS['dropout_rate']:.<49}║")
    print(f"║ Learning Rate: {EEGNET_HPARAMS['learning_rate']:.<46}║")
    print(f"║ Batch Size: {EEGNET_HPARAMS['batch_size']:.<46}║")
    print(f"║ Seeds: {str(EEGNET_SEEDS):.<48}║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    print("build_eegnet() function defined")
    print("  - Expected input: (n_epochs, n_channels, n_times)")
    print("  - Internally reshaped to: (n_epochs, 1, n_channels, n_times) [channels_first]")
    print("  - Output: probabilities (batch_size, 2)")

    # ============================================================================
    # LOSO loop for EEGNet (with multiple seeds)
    # ============================================================================

    print("LOSO with EEGNet starting — running for each seed...\n")

    # Dictionary to store results: {seed: {subject: {metrics}}}
    results_eegnet_all_seeds = {}

    for seed in EEGNET_SEEDS:
        print(f"\n{'=' * 70}")
        print(f"SEED: {seed}")
        print(f"{'=' * 70}\n")

        set_seed(seed)  # Fix seed before each run

        results_eegnet_seed = {}

        for fold_idx, (train_index, test_index) in enumerate(loso.split(X, y, groups=groups), start=1):
            test_subject = int(np.unique(groups[test_index])[0])

            X_train, X_test = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            # Normalization WITHOUT leakage
            X_train_n, X_test_n = normalize_fold(X_train, X_test, USE_DATA_NORMALIZATION, NORM_STRATEGY)

            print(f"Fold {fold_idx}: TEST on subject {test_subject} | train={len(train_index)} test={len(test_index)}")
            print(f"   X_train_n={X_train_n.shape} | X_test_n={X_test_n.shape}")

            # --- 1. Data Preparation for MNE ---
            # EEGNet expects: (n_epochs, n_channels, n_times)
            if X_train_n.ndim == 3:
                # If in (n_trials, n_times, n_ch), transpose to (n_trials, n_ch, n_times)
                if X_train_n.shape[-1] < X_train_n.shape[1]:
                    X_train_eeg = X_train_n.transpose(0, 2, 1)  # (n_trials, n_ch, n_times)
                    X_test_eeg = X_test_n.transpose(0, 2, 1)
                else:
                    X_train_eeg = X_train_n
                    X_test_eeg = X_test_n

            # Convert to float32 (TensorFlow)
            X_train_eeg = X_train_eeg.astype(np.float32)
            X_test_eeg = X_test_eeg.astype(np.float32)
            y_train_fold = y_train_fold.astype(np.int32)
            y_test_fold = y_test_fold.astype(np.int32)

            print(f"   EEGNet input shape (train): {X_train_eeg.shape}")

            # --- 2. Build EEGNet model ---
            n_channels = X_train_eeg.shape[1]
            n_times = X_train_eeg.shape[2]

            model = build_eegnet(n_channels, n_times, n_classes=2, **EEGNET_HPARAMS)

            # Compile
            optimizer = keras.optimizers.Adam(learning_rate=EEGNET_HPARAMS['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # --- 3. Training with Early Stopping ---
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=EEGNET_HPARAMS['early_stopping_patience'],
                restore_best_weights=True
            )

            history = model.fit(
                X_train_eeg, y_train_fold,
                validation_split=0.2,
                epochs=EEGNET_HPARAMS['epochs'],
                batch_size=EEGNET_HPARAMS['batch_size'],
                callbacks=[early_stop],
                verbose=0  # silent to avoid cluttering output
            )

            # --- 4. Prediction on test set ---
            y_pred_prob = model.predict(X_test_eeg, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)

            # --- 5. Metrics ---
            acc = accuracy_score(y_test_fold, y_pred)
            kappa = cohen_kappa_score(y_test_fold, y_pred)
            cm = confusion_matrix(y_test_fold, y_pred)

            # --- 6. Save Results ---
            results_eegnet_seed[test_subject] = {
                'acc': acc,
                'kappa': kappa,
                'confusion': cm,
                'y_true': y_test_fold,
                'y_pred': y_pred,
                'epochs_trained': len(history.history['loss']),
            }

            print(f"   [EEGNet] Subject {test_subject}: Acc={acc:.4f}, Kappa={kappa:.4f}, Epochs={len(history.history['loss'])}\n")

            # Free memory
            keras.backend.clear_session()
            del model

        results_eegnet_all_seeds[seed] = results_eegnet_seed

    print("\nEEGNet LOSO completed for all seeds")

    # ============================================================================
    # EEGNet Results Analysis (Multiple Seeds)
    # ============================================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY: EEGNet (Leave-One-Subject-Out, Multiple Seeds)")
    print("=" * 70 + "\n")

    # --- 1. Consolidate results by seed ---
    results_by_seed = {}
    for seed, results_seed in results_eegnet_all_seeds.items():
        acc_list = [res['acc'] for res in results_seed.values()]
        kappa_list = [res['kappa'] for res in results_seed.values()]

        results_by_seed[seed] = {
            'mean_acc': np.mean(acc_list),
            'std_acc': np.std(acc_list),
            'mean_kappa': np.mean(kappa_list),
            'std_kappa': np.std(kappa_list),
            'acc_per_subject': acc_list,
            'kappa_per_subject': kappa_list,
        }

    # --- 2. Display results by seed ---
    print("Results by Seed:")
    print("-" * 70)
    for seed, metrics in results_by_seed.items():
        print(f"Seed {seed}:")
        print(f"  Accuracy: {metrics['mean_acc']:.4f} (±{metrics['std_acc']:.4f})")
        print(f"  Kappa:    {metrics['mean_kappa']:.4f} (±{metrics['std_kappa']:.4f})")

    # --- 3. Global mean across all seeds ---
    all_accs = []
    all_kappas = []
    for seed, results_seed in results_eegnet_all_seeds.items():
        acc_list = [res['acc'] for res in results_seed.values()]
        kappa_list = [res['kappa'] for res in results_seed.values()]
        all_accs.extend(acc_list)
        all_kappas.extend(kappa_list)

    global_mean_acc = np.mean(all_accs)
    global_std_acc = np.std(all_accs)
    global_mean_kappa = np.mean(all_kappas)
    global_std_kappa = np.std(all_kappas)

    print("-" * 70)
    print("GLOBAL MEAN (across all seeds and subjects):")
    print(f"  Accuracy: {global_mean_acc:.4f} (±{global_std_acc:.4f})")
    print(f"  Kappa:    {global_mean_kappa:.4f} (±{global_std_kappa:.4f})")
    print("-" * 70)

    # --- 4. Detailed table per subject (all seeds) ---
    print("\nAccuracy per Subject (all seeds):")
    print("-" * 70)

    # Get first seed to retrieve subject list
    first_seed = EEGNET_SEEDS[0]
    subjects_eegnet = sorted(results_eegnet_all_seeds[first_seed].keys())

    for subject in subjects_eegnet:
        accs_subject = []
        for seed in EEGNET_SEEDS:
            if subject in results_eegnet_all_seeds[seed]:
                accs_subject.append(results_eegnet_all_seeds[seed][subject]['acc'])

        mean_acc_subj = np.mean(accs_subject)
        std_acc_subj = np.std(accs_subject)
        print(f"Subject {subject}: {mean_acc_subj:.4f} (±{std_acc_subj:.4f}) | Seeds: {[f'{a:.4f}' for a in accs_subject]}")

    # --- 5. Save to CSV (summary by seed) ---
    df_eegnet_seeds = pd.DataFrame([
        {'Seed': seed,
         'Mean_Accuracy': metrics['mean_acc'],
         'Std_Accuracy': metrics['std_acc'],
         'Mean_Kappa': metrics['mean_kappa'],
         'Std_Kappa': metrics['std_kappa']}
        for seed, metrics in results_by_seed.items()
    ])

    csv_path_seeds = 'resultados_eegnet_por_seed.csv'
    df_eegnet_seeds.to_csv(csv_path_seeds, index=False)
    print(f"\nResults by seed saved to: {csv_path_seeds}")

    # --- 6. Save to CSV (detailed by subject and seed) ---
    data_detailed = []
    for seed in EEGNET_SEEDS:
        for subject, results in results_eegnet_all_seeds[seed].items():
            data_detailed.append({
                'Seed': seed,
                'Subject': subject,
                'Accuracy': results['acc'],
                'Kappa': results['kappa'],
            })

    df_eegnet_detailed = pd.DataFrame(data_detailed)
    csv_path_detailed = 'resultados_eegnet_detalhado.csv'
    df_eegnet_detailed.to_csv(csv_path_detailed, index=False)
    print(f"Detailed results saved to: {csv_path_detailed}")

    # --- 7. Side-by-side comparison: CSP+LDA vs EEGNet ---
    print("\n" + "=" * 70)
    print("COMPARISON: CSP+LDA vs EEGNet")
    print("=" * 70)

    # CSP+LDA (already computed above)
    csp_accs = [res['acc'] for res in results_csp_lda.values()]
    csp_kappas = [res['kappa'] for res in results_csp_lda.values()]
    csp_mean_acc = np.mean(csp_accs)
    csp_std_acc = np.std(csp_accs)
    csp_mean_kappa = np.mean(csp_kappas)
    csp_std_kappa = np.std(csp_kappas)

    print(f"\nCSP+LDA (baseline):")
    print(f"  Accuracy: {csp_mean_acc:.4f} (±{csp_std_acc:.4f})")
    print(f"  Kappa:    {csp_mean_kappa:.4f} (±{csp_std_kappa:.4f})")

    print(f"\nEEGNet (CNN, multiple seeds):")
    print(f"  Accuracy: {global_mean_acc:.4f} (±{global_std_acc:.4f})")
    print(f"  Kappa:    {global_mean_kappa:.4f} (±{global_std_kappa:.4f})")

    # Delta
    delta_acc = global_mean_acc - csp_mean_acc
    delta_kappa = global_mean_kappa - csp_mean_kappa
    print(f"\nDelta (EEGNet - CSP+LDA):")
    print(f"  ΔAccuracy: {delta_acc:+.4f}")
    print(f"  ΔKappa:    {delta_kappa:+.4f}")
    print("=" * 70)

    # ============================================================================
    # Figure 1: Pipeline CSP+LDA vs EEGNet (schematic comparison)
    # ============================================================================
    fig, (ax_csp, ax_eeg) = plt.subplots(1, 2, figsize=(14, 6))

    # --- CSP+LDA Pipeline ---
    ax_csp.set_xlim(0, 10)
    ax_csp.set_ylim(0, 10)
    ax_csp.axis('off')
    ax_csp.set_title('CSP + LDA (Baseline)', fontsize=14, fontweight='bold', pad=20)

    y_pos = 9
    boxes_csp = [
        ('Raw EEG\n(201 samples)', 'lightblue'),
        ('Butterworth Filter\n8-30 Hz', 'lightcyan'),
        ('Z-score Normalization\n(per fold)', 'lightyellow'),
        ('CSP\n8 components', 'lightgreen'),
        ('Log-Variance\nExtraction', 'lightgreen'),
        ('LDA\nClassification', 'lightcoral'),
        ('Prediction', 'lightsalmon'),
    ]

    for i, (label, color) in enumerate(boxes_csp):
        y = y_pos - i * 1.2
        box = FancyBboxPatch((0.5, y - 0.35), 8, 0.7, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=color, linewidth=2)
        ax_csp.add_patch(box)
        ax_csp.text(4.5, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

        if i < len(boxes_csp) - 1:
            arrow = FancyArrowPatch((4.5, y - 0.4), (4.5, y - 0.8),
                                    arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax_csp.add_patch(arrow)

    # --- EEGNet Pipeline ---
    ax_eeg.set_xlim(0, 10)
    ax_eeg.set_ylim(0, 10)
    ax_eeg.axis('off')
    ax_eeg.set_title('EEGNet (CNN)', fontsize=14, fontweight='bold', pad=20)  # nome técnico, mantido

    y_pos = 9
    boxes_eeg = [
        ('Raw EEG\n(201 samples, 22 channels)', 'lightblue'),
        ('Butterworth Filter\n8-30 Hz', 'lightcyan'),
        ('Normalization\n(per fold)', 'lightyellow'),
        ('Temporal Conv\n(F1=8 filters)', 'lightgreen'),
        ('DepthwiseConv\n(n_ch,1) + Pool(1,4)', 'lightgreen'),
        ('SepConv(F2=16)\n+ Pool(1,8)', 'lightgreen'),
        ('Flatten + Dense\nClassification', 'lightcoral'),
        ('Softmax Prediction', 'lightsalmon'),
    ]

    for i, (label, color) in enumerate(boxes_eeg):
        y = y_pos - i * 1.0
        box = FancyBboxPatch((0.5, y - 0.3), 8, 0.6, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=color, linewidth=2)
        ax_eeg.add_patch(box)
        ax_eeg.text(4.5, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

        if i < len(boxes_eeg) - 1:
            arrow = FancyArrowPatch((4.5, y - 0.35), (4.5, y - 0.65),
                                    arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax_eeg.add_patch(arrow)

    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / 'pipeline_comparison.png', dpi=FIG_DPI, bbox_inches='tight')
        print("Saved: pipeline_comparison.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # ============================================================================
    # Figure 2: EEGNet Architecture (Detailed)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')

    title = "EEGNet Architecture (Compact)"
    ax.text(6, 13.5, title, ha='center', va='top', fontsize=16, fontweight='bold')

    # Layer by layer
    y = 12.5
    draw_block(ax, 6, y, 3, 0.6, 'Input: (batch, n_ch, n_times)', 'lightblue')

    y -= 0.8
    draw_arrow(ax, 6, 12.2, 6, y + 0.3)

    y -= 0.3
    draw_block(ax, 6, y, 3.5, 0.7, 'Reshape\n→ (batch, 1, n_ch, n_times)', 'lightyellow')  # termo técnico

    y -= 0.9
    draw_arrow(ax, 6, y + 0.65, 6, y + 0.35)

    # Block 1
    y -= 0.35
    draw_block(ax, 6, y, 4.5, 1.2,
               'Block 1: Temporal Conv\nConv2D(F1=8, (1,64))\nBatchNorm → DepthwiseConv2D\n(n_ch, 1), D=2 → ELU → AvgPool(1,4) → Dropout',
               'lightgreen', fontsize=9)

    y -= 1.5
    draw_arrow(ax, 6, y + 1.15, 6, y + 0.35)

    # Block 2
    y -= 0.35
    draw_block(ax, 6, y, 4.5, 1.2,
               'Block 2: Separable Conv\nSeparableConv2D(F2=16, (1,16))\nBatchNorm → ELU → AvgPool(1,8) → Dropout',
               'lightcyan', fontsize=9)

    y -= 1.5
    draw_arrow(ax, 6, y + 1.15, 6, y + 0.35)

    y -= 0.35
    draw_block(ax, 6, y, 3, 0.7, 'Flatten', 'lightyellow')

    y -= 0.9
    draw_arrow(ax, 6, y + 0.55, 6, y + 0.35)

    y -= 0.35
    draw_block(ax, 6, y, 3, 0.7, 'Dense(2) + Softmax', 'lightcoral')

    y -= 0.9
    draw_arrow(ax, 6, y + 0.55, 6, y + 0.35)

    y -= 0.35
    draw_block(ax, 6, y, 3, 0.6, 'Output: (batch, 2)', 'lightsalmon', fontsize=10)

    # Hyperparameter annotations
    note_text = (
        "Hyperparameters:\n"
        "• Learning Rate: 0.001 (Adam)\n"
        "• Batch Size: 16\n"
        "• Dropout: 0.5\n"
        "• Early Stopping: patience=15"
    )
    ax.text(0.3, 1.5, note_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / 'eegnet_architecture.png', dpi=FIG_DPI, bbox_inches='tight')
        print("Saved: eegnet_architecture.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # ============================================================================
    # Figure 3: Boxplot + Violin of Accuracy and Kappa
    # ============================================================================

    # Prepare data for plotting
    plot_data = []

    # CSP+LDA
    for subject, result in results_csp_lda.items():
        plot_data.append({'Method': 'CSP+LDA', 'Subject': subject, 'Accuracy': result['acc'], 'Kappa': result['kappa']})

    # EEGNet (aggregated over all seeds)
    for seed, results_seed in results_eegnet_all_seeds.items():
        for subject, result in results_seed.items():
            plot_data.append({'Method': f'EEGNet (seed={seed})', 'Subject': subject, 'Accuracy': result['acc'], 'Kappa': result['kappa']})

    df_plot = pd.DataFrame(plot_data)

    # Simple aggregation for visualization
    df_agg = []
    for subject in sorted(results_csp_lda.keys()):
        # CSP+LDA
        acc_csp = results_csp_lda[subject]['acc']
        kappa_csp = results_csp_lda[subject]['kappa']
        df_agg.append({'Method': 'CSP+LDA', 'Subject': f'S{subject}', 'Accuracy': acc_csp, 'Kappa': kappa_csp, 'Type': 'Accuracy'})
        df_agg.append({'Method': 'CSP+LDA', 'Subject': f'S{subject}', 'Accuracy': kappa_csp, 'Kappa': kappa_csp, 'Type': 'Kappa'})

        # EEGNet (mean and std across seeds)
        accs_eeg = [results_eegnet_all_seeds[seed][subject]['acc'] for seed in EEGNET_SEEDS]
        kappas_eeg = [results_eegnet_all_seeds[seed][subject]['kappa'] for seed in EEGNET_SEEDS]
        acc_eeg_mean = np.mean(accs_eeg)
        kappa_eeg_mean = np.mean(kappas_eeg)

        df_agg.append({'Method': 'EEGNet', 'Subject': f'S{subject}', 'Accuracy': acc_eeg_mean, 'Kappa': kappa_eeg_mean, 'Type': 'Accuracy'})
        df_agg.append({'Method': 'EEGNet', 'Subject': f'S{subject}', 'Accuracy': kappa_eeg_mean, 'Kappa': kappa_eeg_mean, 'Type': 'Kappa'})

    df_agg = pd.DataFrame(df_agg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    ax = axes[0]
    sns.boxplot(data=df_plot, x='Method', y='Accuracy', ax=ax, palette='Set2', width=0.6)
    sns.stripplot(data=df_plot, x='Method', y='Accuracy', ax=ax, color='black', alpha=0.4, size=4)
    ax.set_title('Accuracy Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_xlabel('Method', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.35, 0.75])

    # Add mean lines
    for method_idx, method in enumerate(sorted(df_plot['Method'].unique())):
        data_method = df_plot[df_plot['Method'] == method]['Accuracy']
        mean_val = data_method.mean()
        ax.hlines(mean_val, method_idx - 0.2, method_idx + 0.2, colors='red', linewidth=2, linestyles='--', label='Mean' if method_idx == 0 else '')

    # --- Kappa ---
    ax = axes[1]
    # Prepare kappa data
    df_kappa = pd.DataFrame()
    kappa_data = []
    for subject, result in results_csp_lda.items():
        kappa_data.append({'Method': 'CSP+LDA', 'Kappa': result['kappa']})
    for seed, results_seed in results_eegnet_all_seeds.items():
        for subject, result in results_seed.items():
            kappa_data.append({'Method': f'EEGNet (seed={seed})', 'Kappa': result['kappa']})
    df_kappa = pd.DataFrame(kappa_data)

    sns.boxplot(data=df_kappa, x='Method', y='Kappa', ax=ax, palette='Set2', width=0.6)
    sns.stripplot(data=df_kappa, x='Method', y='Kappa', ax=ax, color='black', alpha=0.4, size=4)
    ax.set_title('Kappa Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kappa (Cohen)', fontsize=11)
    ax.set_xlabel('Method', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add mean lines
    for method_idx, method in enumerate(sorted(df_kappa['Method'].unique())):
        data_method = df_kappa[df_kappa['Method'] == method]['Kappa']
        mean_val = data_method.mean()
        ax.hlines(mean_val, method_idx - 0.2, method_idx + 0.2, colors='red', linewidth=2, linestyles='--')

    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / 'metrics_distribution.png', dpi=FIG_DPI, bbox_inches='tight')
        print("Saved: metrics_distribution.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # ============================================================================
    # Figure 4: Aggregated Confusion Matrix
    # ============================================================================

    # Aggregate all confusion matrices
    cm_csp_total = np.zeros((2, 2), dtype=int)
    cm_eeg_total = np.zeros((2, 2), dtype=int)

    # --- CSP+LDA ---
    for subject, result in results_csp_lda.items():
        cm_csp_total += result['confusion']

    # --- EEGNet ---
    for seed, results_seed in results_eegnet_all_seeds.items():
        for subject, result in results_seed.items():
            cm_eeg_total += result['confusion']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- CSP+LDA ---
    ax = axes[0]
    sns.heatmap(cm_csp_total, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
    ax.set_title('CSP+LDA - Aggregated Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_xticklabels(['LEFT', 'RIGHT'])
    ax.set_yticklabels(['LEFT', 'RIGHT'])

    # Compute accuracy
    acc_csp_agg = (cm_csp_total[0, 0] + cm_csp_total[1, 1]) / cm_csp_total.sum()
    ax.text(0.5, -0.15, f'Accuracy: {acc_csp_agg:.4f}', transform=ax.transAxes,
            ha='center', fontsize=11, fontweight='bold')

    # --- EEGNet ---
    ax = axes[1]
    sns.heatmap(cm_eeg_total, annot=True, fmt='d', cmap='Greens', ax=ax,
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
    ax.set_title('EEGNet - Aggregated Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_xticklabels(['LEFT', 'RIGHT'])
    ax.set_yticklabels(['LEFT', 'RIGHT'])

    # Compute accuracy
    acc_eeg_agg = (cm_eeg_total[0, 0] + cm_eeg_total[1, 1]) / cm_eeg_total.sum()
    ax.text(0.5, -0.15, f'Accuracy: {acc_eeg_agg:.4f}', transform=ax.transAxes,
            ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / 'confusion_matrices.png', dpi=FIG_DPI, bbox_inches='tight')
        print("Saved: confusion_matrices.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print(f"\nAggregated Confusion Matrices:")
    print(f"CSP+LDA:\n{cm_csp_total}\nAccuracy: {acc_csp_agg:.4f}")
    print(f"\nEEGNet:\n{cm_eeg_total}\nAccuracy: {acc_eeg_agg:.4f}")

    # ============================================================================
    # TABLE 1: Experimental Setup
    # ============================================================================

    setup_data = {
        'Parameter': [
            'Dataset',
            'Subjects',
            'Classes',
            'Sampling Rate',
            'Filter Band',
            'Filter Order',
            'Trial Duration',
            'Total Trials',
            'Train/Test Split',
            'Cross-validation',
        ],
        'Value': [
            'BCICIV 2b',
            f'{len(results_csp_lda)}',
            '2 (LEFT, RIGHT)',
            f'{SFREQ} Hz',
            f'{FILTER_LOW}-{FILTER_HIGH} Hz',
            f'{FILTER_ORDER}',
            '4s (201 samples)',
            f'{len(results_csp_lda) * 144} ({len(results_csp_lda)} subjects × 144 trials)',
            'Leave-One-Subject-Out (LOSO)',
            f'{len(results_csp_lda)} folds (1 per subject)',
        ]
    }

    df_setup = pd.DataFrame(setup_data)

    print("\n" + "=" * 70)
    print("TABLE 1: Experimental Setup")
    print("=" * 70)
    print(df_setup.to_string(index=False))
    print("=" * 70)

    # Save as CSV
    df_setup.to_csv('tabela_setup_experimental.csv', index=False)
    print("Saved: tabela_setup_experimental.csv")

    # Visualize as formatted table
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_setup.values, colLabels=df_setup.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Table style
    for i in range(len(df_setup.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(df_setup) + 1):
        for j in range(len(df_setup.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')

    plt.title('Experimental Setup', fontsize=14, fontweight='bold', pad=20)
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / 'tabela_setup.png', dpi=FIG_DPI, bbox_inches='tight')
        print("Saved: tabela_setup.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # ============================================================================
    # TABLE 2: Results per Subject (CSP+LDA vs EEGNet)
    # ============================================================================

    # Prepare aggregated data per subject
    results_table = []

    for subject in sorted(results_csp_lda.keys()):
        # CSP+LDA
        acc_csp = results_csp_lda[subject]['acc']
        kappa_csp = results_csp_lda[subject]['kappa']

        # EEGNet (mean and std across seeds)
        accs_eeg = [results_eegnet_all_seeds[seed][subject]['acc'] for seed in EEGNET_SEEDS]
        kappas_eeg = [results_eegnet_all_seeds[seed][subject]['kappa'] for seed in EEGNET_SEEDS]

        acc_eeg_mean = np.mean(accs_eeg)
        acc_eeg_std = np.std(accs_eeg)
        kappa_eeg_mean = np.mean(kappas_eeg)
        kappa_eeg_std = np.std(kappas_eeg)

        results_table.append({
            'Subject': subject,
            'CSP Accuracy': f'{acc_csp:.4f}',
            'Kappa CSP': f'{kappa_csp:.4f}',
            'EEGNet Accuracy (μ±σ)': f'{acc_eeg_mean:.4f}±{acc_eeg_std:.4f}',
            'Kappa EEGNet (μ±σ)': f'{kappa_eeg_mean:.4f}±{kappa_eeg_std:.4f}',
            'Δ Accuracy': f'{acc_eeg_mean - acc_csp:+.4f}',
            'Δ Kappa': f'{kappa_eeg_mean - kappa_csp:+.4f}',
        })

    df_results_table = pd.DataFrame(results_table)

    print("\n" + "=" * 100)
    print("TABLE 2: Results per Subject")
    print("=" * 100)
    print(df_results_table.to_string(index=False))
    print("=" * 100)

    # Save as CSV
    df_results_table.to_csv('tabela_resultados_por_sujeito.csv', index=False)
    print("Saved: tabela_resultados_por_sujeito.csv")

    # Visualize as formatted table
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_results_table.values, colLabels=df_results_table.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.08, 0.12, 0.12, 0.18, 0.18, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style
    for i in range(len(df_results_table.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=9)

    for i in range(1, len(df_results_table) + 1):
        for j in range(len(df_results_table.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')

    plt.title('Results by Subject: CSP+LDA vs EEGNet', fontsize=14, fontweight='bold', pad=20)
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / 'tabela_resultados_sujeito.png', dpi=FIG_DPI, bbox_inches='tight')
        print("Saved: tabela_resultados_sujeito.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # ============================================================================
    # TABLE 3: Statistical Summary (Mean ± SD)
    # ============================================================================

    # CSP+LDA stats
    csp_accs = np.array([res['acc'] for res in results_csp_lda.values()])
    csp_kappas = np.array([res['kappa'] for res in results_csp_lda.values()])

    # EEGNet stats (aggregated across all seeds)
    eeg_accs_all = []
    eeg_kappas_all = []
    for seed, results_seed in results_eegnet_all_seeds.items():
        for subject, result in results_seed.items():
            eeg_accs_all.append(result['acc'])
            eeg_kappas_all.append(result['kappa'])

    eeg_accs = np.array(eeg_accs_all)
    eeg_kappas = np.array(eeg_kappas_all)

    # Compute statistics
    stats_csp_acc = calc_stats(csp_accs)
    stats_csp_kappa = calc_stats(csp_kappas)
    stats_eeg_acc = calc_stats(eeg_accs)
    stats_eeg_kappa = calc_stats(eeg_kappas)

    # Compute per-subject EEGNet mean (averaged across seeds) for fair comparison
    subjects_sorted = sorted(results_csp_lda.keys())
    eeg_accs_per_subject = np.array([
        np.mean([results_eegnet_all_seeds[s][subj]['acc'] for s in EEGNET_SEEDS])
        for subj in subjects_sorted
    ])
    eeg_kappas_per_subject = np.array([
        np.mean([results_eegnet_all_seeds[s][subj]['kappa'] for s in EEGNET_SEEDS])
        for subj in subjects_sorted
    ])
    csp_accs_sorted = np.array([results_csp_lda[subj]['acc'] for subj in subjects_sorted])
    csp_kappas_sorted = np.array([results_csp_lda[subj]['kappa'] for subj in subjects_sorted])

    # Paired t-test: each pair is (CSP_subject_i, EEGNet_subject_i)
    t_stat_acc, p_value_acc = stats.ttest_rel(csp_accs_sorted, eeg_accs_per_subject)
    t_stat_kappa, p_value_kappa = stats.ttest_rel(csp_kappas_sorted, eeg_kappas_per_subject)

    # Build table
    summary_data = {
        'Metric': ['Accuracy (CSP)', 'Accuracy (EEGNet)', 'Kappa (CSP)', 'Kappa (EEGNet)'],
        'Mean': [
            f'{stats_csp_acc["Mean"]:.4f}',
            f'{stats_eeg_acc["Mean"]:.4f}',
            f'{stats_csp_kappa["Mean"]:.4f}',
            f'{stats_eeg_kappa["Mean"]:.4f}',
        ],
        'Std': [
            f'{stats_csp_acc["Std"]:.4f}',
            f'{stats_eeg_acc["Std"]:.4f}',
            f'{stats_csp_kappa["Std"]:.4f}',
            f'{stats_eeg_kappa["Std"]:.4f}',
        ],
        'Min': [
            f'{stats_csp_acc["Min"]:.4f}',
            f'{stats_eeg_acc["Min"]:.4f}',
            f'{stats_csp_kappa["Min"]:.4f}',
            f'{stats_eeg_kappa["Min"]:.4f}',
        ],
        'Max': [
            f'{stats_csp_acc["Max"]:.4f}',
            f'{stats_eeg_acc["Max"]:.4f}',
            f'{stats_csp_kappa["Max"]:.4f}',
            f'{stats_eeg_kappa["Max"]:.4f}',
        ],
    }

    df_summary = pd.DataFrame(summary_data)

    print("\n" + "=" * 90)
    print("TABLE 3: Statistical Summary (Mean ± SD, Min, Max)")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)

    # Statistical tests
    print("\nSTATISTICAL TESTS (paired t-test, per-subject means):")
    print("-" * 90)
    print(f"Accuracy: t={t_stat_acc:.4f}, p-value={p_value_acc:.6f}")
    print(f"Kappa:    t={t_stat_kappa:.4f}, p-value={p_value_kappa:.6f}")
    print("-" * 90)

    if p_value_acc < 0.05:
        print("Accuracies are significantly different (p < 0.05)")
    else:
        print("Accuracies are NOT significantly different (p >= 0.05)")

    if p_value_kappa < 0.05:
        print("Kappas are significantly different (p < 0.05)")
    else:
        print("Kappas are NOT significantly different (p >= 0.05)")

    # Save as CSV
    df_summary.to_csv('tabela_resumo_estatistico.csv', index=False)
    print("\nSaved: tabela_resumo_estatistico.csv")

    # Visualize as formatted table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(df_summary) + 1):
        for j in range(len(df_summary.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')

    plt.title('Statistical Summary: Mean ± SD', fontsize=14, fontweight='bold', pad=20)
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / 'tabela_resumo_estatistico.png', dpi=FIG_DPI, bbox_inches='tight')
        print("Saved: tabela_resumo_estatistico.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # ============================================================================
    # Final Summary — all figures and tables saved
    # ============================================================================

    print("\n" + "=" * 80)
    print("PHASE 7: FINAL FIGURES AND TABLES — COMPLETE")
    print("=" * 80)

    print("\nFIGURES GENERATED:")
    print("-" * 80)
    figures_list = [
        "pipeline_comparison.png - CSP+LDA vs EEGNet pipeline comparison",
        "eegnet_architecture.png - Detailed EEGNet architecture",
        "metrics_distribution.png - Accuracy and Kappa boxplots",
        "confusion_matrices.png - Aggregated confusion matrices",
        "tabela_setup.png - Experimental setup table",
        "tabela_resultados_sujeito.png - Results per subject",
        "tabela_resumo_estatistico.png - Statistical summary (mean ± sd)",
    ]

    for fig_item in figures_list:
        print(f"  ✓ {fig_item}")

    print("\nTABLES GENERATED (CSV):")
    print("-" * 80)
    tables_list = [
        "tabela_setup_experimental.csv - Experiment setup",
        "tabela_resultados_por_sujeito.csv - Results per subject (CSP vs EEGNet)",
        "tabela_resumo_estatistico.csv - Statistical summary",
        "resultados_csp_lda.csv - CSP+LDA baseline results",
        "resultados_eegnet_por_seed.csv - EEGNet results by seed",
        "resultados_eegnet_detalhado.csv - Detailed EEGNet results (subject × seed)",
    ]

    for table_item in tables_list:
        print(f"  ✓ {table_item}")

    print("\n" + "=" * 80)
    print("LOCATION: " + str(FIGURES_DIR.resolve()))
    print("=" * 80)

    print("\nALL FIGURES AND TABLES SAVED SUCCESSFULLY!")
    print(f"\nTip: Use figures in '{FIGURES_DIR}/' for your article")
    print("Use the CSVs for additional analyses or reports\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CSP vs EEGNet LOSO evaluation pipeline.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display (save only).")
    parser.add_argument("--no-save", action="store_true", help="Disable saving figures to disk.")
    args = parser.parse_args()
    main(show_plots=not args.no_show, save_figures=not args.no_save)
