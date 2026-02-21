"""
ECG ANALYSIS  — v5 
─────────────────────────────────────────────────────
Install: pip install customtkinter scipy numpy pandas matplotlib neurokit2 openpyxl h5py
Run    : python ecg_app.py
"""

from __future__ import annotations

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading, io, os, zipfile, warnings, logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import scipy.io
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

warnings.filterwarnings("ignore")

# ── NumPy 2.0 shim ────────────────────────────────────────────
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# ── Optional dependencies ──────────────────────────────────────
try:
    import neurokit2 as nk
    NK_OK = True
except ImportError:
    NK_OK = False

try:
    import h5py
    H5_OK = True
except ImportError:
    H5_OK = False

# ── Logging: file + console ────────────────────────────────────
LOG_FILE = Path.home() / "ecg_analysis.log"
_handlers: list[logging.Handler] = [
    logging.FileHandler(LOG_FILE, encoding="utf-8"),
    logging.StreamHandler(),
]
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(funcName)s — %(message)s",
    handlers=_handlers,
)
log = logging.getLogger("ecg")
log.info("ECG Analysis v5 starting  (NK_OK=%s  H5_OK=%s)", NK_OK, H5_OK)

# ════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ════════════════════════════════════════════════════════════

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

BG      = "#FFFFFF"
PANEL   = "#F7F7F7"
CARD    = "#FAFAFA"
BORDER  = "#E4E4E4"
BORDER2 = "#BDBDBD"
RED     = "#D32F2F"
BLUE    = "#1565C0"
GREEN   = "#2E7D32"
ORANGE  = "#E65100"
MUTED   = "#6B6B6B"
LIGHT   = "#ADADAD"
TEXT    = "#1A1A1A"

MP = dict(
    bg="#FFFFFF", axes="#FAFAFA", grid="#EEEEEE",
    text="#2A2A2A", muted="#9E9E9E", border="#D0D0D0",
    signal="#1565C0", raw="#90A4AE",
    rpeak_ok="#2E7D32", rpeak_bad="#BDBDBD",
    threshold="#D32F2F",
)

def F(s: int = 12, b: bool = False) -> tuple:
    return ("Helvetica", s, "bold" if b else "normal")

FT  = F(14, True);  FH  = F(11, True);  FL  = F(11)
FSM = F(10);        FM  = ("Helvetica", 11);  FMS = ("Helvetica", 10)

# ════════════════════════════════════════════════════════════
#  MAT / HDF5 LOADER
# ════════════════════════════════════════════════════════════

def load_mat_signal(
    filepath: str,
    channel: str,
) -> tuple[np.ndarray, str, list[str]]:
    """Load an ECG signal from a MATLAB .mat file (v5/v6 or v7.3 HDF5).

    Parameters
    ----------
    filepath : str   Path to the .mat file.
    channel  : str   Preferred variable name.  Auto-selects if not found.

    Returns
    -------
    signal           : np.ndarray (float64, 1-D)
    detected_channel : str
    all_keys         : list[str]

    Raises
    ------
    ImportError  if MATLAB v7.3 file and h5py is not installed.
    ValueError   if no suitable ECG channel is found.
    """
    def _score(a: np.ndarray) -> int:
        if len(a) < 200 or a.dtype.kind not in "fi":
            return -1
        d = np.diff(a[:500])
        if len(d) > 0 and np.std(d) / (abs(np.mean(d)) + 1e-12) < 0.001:
            return -1  # monotone ramp = time vector
        return len(a) * (2 if a.std() > 1e-4 else 1)

    def _pick(data: dict, want: str) -> tuple[np.ndarray, str, list]:
        keys = sorted(data.keys())
        if want in data and len(data[want]) > 100:
            return data[want].astype(np.float64), want, keys
        best_k = max(data, key=lambda k: _score(data[k]), default=None)
        if best_k and _score(data[best_k]) > 0:
            log.warning("Channel '%s' not found; auto-selected '%s'", want, best_k)
            return data[best_k].astype(np.float64), best_k, keys
        raise ValueError(f"No ECG channel found. Available keys: {keys}")

    # Try MATLAB v5/v6
    try:
        mat = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        flat: dict[str, np.ndarray] = {}
        for k, v in mat.items():
            if k.startswith("_"):
                continue
            try:
                if   hasattr(v, "values"): flat[k] = np.array(v.values).flatten()
                elif hasattr(v, "data"):   flat[k] = np.array(v.data).flatten()
                else:                      flat[k] = np.array(v).flatten()
            except Exception as exc:
                log.debug("Skipping mat key '%s': %s", k, exc)
        return _pick(flat, channel)
    except NotImplementedError:
        pass  # v7.3 — fall through to h5py

    # MATLAB v7.3 (HDF5)
    if not H5_OK:
        raise ImportError(
            "MATLAB v7.3 (HDF5) file detected — h5py is required.\n"
            "Run:  pip install h5py"
        )
    flat = {}
    with h5py.File(filepath, "r") as f:
        for k in f.keys():
            g = f[k]
            found = False
            for sub in ("values", "data"):
                if isinstance(g, h5py.Group) and sub in g:
                    try:
                        a = np.array(g[sub]).flatten()
                        if len(a) > 100:
                            flat[k] = a
                    except Exception as exc:
                        log.debug("HDF5 '%s/%s': %s", k, sub, exc)
                    found = True
                    break
            if not found and isinstance(g, h5py.Dataset):
                try:
                    a = np.array(g).flatten()
                    if len(a) > 100:
                        flat[k] = a
                except Exception as exc:
                    log.debug("HDF5 dataset '%s': %s", k, exc)
    return _pick(flat, channel)


def list_channels(filepath: str) -> str:
    """Return a human-readable listing of all channels in a .mat file."""
    try:
        mat = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        lines = ["Format: MATLAB v5/v6\n"]
        for k, v in mat.items():
            if k.startswith("_"):
                continue
            try:
                if   hasattr(v, "values"): a = np.array(v.values).flatten()
                elif hasattr(v, "data"):   a = np.array(v.data).flatten()
                else:                      a = np.array(v).flatten()
                lines.append(f"  • {k:<30} {len(a):>9,} samples   {a.dtype}")
            except Exception:
                lines.append(f"  • {k}  (non-numeric)")
        return "\n".join(lines)
    except NotImplementedError:
        pass
    if not H5_OK:
        return "MATLAB v7.3 file — install h5py to inspect"
    lines = ["Format: MATLAB v7.3 (HDF5)\n"]
    with h5py.File(filepath, "r") as f:
        for k in f.keys():
            g = f[k]
            found = False
            for sub in ("values", "data"):
                if isinstance(g, h5py.Group) and sub in g:
                    try:
                        a = np.array(g[sub]).flatten()
                        lines.append(
                            f"  • {k:<30} {len(a):>9,} samples   {a.dtype}  ← waveform")
                    except Exception:
                        lines.append(f"  • {k}  (read error)")
                    found = True
                    break
            if not found:
                if isinstance(g, h5py.Dataset):
                    try:
                        a = np.array(g).flatten()
                        lines.append(f"  • {k:<30} {len(a):>9,} samples   {a.dtype}")
                    except Exception:
                        lines.append(f"  • {k}  (read error)")
                else:
                    lines.append(f"  • {k:<30} [group: {list(g.keys())}]")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ════════════════════════════════════════════════════════════

def bandpass(
    signal: np.ndarray,
    fs: float,
    lo: float = 0.5,
    hi: float = 40.0,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    signal : 1-D float array
    fs     : sampling rate (Hz)
    lo, hi : passband edges (Hz)

    Returns the filtered signal unchanged if the passband is invalid.
    """
    nyq = fs / 2
    lo_ = max(0.01, min(lo / nyq, 0.99))
    hi_ = max(0.01, min(hi / nyq, 0.99))
    if lo_ >= hi_:
        log.warning("bandpass: degenerate passband [%.4f, %.4f] — skipped", lo_, hi_)
        return signal
    b, a = butter(3, [lo_, hi_], btype="band")
    return filtfilt(b, a, signal)


def notch(signal: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch filter (default: 50 Hz mains rejection)."""
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, signal)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Return zero-mean unit-variance signal. No-op if std ≈ 0."""
    s = signal.std()
    if s < 1e-10:
        log.warning("normalize: signal std ≈ 0 — returning unchanged")
        return signal
    return (signal - signal.mean()) / s


def ds(arr: np.ndarray, n: int = 6000) -> np.ndarray:
    """Decimate array to at most *n* points for display.

    Uses uniform striding — O(1) — safe for arrays of any size.
    For very long recordings (>1M samples) this keeps the UI responsive
    without chunked/lazy loading.
    """
    if len(arr) <= n:
        return arr
    step = len(arr) // n
    return arr[::step]


def fix_polarity(
    cleaned: np.ndarray,
    fs: float,
    min_dist_ms: float = 250.0,
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """Determine signal polarity and extract all R-peak candidates.

    ECG signals recorded via Spike2 are sometimes inverted (R-peaks point
    downwards).  This function tests both polarities and scores each by
    counting peaks whose prominence exceeds 50 % of the median prominence.
    The polarity with the higher score is returned.

    This is called ONCE after filtering.  The returned candidate array and
    prominence array are cached and reused by apply_threshold() on every
    slider change, avoiding repeated peak detection.

    Parameters
    ----------
    cleaned     : normalised ECG signal (1-D, float64)
    fs          : sampling rate (Hz)
    min_dist_ms : minimum R-R distance in milliseconds (default 250 ms = 240 bpm max)

    Returns
    -------
    signal_out  : polarity-corrected signal
    inverted    : True if the signal was flipped
    candidates  : np.ndarray[int]   — all candidate peak indices
    prominences : np.ndarray[float] — corresponding prominence values
    """
    min_dist = max(1, int(min_dist_ms / 1000 * fs))

    def _cands(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, props = find_peaks(sig, distance=min_dist, prominence=0)
        return c, props.get("prominences", np.array([]))

    cp, pp = _cands(cleaned)
    cn, pn = _cands(-cleaned)

    def _score(c: np.ndarray, p: np.ndarray) -> int:
        return int((p > np.median(p) * 0.5).sum()) if len(p) else 0

    if _score(cn, pn) > _score(cp, pp) * 1.15:
        log.debug("fix_polarity: inverted signal detected — flipping")
        return -cleaned, True, cn, pn
    return cleaned, False, cp, pp


def apply_threshold(
    cleaned: np.ndarray,
    cands: np.ndarray,
    proms: np.ndarray,
    thresh_frac: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Filter pre-computed peak candidates by a prominence threshold.

    This is a pure array operation (no peak detection) — runs in < 1 ms
    even for very long recordings, making the threshold slider feel instant.

    Parameters
    ----------
    cleaned     : polarity-corrected, normalised ECG signal
    cands       : all candidate peak indices (from fix_polarity)
    proms       : corresponding prominences (from fix_polarity)
    thresh_frac : sensitivity multiplier applied to the 75th-percentile
                  prominence.  0 → accept everything;  1 → balanced;
                  > 1 → increasingly strict.

    Returns
    -------
    accepted    : np.ndarray[int]   — accepted R-peak indices
    rejected    : np.ndarray[int]   — rejected candidate indices (shown grey)
    thresh_amp  : float             — signal amplitude at the threshold
                                     (y-coordinate of the horizontal guide line)
    """
    if len(cands) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0.0

    ref_prom    = np.percentile(proms, 75)
    thresh_prom = thresh_frac * ref_prom
    mask        = proms >= thresh_prom
    accepted    = cands[mask]
    rejected    = cands[~mask]

    thresh_amp = (
        float(np.percentile(cleaned[accepted], 5)) if len(accepted) > 0
        else float(thresh_frac * np.percentile(cleaned[cands], 75))
    )

    log.debug("apply_threshold: frac=%.3f → %d accepted / %d rejected",
              thresh_frac, len(accepted), len(rejected))
    return accepted, rejected, thresh_amp


def detect_peaks_threshold(
    cleaned: np.ndarray,
    fs: float,
    thresh_frac: float,
    min_dist_ms: float = 250.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    """One-shot convenience wrapper: fix polarity then apply threshold.

    Used only for the initial preview load.  Subsequent threshold changes
    call apply_threshold() directly with the cached candidates.
    """
    cleaned_out, inverted, cands, proms = fix_polarity(cleaned, fs, min_dist_ms)
    accepted, rejected, thresh_amp = apply_threshold(cleaned_out, cands, proms, thresh_frac)
    return accepted, rejected, thresh_amp, cleaned_out, inverted


def run_full_analysis(
    cleaned: np.ndarray,
    rpeaks: np.ndarray,
    fs: float,
) -> dict:
    """Run the complete HRV and interval analysis pipeline.

    Parameters
    ----------
    cleaned : np.ndarray (float64, 1-D)
        Normalised, polarity-corrected ECG signal.
    rpeaks  : np.ndarray (int, 1-D)
        R-peak sample indices.  Minimum 5 peaks required.
    fs      : float
        Sampling frequency in Hz.

    Returns
    -------
    dict with keys:
        hr           dict       — mean/min/max/std HR (bpm) + n beats
        rr_ms        ndarray    — all RR intervals (ms), unfiltered
        rr_df        DataFrame  — physiologically filtered RR timeseries
        hrv_time     DataFrame  — NeuroKit2 time-domain HRV metrics
        hrv_freq     DataFrame  — frequency-domain HRV (empty if failed)
        hrv_nonlin   DataFrame  — non-linear HRV (empty if failed)
        intervals    DataFrame  — PR / QRS / QT / QTc per beat
        beat_template ndarray | None  — average beat waveform
        beat_time     ndarray | None  — time axis in ms (-400 … +400)

    Raises
    ------
    RuntimeError  if neurokit2 is not installed.
    ValueError    if fewer than 3 R-peaks are provided.
    """
    if not NK_OK:
        raise RuntimeError(
            "neurokit2 is required for full analysis.\n"
            "Install it:  pip install neurokit2"
        )

    rpeaks = np.sort(np.array(rpeaks, dtype=int).flatten())
    if len(rpeaks) < 3:
        raise ValueError(f"Too few R-peaks ({len(rpeaks)}) — need at least 3.")

    rr_ms = np.diff(rpeaks).astype(np.float64) / fs * 1000
    log.info("run_full_analysis: %d peaks, %.1f s, mean RR=%.1f ms",
             len(rpeaks), len(cleaned) / fs, float(rr_ms.mean()))

    # ── Time-domain HRV ──────────────────────────────────────
    hrv_t = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)

    # ── Frequency-domain HRV ─────────────────────────────────
    try:
        hrv_f = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False, normalize=True)
    except Exception as exc:
        log.warning("hrv_frequency failed: %s", exc)
        hrv_f = pd.DataFrame()

    # ── Non-linear HRV ───────────────────────────────────────
    try:
        hrv_nl = nk.hrv_nonlinear(rpeaks, sampling_rate=fs, show=False)
    except Exception as exc:
        log.warning("hrv_nonlinear failed: %s", exc)
        hrv_nl = pd.DataFrame()

    # ── RR timeseries ─────────────────────────────────────────
    rr_df = pd.DataFrame({
        "Time_s": rpeaks[1:] / fs,
        "RR_ms":  rr_ms,
        "HR_bpm": 60000.0 / rr_ms,
    })
    rr_df = rr_df[(rr_df.RR_ms > 240) & (rr_df.RR_ms < 3000)].copy()
    log.debug("rr_df: %d / %d intervals kept after physiological filter",
              len(rr_df), len(rr_ms))

    hr = {
        "mean": float(np.nanmean(60000 / rr_ms)),
        "min":  float(60000 / np.percentile(rr_ms, 98)),
        "max":  float(60000 / np.percentile(rr_ms, 2)),
        "std":  float(np.nanstd(60000 / rr_ms)),
        "n":    int(len(rpeaks)),
    }

    # ── ECG interval delineation ──────────────────────────────
    try:
        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt")

        def gw(k: str) -> np.ndarray:
            return np.array(waves.get(k, [np.nan] * len(rpeaks)), dtype=float)

        rf  = rpeaks.astype(float)
        n   = min(len(rf), len(gw("ECG_T_Offsets")), len(gw("ECG_P_Onsets"))) - 1
        ivl = pd.DataFrame({
            "RR_ms":  rr_ms[:n],
            "PR_ms":  (rf[:n] - gw("ECG_P_Onsets")[:n]) / fs * 1000,
            "QRS_ms": (gw("ECG_S_Peaks")[:n] - gw("ECG_Q_Peaks")[:n]) / fs * 1000,
            "QT_ms":  (gw("ECG_T_Offsets")[:n] - rf[:n]) / fs * 1000,
        }).assign(
            QTc_ms=lambda d: d.QT_ms / np.sqrt(np.clip(d.RR_ms, 1, None) / 1000)
        ).dropna()
        log.info("ECG delineation: %d beats with complete P/Q/S/T", len(ivl))
    except Exception as exc:
        log.warning("ECG delineation failed (may lack clear P/Q/S/T): %s", exc)
        ivl = pd.DataFrame({"RR_ms": rr_ms})

    # ── Average beat template ─────────────────────────────────
    beat_template: Optional[np.ndarray] = None
    beat_time:     Optional[np.ndarray] = None
    try:
        hw = int(0.4 * fs)    # ±400 ms window
        templates = [
            cleaned[rp - hw: rp + hw]
            for rp in rpeaks
            if rp - hw >= 0 and rp + hw < len(cleaned)
        ]
        if templates:
            beat_template = np.mean(templates, axis=0)
            beat_time     = np.arange(-hw, hw) / fs * 1000
            log.debug("Beat template: %d beats averaged", len(templates))
    except Exception as exc:
        log.warning("Beat template failed: %s", exc)

    return dict(
        hr=hr, rr_ms=rr_ms, rr_df=rr_df,
        hrv_time=hrv_t, hrv_freq=hrv_f, hrv_nonlin=hrv_nl,
        intervals=ivl, beat_template=beat_template, beat_time=beat_time,
    )

# ════════════════════════════════════════════════════════════
#  FIGURE FACTORY
# ════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MP["border"], "axes.labelcolor": MP["text"],
    "xtick.color": MP["muted"], "ytick.color": MP["muted"],
    "figure.facecolor": MP["bg"], "axes.facecolor": MP["axes"],
    "grid.color": MP["grid"], "grid.linewidth": 0.7,
    "lines.antialiased": True,
})

def mfig(w=12, h=4, rows=1, cols=1, **kw):
    fig, ax = plt.subplots(rows, cols, figsize=(w, h), **kw)
    fig.patch.set_facecolor(MP["bg"])
    for a in (np.array(ax).flatten() if hasattr(ax, "__iter__") else [ax]):
        a.set_facecolor(MP["axes"]); a.grid(True)
    fig.tight_layout(pad=2.0)
    return fig, ax

def ds(arr, n=6000):
    """Downsample for display."""
    if len(arr) <= n: return arr
    return arr[::len(arr)//n]


def _style_ax(ax):
    """Apply standard white-theme style to any matplotlib Axes."""
    ax.set_facecolor(MP["axes"])
    ax.grid(True, color=MP["grid"], lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MP["border"])
    ax.spines["bottom"].set_color(MP["border"])
    ax.tick_params(colors=MP["muted"])
    ax.xaxis.label.set_color(MP["text"])
    ax.yaxis.label.set_color(MP["text"])
    ax.title.set_color(MP["text"])


# ════════════════════════════════════════════════════════════
#  CANVAS MANAGER
# ════════════════════════════════════════════════════════════

class CanvasSlot:
    """
    Wraps a matplotlib canvas. The Figure is created ONCE and reused forever.
    update() receives a draw-function that populates the figure axes.
    This avoids all tkinter threading issues caused by figure GC.
    """
    def __init__(self, parent, w, h, toolbar=True):
        self.frame  = ctk.CTkFrame(parent, fg_color="transparent")
        self.frame.pack(fill="both", expand=True)
        self.w, self.h = w, h
        # Create the permanent figure
        self.fig = plt.Figure(figsize=(w, h), facecolor=MP["bg"],
                              tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=2)
        if toolbar:
            tb = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)
            tb.configure(background=PANEL)
            tb.pack(side="bottom", fill="x", padx=6)
        self._show_placeholder()

    def _show_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(MP["axes"])
        ax.text(0.5, 0.5, "Run analysis to display",
                ha="center", va="center",
                color=MP["muted"], fontsize=10,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(MP["border"])
        self.canvas.draw_idle()

    def update(self, draw_fn):
        """
        draw_fn(fig) — receives the permanent Figure, should clear and repopulate it.
        Called from main thread only. Never creates a new Figure.
        """
        self.fig.clear()
        self.fig.patch.set_facecolor(MP["bg"])
        try:
            draw_fn(self.fig)
        except Exception as e:
            log.exception("CanvasSlot draw error in slot")
            ax = self.fig.add_subplot(111)
            ax.set_facecolor(MP["axes"])
            ax.text(0.5, 0.5, f"Draw error:\n{e}",
                    ha="center", va="center", color="#D32F2F",
                    fontsize=9, transform=ax.transAxes, wrap=True)
        self.canvas.draw_idle()


# ════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════

class ECGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ECG Analysis")
        self.geometry("1480x900")
        self.minsize(1200, 750)
        self.configure(fg_color=BG)

        # State
        self._filepath   = None
        self._signal_raw = None   # raw float64 array
        self._signal_flt = None   # filtered/cleaned for detection
        self._time       = None
        self._fs         = 1000
        self._rpeaks_ok  = None   # accepted peaks (numpy int array)
        self._rpeaks_rej = None   # rejected candidates
        self._results    = None   # full HRV results dict
        self._thresh_amp = 0.0    # for display
        self._recent     = []     # recent file paths
        self._dark_mode  = False  # theme toggle
        self._sig_quality= None   # 0-100 quality score
        self._all_cands  = None   # all peak candidates (fixed after polarity)
        self._all_proms  = None   # their prominences

        # ── Display cache (invalidated on file load) ─────────
        # Downsampled arrays so _draw_overview/_draw_detail
        # don't re-stride the full signal on every redraw.
        self._ds_time:   np.ndarray | None = None  # ds(time, 5000)
        self._ds_sig:    np.ndarray | None = None  # ds(signal, 5000)

        self._slots: dict[str, CanvasSlot] = {}
        self._containers: dict[str, ctk.CTkFrame] = {}
        self._epoch_df = None
        self._build()
        self.after(200, self._setup_dnd)

    # ════════════════════════════════════════════════════════
    #  BUILD UI
    # ════════════════════════════════════════════════════════

    def _build(self):
        self.sidebar = ctk.CTkFrame(self, width=280, fg_color=PANEL,
                                     corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.pack(side="left", fill="both", expand=True)
        self._build_kpi(main)
        self._build_tabs(main)

    # ─── SIDEBAR ─────────────────────────────────────────────

    def _build_sidebar(self):
        # Wrap everything in a scrollable frame so nothing gets cut off
        scroll = ctk.CTkScrollableFrame(self.sidebar, fg_color=PANEL,
                                         scrollbar_button_color=BORDER,
                                         scrollbar_button_hover_color=BORDER2)
        scroll.pack(fill="both", expand=True)
        s = scroll
        PX = dict(padx=16)

        # Title
        ctk.CTkLabel(s, text="ECG Analysis", font=FT,
                     text_color=TEXT, anchor="w").pack(**PX, pady=(20,2), fill="x")
        ctk.CTkLabel(s, text="Spike2  ·  .mat", font=FSM,
                     text_color=MUTED, anchor="w").pack(**PX, pady=(0,10), fill="x")
        self._sep(s)

        # ── FILE ──────────────────────────────────────────────
        self._hdr(s, "FILE")
        self.lbl_file = ctk.CTkLabel(s, text="No file loaded", font=FSM,
                                      text_color=MUTED, wraplength=245,
                                      anchor="w", justify="left")
        self.lbl_file.pack(**PX, pady=(0,6), fill="x")

        ctk.CTkButton(s, text="Open .mat file", command=self._open_file,
                      fg_color=BLUE, hover_color="#0D47A1", text_color="white",
                      font=FL, height=34, corner_radius=5
                      ).pack(**PX, fill="x", pady=(0,4))
        ctk.CTkButton(s, text="🔑  Show channels", command=self._show_channels,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FSM, height=28, corner_radius=5
                      ).pack(**PX, fill="x", pady=(0,4))
        self.btn_recent = ctk.CTkButton(s, text="🕒  Recent files",
                      command=self._open_recent,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FSM, height=28, corner_radius=5)
        self.btn_recent.pack(**PX, fill="x", pady=(0,4))
        ctk.CTkLabel(s, text="or drag & drop a .mat file onto the window",
                     font=F(9), text_color=LIGHT,
                     anchor="w", wraplength=245).pack(**PX, pady=(0,4), fill="x")

        self._entry(s, "Channel name",   "channel", "ECG",        PX)
        self._entry(s, "Subject ID",     "subject", "subject_01", PX)
        self._sep(s)

        # ── SIGNAL ────────────────────────────────────────────
        self._hdr(s, "SIGNAL")
        self._entry(s, "Sampling rate (Hz)", "fs", "1000", PX)

        row = ctk.CTkFrame(s, fg_color="transparent")
        row.pack(**PX, fill="x", pady=(0,8))
        for lbl, attr, val in [("Start (s)","t_start","0"),
                                 ("End (s)",  "t_end",  "0")]:
            col = ctk.CTkFrame(row, fg_color="transparent")
            col.pack(side="left", fill="x", expand=True,
                     padx=(0,6) if lbl.startswith("S") else 0)
            ctk.CTkLabel(col, text=lbl, font=FSM, text_color=MUTED).pack(anchor="w")
            e = ctk.CTkEntry(col, font=FL, height=28, fg_color=BG,
                              border_color=BORDER2, text_color=TEXT)
            e.insert(0, val); e.pack(fill="x")
            setattr(self, f"ent_{attr}", e)
        self._sep(s)

        # ── FILTERS ───────────────────────────────────────────
        self._hdr(s, "FILTERS")
        row2 = ctk.CTkFrame(s, fg_color="transparent")
        row2.pack(**PX, fill="x", pady=(0,6))
        for lbl, attr, val in [("LP cut (Hz)","lp","0.5"),
                                 ("HP cut (Hz)","hp","40")]:
            col = ctk.CTkFrame(row2, fg_color="transparent")
            col.pack(side="left", fill="x", expand=True,
                     padx=(0,6) if lbl.startswith("L") else 0)
            ctk.CTkLabel(col, text=lbl, font=FSM, text_color=MUTED).pack(anchor="w")
            e = ctk.CTkEntry(col, font=FL, height=28, fg_color=BG,
                              border_color=BORDER2, text_color=TEXT)
            e.insert(0, val); e.pack(fill="x")
            setattr(self, f"ent_{attr}", e)

        self.sw_notch = ctk.CTkSwitch(s, text="Notch 50 Hz", font=FL,
                                       text_color=MUTED, progress_color=BLUE,
                                       button_color=BORDER2)
        self.sw_notch.pack(**PX, anchor="w", pady=(0,4))

        self.cb_clean = ctk.CTkComboBox(s, font=FL, height=28,
                                         fg_color=BG, border_color=BORDER2,
                                         button_color=BORDER2, text_color=TEXT,
                                         dropdown_fg_color=BG, dropdown_text_color=TEXT,
                                         values=["neurokit","pantompkins1985",
                                                 "elgendi2010","hamilton2002","biosppy"])
        self.cb_clean.set("neurokit")
        self.cb_clean.pack(**PX, fill="x", pady=(0,4))
        self._sep(s)

        # ── DETECTION ─────────────────────────────────────────
        self._hdr(s, "DETECTION")

        ctk.CTkLabel(s, text="Min R-R distance (ms)", font=FSM,
                     text_color=MUTED, anchor="w").pack(**PX, fill="x")
        self.ent_minrr = ctk.CTkEntry(s, font=FL, height=28, fg_color=BG,
                                       border_color=BORDER2, text_color=TEXT)
        self.ent_minrr.insert(0, "250"); self.ent_minrr.pack(**PX, fill="x", pady=(2,8))

        # Threshold — continuous slider + precise entry field
        self.lbl_thr = ctk.CTkLabel(s, text="Sensitivity:  0.50",
                                     font=FSM, text_color=MUTED, anchor="w")
        self.lbl_thr.pack(**PX, fill="x")
        ctk.CTkLabel(s, text="↑ strict  (fewer)          ↓ sensitive  (more)",
                     font=F(9), text_color=LIGHT, anchor="w").pack(**PX, fill="x")
        # Slider — continuous, no steps
        self.sl_thr = ctk.CTkSlider(s, from_=0.01, to=2.0,
                                     progress_color=RED, button_color=RED,
                                     fg_color=BORDER, command=self._on_thr)
        self.sl_thr.set(0.50); self.sl_thr.pack(**PX, fill="x", pady=(2,4))
        # Precise entry
        thr_row = ctk.CTkFrame(s, fg_color="transparent")
        thr_row.pack(**PX, fill="x", pady=(0,8))
        ctk.CTkLabel(thr_row, text="Exact value:", font=FSM,
                     text_color=MUTED).pack(side="left")
        self.ent_thr = ctk.CTkEntry(thr_row, width=70, height=26, font=FL,
                                     fg_color=BG, border_color=BORDER2,
                                     text_color=TEXT)
        self.ent_thr.insert(0, "0.50")
        self.ent_thr.pack(side="left", padx=6)
        self.ent_thr.bind("<Return>",   self._on_thr_entry)
        self.ent_thr.bind("<FocusOut>", self._on_thr_entry)

        # LIVE peak count
        self.lbl_npeaks = ctk.CTkLabel(s, text="Peaks detected: —",
                                        font=F(12, True), text_color=BLUE, anchor="w")
        self.lbl_npeaks.pack(**PX, fill="x", pady=(0,6))

        self.sw_art = ctk.CTkSwitch(s, text="Artifact correction", font=FL,
                                     text_color=MUTED, progress_color=BLUE,
                                     button_color=BORDER2)
        self.sw_art.select(); self.sw_art.pack(**PX, anchor="w", pady=(0,4))
        self._sep(s)

        # ── EPOCH ANALYSIS ────────────────────────────────────
        self._hdr(s, "EPOCH ANALYSIS")
        row_ep = ctk.CTkFrame(s, fg_color="transparent")
        row_ep.pack(**PX, fill="x", pady=(0,4))
        ctk.CTkLabel(row_ep, text="Epoch (s)", font=FSM,
                     text_color=MUTED).pack(side="left")
        self.ent_epoch = ctk.CTkEntry(row_ep, width=60, height=26,
                                       font=FL, fg_color=BG,
                                       border_color=BORDER2, text_color=TEXT)
        self.ent_epoch.insert(0, "300"); self.ent_epoch.pack(side="left", padx=6)
        ctk.CTkLabel(row_ep, text="overlap (s)", font=FSM,
                     text_color=MUTED).pack(side="left")
        self.ent_overlap = ctk.CTkEntry(row_ep, width=50, height=26,
                                         font=FL, fg_color=BG,
                                         border_color=BORDER2, text_color=TEXT)
        self.ent_overlap.insert(0, "0"); self.ent_overlap.pack(side="left", padx=4)
        self.sw_epoch = ctk.CTkSwitch(s, text="Segment into epochs", font=FL,
                                       text_color=MUTED, progress_color=BLUE,
                                       button_color=BORDER2)
        self.sw_epoch.pack(**PX, anchor="w", pady=(0,4))
        self._sep(s)

        # ── ACTIONS ───────────────────────────────────────────
        self.btn_preview = ctk.CTkButton(s, text="▶  Preview Detection",
                                          command=self._preview,
                                          fg_color=BLUE, hover_color="#0D47A1",
                                          text_color="white", font=F(12,True),
                                          height=36, corner_radius=5)
        self.btn_preview.pack(**PX, fill="x", pady=(0,6))

        btn_run_color = RED if NK_OK else BORDER2
        btn_run_hover = "#B71C1C" if NK_OK else BORDER2
        self.btn_run = ctk.CTkButton(s, text="⚡  Run Full Analysis",
                                      command=self._run,
                                      fg_color=btn_run_color,
                                      hover_color=btn_run_hover,
                                      text_color="white", font=F(13,True),
                                      height=40, corner_radius=5,
                                      state="normal" if NK_OK else "disabled")
        self.btn_run.pack(**PX, fill="x", pady=(0,6))
        if not NK_OK:
            ctk.CTkLabel(s, text="⚠ pip install neurokit2",
                         font=F(9), text_color=ORANGE,
                         anchor="w").pack(**PX, fill="x", pady=(0,4))
        if not H5_OK:
            ctk.CTkLabel(s, text="ℹ pip install h5py  (for .mat v7.3)",
                         font=F(9), text_color=MUTED,
                         anchor="w").pack(**PX, fill="x", pady=(0,4))

        self.btn_xl = ctk.CTkButton(s, text="Export Excel",
                                     command=self._export_excel,
                                     fg_color=BORDER, hover_color=BORDER2,
                                     text_color=MUTED, font=FL, height=30,
                                     corner_radius=5)
        self.btn_xl.pack(**PX, fill="x", pady=2)
        self.btn_zip = ctk.CTkButton(s, text="Export ZIP  (Excel + Figures)",
                                      command=self._export_zip,
                                      fg_color=BORDER, hover_color=BORDER2,
                                      text_color=MUTED, font=FL, height=30,
                                      corner_radius=5)
        self.btn_zip.pack(**PX, fill="x", pady=2)

        self._sep(s)
        self.lbl_status = ctk.CTkLabel(s, text="Ready", font=FSM,
                                        text_color=MUTED, anchor="w",
                                        wraplength=248, justify="left")
        self.lbl_status.pack(**PX, pady=4, fill="x")

    # ─── KPI BAR ──────────────────────────────────────────────

    def _build_kpi(self, parent):
        bar = ctk.CTkFrame(parent, fg_color=PANEL, height=70, corner_radius=0)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        ctk.CTkFrame(bar, height=1, fg_color=BORDER).pack(side="bottom", fill="x")

        # Progress bar (hidden when idle)
        self.progress = ctk.CTkProgressBar(bar, height=3, mode="indeterminate",
                                            progress_color=BLUE, fg_color=BORDER)
        self.progress.pack(side="bottom", fill="x")
        self.progress.pack_forget()

        self._kpi = {}
        for lbl, key in [("HR Mean","hr_mean"),("HR Range","hr_range"),
                          ("Mean RR","rr_mean"), ("N Beats","n_beats"),
                          ("SDNN",   "sdnn"),    ("RMSSD","rmssd"),
                          ("pNN50",  "pnn50"),   ("Duration","dur")]:
            f = ctk.CTkFrame(bar, fg_color="transparent")
            f.pack(side="left", padx=16, pady=8)
            ctk.CTkFrame(bar, width=1, fg_color=BORDER).pack(
                side="left", fill="y", pady=14)
            ctk.CTkLabel(f, text=lbl, font=F(9), text_color=MUTED).pack(anchor="w")
            v = ctk.CTkLabel(f, text="—", font=F(18, True), text_color=TEXT)
            v.pack(anchor="w")
            self._kpi[key] = v

        # Right-side tools
        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.pack(side="right", padx=12)
        self.lbl_quality = ctk.CTkLabel(right, text="", font=F(10,True),
                                         text_color=MUTED)
        self.lbl_quality.pack(anchor="e", pady=(6,2))
        ctk.CTkButton(right, text="☀ / ☾", width=56, height=24,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=F(10), command=self._toggle_dark,
                      corner_radius=4).pack(anchor="e")

    # ─── TABS ─────────────────────────────────────────────────

    def _build_tabs(self, parent):
        self.tabs = ctk.CTkTabview(
            parent, fg_color=BG,
            segmented_button_fg_color=PANEL,
            segmented_button_selected_color=RED,
            segmented_button_selected_hover_color="#B71C1C",
            segmented_button_unselected_color=PANEL,
            segmented_button_unselected_hover_color=BORDER,
            text_color=TEXT, text_color_disabled=MUTED,
        )
        self.tabs.pack(fill="both", expand=True)
        for t in ["Detection","RR / HR","HRV","Non-linear",
                  "Intervals","Beat Template","Epochs","Summary"]:
            self.tabs.add(t)

        self._build_tab_detect()
        self._build_tab_rr()
        self._build_tab_hrv()
        self._build_tab_nonlin()
        self._build_tab_intervals()
        self._build_tab_beat()
        self._build_tab_epochs()
        self._build_tab_summary()

    # ─── TAB: DETECTION ───────────────────────────────────────

    def _build_tab_detect(self):
        t = self.tabs.tab("Detection")

        # Top: overview plot (full signal)
        ctk.CTkLabel(t, text="SIGNAL OVERVIEW", font=F(9,True),
                     text_color=MUTED, anchor="w").pack(anchor="w", padx=10, pady=(6,0))
        self._slots["overview"] = CanvasSlot(t, 14, 2.2, toolbar=False)

        # Divider
        ctk.CTkFrame(t, height=1, fg_color=BORDER).pack(fill="x", padx=8, pady=4)

        # Bottom: detail view with threshold line
        hdr = ctk.CTkFrame(t, fg_color="transparent")
        hdr.pack(fill="x", padx=10)
        ctk.CTkLabel(hdr, text="DETAIL VIEW  —  threshold line & detected peaks",
                     font=F(9,True), text_color=MUTED).pack(side="left", anchor="w")
        # Navigation
        ctk.CTkLabel(hdr, text="Navigate:", font=FSM,
                     text_color=MUTED).pack(side="left", padx=(30,4))
        ctk.CTkButton(hdr, text="◀", width=32, height=26, font=FL,
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      command=lambda: self._nav(-1)
                      ).pack(side="left", padx=2)
        ctk.CTkButton(hdr, text="▶", width=32, height=26, font=FL,
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      command=lambda: self._nav(+1)
                      ).pack(side="left", padx=2)
        ctk.CTkButton(hdr, text="Reset", width=54, height=26, font=FSM,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      command=self._nav_reset
                      ).pack(side="left", padx=2)

        ctk.CTkLabel(hdr, text="Window:", font=FSM,
                     text_color=MUTED).pack(side="left", padx=(16,4))
        self.ent_window = ctk.CTkEntry(hdr, width=52, height=26, font=FL,
                                        fg_color=BG, border_color=BORDER2,
                                        text_color=TEXT)
        self.ent_window.insert(0, "10"); self.ent_window.pack(side="left")
        ctk.CTkLabel(hdr, text="s", font=FSM, text_color=MUTED).pack(side="left", padx=2)

        self._slots["detail"] = CanvasSlot(t, 14, 3.6, toolbar=False)
        self._nav_pos = 0.0   # current view start in seconds

    # ─── TAB: RR / HR ─────────────────────────────────────────

    def _build_tab_rr(self):
        t = self.tabs.tab("RR / HR")
        # Top: full-width tachogram
        self._slots["rr"] = CanvasSlot(t, 14, 4.2)
        # Bottom: stats table + histogram side by side
        row = ctk.CTkFrame(t, fg_color="transparent")
        row.pack(fill="both", expand=True, padx=8, pady=(0,6))
        lf = ctk.CTkFrame(row, fg_color=CARD, corner_radius=6)
        lf.pack(side="left", fill="both", expand=True, padx=(0,6))
        ctk.CTkLabel(lf, text="RR STATISTICS", font=F(10,True),
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        self.txt_rr = self._textbox(lf, -1, padx=10, expand=True)
        rf = ctk.CTkFrame(row, fg_color=CARD, corner_radius=6)
        rf.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(rf, text="DISTRIBUTION", font=F(10,True),
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        self._slots["rr_hist"] = CanvasSlot(rf, 6, 4)

    # ─── TAB: HRV ─────────────────────────────────────────────

    def _build_tab_hrv(self):
        t = self.tabs.tab("HRV")
        row = ctk.CTkFrame(t, fg_color="transparent")
        row.pack(fill="both", expand=True, padx=8, pady=6)

        # Left column — tables
        lf = ctk.CTkFrame(row, fg_color=CARD, corner_radius=6)
        lf.pack(side="left", fill="both", expand=True, padx=(0,6))
        ctk.CTkLabel(lf, text="TIME DOMAIN", font=F(10,True),
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        self.txt_td = self._textbox(lf, -1, padx=10, expand=True)
        ctk.CTkLabel(lf, text="FREQUENCY DOMAIN  (%)", font=F(10,True),
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        self.txt_fd = self._textbox(lf, -1, padx=10, expand=True)

        # Right column — PSD on top, radar below
        rf = ctk.CTkFrame(row, fg_color="transparent")
        rf.pack(side="left", fill="both", expand=True)
        top_r = ctk.CTkFrame(rf, fg_color=CARD, corner_radius=6)
        top_r.pack(fill="both", expand=True, pady=(0,6))
        ctk.CTkLabel(top_r, text="POWER SPECTRAL DENSITY  (Welch)", font=F(10,True),
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        self._slots["psd"] = CanvasSlot(top_r, 7, 3.8)
        bot_r = ctk.CTkFrame(rf, fg_color=CARD, corner_radius=6)
        bot_r.pack(fill="both", expand=True)
        ctk.CTkLabel(bot_r, text="HRV PROFILE  (normalised)", font=F(10,True),
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        self._slots["radar"] = CanvasSlot(bot_r, 7, 3.8)

    # ─── TAB: NON-LINEAR ──────────────────────────────────────

    def _build_tab_nonlin(self):
        t = self.tabs.tab("Non-linear")
        row = ctk.CTkFrame(t, fg_color="transparent")
        row.pack(fill="both", expand=True, padx=8, pady=6)

        lf = ctk.CTkFrame(row, fg_color=CARD, corner_radius=6)
        lf.pack(side="left", fill="both", expand=True, padx=(0,6))
        ctk.CTkLabel(lf, text="NON-LINEAR METRICS", font=F(10,True),
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        self.txt_nl = self._textbox(lf, -1, padx=10, expand=True)

        rf = ctk.CTkFrame(row, fg_color=CARD, corner_radius=6)
        rf.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(rf, text="POINCARÉ PLOT", font=F(10,True),
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        self._slots["poincare"] = CanvasSlot(rf, 7, 7)

    # ─── TAB: INTERVALS ───────────────────────────────────────

    def _build_tab_intervals(self):
        t = self.tabs.tab("Intervals")
        top = ctk.CTkFrame(t, fg_color=CARD, corner_radius=6)
        top.pack(fill="both", expand=True, padx=8, pady=(8,4))
        ctk.CTkLabel(top, text="ECG INTERVALS  —  PR · QRS · QT · QTc (Bazett)",
                     font=F(10,True), text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        self._slots["intervals"] = CanvasSlot(top, 14, 5)
        bot = ctk.CTkFrame(t, fg_color=CARD, corner_radius=6)
        bot.pack(fill="x", padx=8, pady=(0,8))
        ctk.CTkLabel(bot, text="DESCRIPTIVE STATISTICS", font=F(10,True),
                     text_color=MUTED).pack(anchor="w", padx=10, pady=(6,2))
        self.txt_ivl = self._textbox(bot, 160, padx=10)

    # ─── TAB: BEAT TEMPLATE ───────────────────────────────────

    def _build_tab_beat(self):
        t = self.tabs.tab("Beat Template")
        top = ctk.CTkFrame(t, fg_color=CARD, corner_radius=6)
        top.pack(fill="both", expand=True, padx=8, pady=(8,4))
        ctk.CTkLabel(top, text="AVERAGE BEAT TEMPLATE",
                     font=F(10,True), text_color=MUTED).pack(anchor="w", padx=10, pady=(8,2))
        ctk.CTkLabel(top, text="Mean ± SD of all aligned beats  —  identifies waveform morphology issues",
                     font=FSM, text_color=LIGHT).pack(anchor="w", padx=10, pady=(0,4))
        self._slots["beat"] = CanvasSlot(top, 14, 5.5)
        bot = ctk.CTkFrame(t, fg_color=CARD, corner_radius=6)
        bot.pack(fill="both", expand=True, padx=8, pady=(0,8))
        ctk.CTkLabel(bot, text="BEAT AMPLITUDE DISTRIBUTION",
                     font=F(10,True), text_color=MUTED).pack(anchor="w", padx=10, pady=(6,2))
        self._slots["beat_dist"] = CanvasSlot(bot, 14, 3.5)

    # ─── TAB: EPOCHS ──────────────────────────────────────────

    def _build_tab_epochs(self):
        t = self.tabs.tab("Epochs")
        hdr = ctk.CTkFrame(t, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(8,4))
        ctk.CTkLabel(hdr, text="HRV ACROSS EPOCHS",
                     font=FH, text_color=MUTED).pack(side="left")
        ctk.CTkButton(hdr, text="⟳  Compute epochs",
                      command=self._compute_epochs,
                      fg_color=BLUE, hover_color="#0D47A1",
                      text_color="white", font=FSM,
                      height=28, corner_radius=5
                      ).pack(side="right")
        ctk.CTkLabel(t,
                     text="Tracks SDNN · RMSSD · HR across time — useful for physiological "
                          "state changes, stress protocols and long recordings",
                     font=FSM, text_color=LIGHT).pack(anchor="w", padx=10, pady=(0,4))
        self._slots["epochs"] = CanvasSlot(t, 14, 4.5)
        self.txt_epochs = self._textbox(t, 200, padx=10)

    # ─── TAB: SUMMARY ─────────────────────────────────────────

    def _build_tab_summary(self):
        t = self.tabs.tab("Summary")
        btn_row = ctk.CTkFrame(t, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(6,0))
        ctk.CTkButton(btn_row, text="📋  Copy to clipboard",
                      command=self._copy_summary,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FSM, height=28, corner_radius=5
                      ).pack(side="left")
        ctk.CTkButton(btn_row, text="💾  Save as .txt",
                      command=self._save_summary_txt,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FSM, height=28, corner_radius=5
                      ).pack(side="left", padx=6)
        self.lbl_epoch_info = ctk.CTkLabel(btn_row, text="", font=FSM,
                                            text_color=MUTED)
        self.lbl_epoch_info.pack(side="right")
        self.txt_sum = self._textbox(t, -1, padx=10, expand=True)

    # ════════════════════════════════════════════════════════
    #  DETECTION ENGINE  (fast, runs on threshold change)
    # ════════════════════════════════════════════════════════

    def _prepare_signal(self):
        """Filter + normalise + fix polarity once. Stores cands/proms for live threshold."""
        if self._signal_raw is None: return
        fs  = self._fs
        sig = self._signal_raw.copy()
        try:
            lp = float(self.ent_lp.get())
            hp = float(self.ent_hp.get())
            sig = bandpass(sig, fs, lp, hp)
        except Exception as _e:
            log.warning("bandpass skipped: %s", _e)
        if self.sw_notch.get():
            try: sig = notch(sig, fs)
            except Exception as _e: log.warning("notch skipped: %s", _e)
        try: sig = nk.ecg_clean(sig, sampling_rate=fs, method=self.cb_clean.get())
        except Exception as _e: log.warning("nk.ecg_clean skipped: %s", _e)
        sig = normalize(sig)
        # Fix polarity ONCE and store candidates
        mindist = self._safe_float(self.ent_minrr, 250.0)
        sig_out, inverted, cands, proms = fix_polarity(sig, fs, mindist)
        self._signal_flt = sig_out
        self._all_cands  = cands
        self._all_proms  = proms

    def _run_detection(self):
        """Apply threshold to pre-computed candidates. Fast — no signal processing."""
        if self._signal_flt is None or self._all_cands is None: return
        thr = self.sl_thr.get()
        rp_ok, rp_rej, t_amp = apply_threshold(
            self._signal_flt, self._all_cands, self._all_proms, thr)
        self._rpeaks_ok  = rp_ok
        self._rpeaks_rej = rp_rej
        self._thresh_amp = t_amp
        n = len(rp_ok)
        self.lbl_npeaks.configure(
            text=f"Peaks detected: {n}",
            text_color=GREEN if n > 10 else RED)

        # Signal quality score (0-100)
        if n > 5 and len(self._time) > 0:
            dur = self._time[-1]
            expected_bpm = 70
            expected_n   = dur / 60 * expected_bpm
            ratio = n / max(expected_n, 1)
            # Regularity: low CV of RR = better quality
            if len(rp_ok) > 3:
                rr_tmp = np.diff(rp_ok) / self._fs * 1000
                rr_cv  = rr_tmp.std() / (rr_tmp.mean() + 1e-6)
                quality = int(np.clip(100 * (1 - rr_cv) * np.clip(ratio, 0.5, 1.5), 0, 100))
            else:
                quality = 30
            self._sig_quality = quality
            color = GREEN if quality >= 70 else (ORANGE if quality >= 40 else RED)
            self.after(0, lambda q=quality, c=color:
                self.lbl_quality.configure(
                    text=f"Signal quality: {q}%", text_color=c))
        return n

    def _on_thr(self, val):
        self.lbl_thr.configure(text=f"Sensitivity:  {val:.3f}")
        # Sync entry field without triggering its callback
        self.ent_thr.delete(0, "end")
        self.ent_thr.insert(0, f"{val:.3f}")
        if self._signal_flt is not None and self._all_cands is not None:
            self._run_detection()
            self._draw_overview()
            self._draw_detail(self._nav_pos)

    def _on_thr_entry(self, event=None):
        """Called when user types a value in the exact-threshold entry."""
        try:
            val = float(self.ent_thr.get())
            val = max(0.01, min(2.0, val))
            self.sl_thr.set(val)
            self._on_thr(val)
        except ValueError:
            pass

    # ════════════════════════════════════════════════════════
    #  DRAWING
    # ════════════════════════════════════════════════════════

    def _draw_overview(self):
        if self._signal_flt is None: return
        # Build / reuse downsampled display cache
        if self._ds_time is None or self._ds_sig is None:
            self._ds_time = ds(self._time, 5000)
            self._ds_sig  = ds(self._signal_flt, 5000)
        t_ds   = self._ds_time
        s_ds   = self._ds_sig
        sig    = self._signal_flt   # full res for scatter y-coords
        rp_ok  = self._rpeaks_ok if self._rpeaks_ok is not None else np.array([])
        t_amp  = self._thresh_amp
        fs     = self._fs
        n_ok   = len(rp_ok)

        def draw(fig):
            ax = fig.add_subplot(111)
            _style_ax(ax)
            ax.plot(t_ds, s_ds, color=MP["signal"], lw=0.5, alpha=0.7)
            if n_ok:
                ax.scatter(rp_ok / fs, sig[rp_ok],
                           color=MP["rpeak_ok"], s=6, zorder=5, alpha=0.6)
            ax.axhline(t_amp, color=MP["threshold"], lw=1.0, ls="--", alpha=0.7)
            ax.set_xlabel("Time (s)"); ax.set_yticks([])
            ax.set_title(f"Overview  ({n_ok} peaks)", fontsize=9, loc="left")
            fig.tight_layout(pad=1.5)
        self._slots["overview"].update(draw)

    def _draw_detail(self, t_start=None):
        if self._signal_flt is None: return
        sig   = self._signal_flt
        time  = self._time
        fs    = self._fs
        try:   win = float(self.ent_window.get())
        except: win = 10.0
        if t_start is None: t_start = self._nav_pos
        t_end_  = min(time[-1], t_start + win)
        m       = (time >= t_start) & (time <= t_end_)
        rp_ok   = self._rpeaks_ok  if self._rpeaks_ok  is not None else np.array([])
        rp_rej  = self._rpeaks_rej if self._rpeaks_rej is not None else np.array([])
        t_amp   = self._thresh_amp

        # Pre-compute masks (outside draw fn for closure capture)
        mask_a = (rp_ok  / fs >= t_start) & (rp_ok  / fs <= t_end_) if len(rp_ok)  else np.array([], bool)
        mask_r = (rp_rej / fs >= t_start) & (rp_rej / fs <= t_end_) if len(rp_rej) else np.array([], bool)
        n_win  = int(mask_a.sum())

        def draw(fig):
            ax = fig.add_subplot(111)
            _style_ax(ax)
            ax.plot(time[m], sig[m], color=MP["signal"], lw=0.8)
            if mask_r.any():
                ax.scatter(rp_rej[mask_r] / fs, sig[rp_rej[mask_r]],
                           color=MP["rpeak_bad"], s=30, zorder=4,
                           marker="o", label="Rejected", alpha=0.5)
            if mask_a.any():
                ax.scatter(rp_ok[mask_a] / fs, sig[rp_ok[mask_a]],
                           color=MP["rpeak_ok"], s=55, zorder=5,
                           marker="o", label="Accepted")
            ax.axhline(t_amp, color=MP["threshold"],
                       lw=1.4, ls="--", label=f"Threshold ({t_amp:.3f})")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (norm.)")
            ax.set_title(f"Detail  {t_start:.1f} – {t_end_:.1f} s   "
                         f"({n_win} peaks in view)", fontsize=10, loc="left")
            ax.legend(fontsize=8, framealpha=0, loc="upper right")
            fig.tight_layout(pad=1.5)
        self._slots["detail"].update(draw)

    def _refresh_detail(self):
        self._draw_detail(self._nav_pos)

    def _nav(self, direction: int) -> None:
        """Navigate the detail view left/right by 80% of the window width."""
        if self._time is None or len(self._time) == 0:
            return
        try:
            win = float(self.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except (ValueError, TypeError):
            win = 10.0
        self._nav_pos = max(0.0, min(float(self._time[-1]) - win,
                                     self._nav_pos + direction * win * 0.8))
        self._draw_detail()

    def _nav_reset(self):
        self._nav_pos = 0.0
        self._draw_detail()

    # ════════════════════════════════════════════════════════
    #  RESULT PLOTS  — each method owns one slot
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _safe_df_val(df: "pd.DataFrame | None", col: str, decimals: int = 3) -> str:
        """Return a formatted scalar from a DataFrame cell, or '—' on any error."""
        try:
            v = float(df[col].values[0])  # type: ignore[index]
            return f"{v:.{decimals}f}" if np.isfinite(v) else "—"
        except Exception:
            return "—"

    @staticmethod
    def _df_to_text(df: "pd.DataFrame | None") -> str:
        """Render every finite numeric column of a NeuroKit2 HRV DataFrame as text."""
        if df is None or df.empty:
            return "  (not computed)"
        rows = []
        for c in df.columns:
            try:
                v = float(df[c].values[0])
                if np.isfinite(v):
                    rows.append(f"  {c.replace('HRV_', ''):<26} {v:>12.4f}")
            except Exception as _e:
                log.debug("_df_to_text skip '%s': %s", c, _e)
        return "\n".join(rows) or "  (no finite values)"

    def _plot_rr(self, r: dict) -> None:
        """Plot RR tachogram + HR trace + distribution histogram."""
        rdf = r["rr_df"]
        rr_ms_raw = r.get("rr_ms", np.array([]))
        if rdf.empty and len(rr_ms_raw) > 1 and self._rpeaks_ok is not None:
            rdf = pd.DataFrame({
                "Time_s": self._rpeaks_ok[1:len(rr_ms_raw)+1] / self._fs,
                "RR_ms":  rr_ms_raw,
                "HR_bpm": 60000.0 / np.clip(rr_ms_raw, 1, None),
            })
        if rdf.empty:
            log.warning("_plot_rr: empty rdf — skipping")
            return

        t_  = ds(rdf["Time_s"].values)
        rr_ = ds(rdf["RR_ms"].values)
        hr_ = ds(rdf["HR_bpm"].values)
        rr_mean = float(rdf["RR_ms"].mean())
        hr_mean = float(rdf["HR_bpm"].mean())
        c1, c2  = "#388E3C", "#E65100"

        def draw_rr(fig):
            axes = fig.subplots(2, 1, sharex=True)
            fig.subplots_adjust(hspace=0.06)
            for ax in axes: _style_ax(ax)
            axes[0].plot(t_, rr_, color=c1, lw=0.7)
            axes[0].axhline(rr_mean, color=c1, ls="--", lw=0.9, alpha=0.5)
            axes[0].set_ylabel("RR (ms)")
            axes[0].set_title("RR Intervals", fontsize=10, loc="left")
            axes[1].plot(t_, hr_, color=c2, lw=0.7)
            axes[1].axhline(hr_mean, color=c2, ls="--", lw=0.9, alpha=0.5)
            axes[1].set_ylabel("HR (bpm)"); axes[1].set_xlabel("Time (s)")
            axes[1].set_title("Instantaneous HR", fontsize=10, loc="left")
            fig.tight_layout(pad=1.5)
        self._slots["rr"].update(draw_rr)

        self._txt(self.txt_rr, "\n".join(
            f"  {k:<14} {v:>10.2f}" for k, v in rdf["RR_ms"].describe().items()))

        rr_clip = rdf["RR_ms"].clip(200, 2500).values
        def draw_hist(fig):
            ax = fig.add_subplot(111); _style_ax(ax)
            ax.hist(rr_clip, bins=50, color=c1, alpha=0.7, edgecolor="white", lw=0.3)
            ax.set_xlabel("RR (ms)"); ax.set_ylabel("Count")
            ax.set_title("RR Distribution", fontsize=10, loc="left")
            fig.tight_layout(pad=1.5)
        self._slots["rr_hist"].update(draw_hist)

    def _plot_hrv_tables(self, r: dict) -> None:
        """Populate time-domain and frequency-domain HRV text boxes."""
        self._txt(self.txt_td, self._df_to_text(r["hrv_time"]))

        fd_df = r["hrv_freq"]
        if fd_df is None or fd_df.empty:
            self._txt(self.txt_fd, "  (not computed)")
            return
        lines: list[str] = []
        for col in fd_df.columns:
            try:
                v = float(fd_df[col].values[0])
                if not np.isfinite(v):
                    continue
                name = col.replace("HRV_", "")
                if name in ("LF", "HF", "VLF"):
                    lines.append(f"  {name:<26} {v*100:>10.1f} %")
                elif name == "LFHF":
                    lines.append(f"  {'LF/HF ratio':<26} {v:>10.3f}")
                elif name in ("LF_peak", "HF_peak", "VLF_peak"):
                    lines.append(f"  {name + ' (Hz)':<26} {v:>10.4f}")
                else:
                    lines.append(f"  {name:<26} {v:>10.4f}")
            except Exception as _e:
                log.debug("_plot_hrv_tables fd skip '%s': %s", col, _e)
        self._txt(self.txt_fd, "\n".join(lines) if lines else "  (not computed)")

    def _plot_psd(self, r: dict) -> None:
        """Welch PSD with VLF / LF / HF band shading."""
        rr_ms = r["rr_ms"]
        if len(rr_ms) < 8:
            log.warning("_plot_psd: too few RR intervals (%d)", len(rr_ms))
            return
        try:
            from scipy.signal import welch as _welch
            ts_ = np.cumsum(rr_ms) / 1000
            tu_ = np.arange(ts_[0], ts_[-1], 0.25)
            if len(tu_) < 4:
                return
            ri_ = np.interp(tu_, ts_, rr_ms)
            fq_, ps_ = _welch(ri_ - ri_.mean(), fs=4.0,
                              nperseg=min(256, len(ri_) // 2))
            bands_ = [("VLF", 0, 0.04, "#1565C0"),
                      ("LF",  0.04, 0.15, "#6A1B9A"),
                      ("HF",  0.15, 0.40, "#1B5E20")]

            def draw_psd(fig):
                ax = fig.add_subplot(111); _style_ax(ax)
                ax.semilogy(fq_, ps_, color="#546E7A", lw=0.9)
                for nm, lo, hi, c in bands_:
                    m = (fq_ >= lo) & (fq_ <= hi)
                    ax.fill_between(fq_, ps_, where=m, alpha=0.3, color=c, label=nm)
                ax.set_xlabel("Hz"); ax.set_ylabel("ms²/Hz")
                ax.set_xlim(0, 0.5)
                ax.legend(fontsize=9, framealpha=0)
                ax.set_title("Power Spectral Density", fontsize=10, loc="left")
                fig.tight_layout(pad=1.5)
            self._slots["psd"].update(draw_psd)
        except Exception as _e:
            log.warning("_plot_psd failed: %s", _e)

    def _plot_radar(self, r: dict) -> None:
        """Normalised HRV radar / spider chart."""
        try:
            vals: dict[str, float] = {}
            for df_, keys_ in [
                (r["hrv_time"],   ["HRV_SDNN", "HRV_RMSSD", "HRV_pNN50"]),
                (r["hrv_freq"],   ["HRV_LF",   "HRV_HF",    "HRV_LFHF"]),
                (r["hrv_nonlin"], ["HRV_SD1",  "HRV_SD2",   "HRV_SampEn"]),
            ]:
                if df_ is None or df_.empty:
                    continue
                for k in keys_:
                    if k not in df_.columns:
                        continue
                    try:
                        v = float(df_[k].values[0])
                        if np.isfinite(v):
                            vals[k.replace("HRV_", "")] = v
                    except Exception as _e:
                        log.debug("_plot_radar skip '%s': %s", k, _e)

            if len(vals) < 3:
                log.debug("_plot_radar: only %d finite metrics — skipping", len(vals))
                return

            lbs_  = list(vals.keys())
            vs_   = np.array(list(vals.values()))
            rng   = vs_.max() - vs_.min()
            vn_   = (vs_ - vs_.min()) / (rng + 1e-9)
            N_    = len(lbs_)
            ang_  = np.linspace(0, 2 * np.pi, N_, endpoint=False).tolist()
            vp_   = vn_.tolist() + [vn_[0]]
            ang_c = ang_ + ang_[:1]

            def draw_radar(fig):
                ax = fig.add_subplot(111, polar=True)
                ax.set_facecolor(MP["axes"])
                ax.plot(ang_c, vp_, color=RED, lw=2)
                ax.fill(ang_c, vp_, color=RED, alpha=0.1)
                ax.set_thetagrids(np.degrees(ang_), lbs_, fontsize=9)
                ax.set_ylim(0, 1); ax.yaxis.set_visible(False)
                ax.grid(color=MP["grid"])
                ax.spines["polar"].set_color(MP["border"])
                fig.tight_layout(pad=1.5)
            self._slots["radar"].update(draw_radar)
        except Exception as _e:
            log.warning("_plot_radar failed: %s", _e)

    def _plot_nonlinear(self, r: dict) -> None:
        """Poincaré plot + non-linear HRV table."""
        self._txt(self.txt_nl, self._df_to_text(r["hrv_nonlin"]))

        rr_ms = r["rr_ms"]
        if len(rr_ms) < 2:
            log.warning("_plot_nonlinear: too few RR values")
            return

        nl   = r["hrv_nonlin"]
        sd1_ = self._safe_df_val(nl, "HRV_SD1", 1)
        sd2_ = self._safe_df_val(nl, "HRV_SD2", 1)
        rr_a = rr_ms[:-1]; rr_b = rr_ms[1:]
        lo   = float(rr_ms.min()) - 20
        hi   = float(rr_ms.max()) + 20
        lim_ = [lo, hi]

        def draw_poincare(fig):
            ax = fig.add_subplot(111); _style_ax(ax)
            ax.scatter(rr_a, rr_b, s=4, alpha=0.3, color=BLUE)
            ax.plot(lim_, lim_, color=BORDER2, lw=1, ls="--")
            ax.set_xlim(lim_); ax.set_ylim(lim_); ax.set_aspect("equal")
            ax.set_xlabel("RR_n (ms)"); ax.set_ylabel("RR_n+1 (ms)")
            ax.set_title(f"Poincaré   SD1={sd1_}  SD2={sd2_}",
                         fontsize=10, loc="left")
            fig.tight_layout(pad=1.5)
        self._slots["poincare"].update(draw_poincare)

    def _plot_intervals(self, r: dict) -> None:
        """Violin + box plot for PR / QRS / QT / QTc intervals."""
        ivl   = r["intervals"]
        cols_ = [c for c in ["PR_ms", "QRS_ms", "QT_ms", "QTc_ms"]
                 if c in ivl.columns and ivl[c].dropna().__len__() > 3]

        if not cols_:
            def draw_na(fig):
                ax = fig.add_subplot(111); _style_ax(ax)
                ax.text(0.5, 0.5,
                        "Interval delineation not available\n"
                        "(requires clear P/Q/S/T waves at high SNR)",
                        ha="center", va="center", fontsize=11,
                        color=MP["muted"], transform=ax.transAxes, linespacing=1.8)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(False)
                fig.tight_layout(pad=1.5)
            self._slots["intervals"].update(draw_na)
            return

        pal_     = ["#1565C0", "#2E7D32", "#AD1457", "#E65100"]
        col_data = [(c, ivl[c].dropna().values, cc)
                    for c, cc in zip(cols_, pal_)]

        def draw_ivl(fig):
            axes = fig.subplots(1, len(col_data), sharey=False)
            if len(col_data) == 1:
                axes = [axes]
            for ax, (col, data, cc) in zip(axes, col_data):
                _style_ax(ax)
                if len(data) < 2:
                    continue
                vp_ = ax.violinplot(data, positions=[0],
                                    showmedians=True, showextrema=False)
                for pc in vp_["bodies"]:
                    pc.set_facecolor(cc); pc.set_alpha(0.2)
                vp_["cmedians"].set_color(cc); vp_["cmedians"].set_linewidth(2)
                ax.boxplot(data, positions=[0], widths=0.12, patch_artist=True,
                           boxprops=dict(facecolor=cc, alpha=0.2),
                           medianprops=dict(color=cc, lw=2),
                           whiskerprops=dict(color=LIGHT),
                           capprops=dict(color=LIGHT),
                           flierprops=dict(marker=".", color=LIGHT, ms=3))
                ax.set_title(col.replace("_ms", ""), fontsize=10)
                ax.set_ylabel("ms"); ax.set_xticks([])
                d2 = data[np.isfinite(data)]
                if len(d2):
                    ax.text(0, d2.max() * 1.03,
                            f"{d2.mean():.1f}±{d2.std():.1f}",
                            ha="center", fontsize=8, color=MUTED)
            fig.tight_layout(pad=1.5)
        self._slots["intervals"].update(draw_ivl)
        self._txt(self.txt_ivl, ivl.describe().round(2).to_string())

    def _plot_beat(self, r: dict) -> None:
        """Average beat template + amplitude/morphology distributions."""
        btime = r.get("beat_time")
        if btime is None or self._rpeaks_ok is None or len(self._rpeaks_ok) == 0:
            log.debug("_plot_beat: no beat_time or rpeaks — skipping")
            return

        hw_   = len(btime) // 2
        sig_  = self._signal_flt
        if sig_ is None:
            return

        tmpl = [
            sig_[rp - hw_: rp + hw_]
            for rp in self._rpeaks_ok
            if rp - hw_ >= 0 and rp + hw_ < len(sig_)
        ]
        if len(tmpl) < 4:
            log.warning("_plot_beat: only %d valid beats — skipping", len(tmpl))
            return

        tmat_     = np.array(tmpl)
        mean_b    = tmat_.mean(axis=0)
        sd_b      = tmat_.std(axis=0)
        n_b       = len(tmpl)
        peak_amps = tmat_[:, hw_]
        stride    = max(1, n_b // 60)   # show at most ~60 individual traces

        def draw_beat(fig):
            ax = fig.add_subplot(111); _style_ax(ax)
            for beat in tmat_[::stride]:
                ax.plot(btime, beat, color="#BDBDBD", lw=0.3, alpha=0.35)
            ax.fill_between(btime, mean_b - sd_b, mean_b + sd_b,
                            color=BLUE, alpha=0.18, label="±1 SD")
            ax.plot(btime, mean_b, color=BLUE, lw=2.2, label="Mean")
            ax.axvline(0, color=RED, lw=1.0, ls="--", alpha=0.7, label="R peak")
            ax.set_xlabel("Time relative to R peak (ms)")
            ax.set_ylabel("Amplitude (norm.)")
            ax.set_title(f"Average Beat  (n={n_b})", fontsize=10, loc="left")
            ax.legend(fontsize=9, framealpha=0)
            fig.tight_layout(pad=1.5)
        self._slots["beat"].update(draw_beat)

        # Correlation of each beat with the mean template
        corr = np.array([
            float(np.corrcoef(mean_b, b)[0, 1])
            for b in tmat_
        ])
        mean_corr = float(np.nanmean(corr))

        def draw_beat_dist(fig):
            axes2 = fig.subplots(1, 2); 
            for a in axes2: _style_ax(a)
            axes2[0].hist(peak_amps, bins=min(40, n_b // 2 + 1),
                          color=BLUE, alpha=0.7, edgecolor="white", lw=0.3)
            axes2[0].set_xlabel("R-peak amplitude (norm.)")
            axes2[0].set_ylabel("Count")
            axes2[0].set_title("R-peak Amplitude", fontsize=10, loc="left")
            axes2[1].hist(corr, bins=min(30, n_b // 2 + 1),
                          color=GREEN, alpha=0.7, edgecolor="white", lw=0.3)
            axes2[1].axvline(mean_corr, color=RED, lw=1.2, ls="--",
                             label=f"mean={mean_corr:.3f}")
            axes2[1].set_xlabel("Correlation with mean beat")
            axes2[1].set_ylabel("Count")
            axes2[1].set_title("Beat Morphology Consistency", fontsize=10, loc="left")
            axes2[1].legend(fontsize=8, framealpha=0)
            fig.tight_layout(pad=1.5)
        self._slots["beat_dist"].update(draw_beat_dist)

    def _plot_summary(self, r: dict) -> None:
        """Build the plain-text summary report."""
        hr_  = r["hr"]
        td_  = r["hrv_time"]; fd_ = r["hrv_freq"]; nl_ = r["hrv_nonlin"]
        ivl  = r["intervals"]
        g = self._safe_df_val   # shorthand

        lines = [
            "═" * 58,
            "  ECG ANALYSIS  —  Summary Report",
            f"  Subject  :  {self.ent_subject.get()}",
            f"  Date     :  {datetime.now():%Y-%m-%d  %H:%M}",
            f"  File     :  {os.path.basename(self._filepath or '')}",
            "═" * 58, "",
            "  HEART RATE",
            f"    Mean             {hr_['mean']:.1f} bpm",
            f"    Min  (2nd %ile)  {hr_['min']:.1f} bpm",
            f"    Max  (98th %ile) {hr_['max']:.1f} bpm",
            f"    SD               {hr_['std']:.2f} bpm",
            f"    N beats          {hr_['n']}", "",
            "  TIME DOMAIN HRV",
            f"    MeanNN   {g(td_,'HRV_MeanNN')} ms",
            f"    SDNN     {g(td_,'HRV_SDNN')} ms",
            f"    RMSSD    {g(td_,'HRV_RMSSD')} ms",
            f"    pNN50    {g(td_,'HRV_pNN50')} %",
            f"    pNN20    {g(td_,'HRV_pNN20')} %", "",
            "  FREQUENCY DOMAIN",
            f"    VLF      {g(fd_,'HRV_VLF')} n.u.",
            f"    LF       {g(fd_,'HRV_LF')} n.u.",
            f"    HF       {g(fd_,'HRV_HF')} n.u.",
            f"    LF/HF    {g(fd_,'HRV_LFHF')}", "",
            "  NON-LINEAR",
            f"    SD1      {g(nl_,'HRV_SD1')} ms",
            f"    SD2      {g(nl_,'HRV_SD2')} ms",
            f"    SampEn   {g(nl_,'HRV_SampEn')}",
            f"    ApEn     {g(nl_,'HRV_ApEn')}",
            f"    DFA α1   {g(nl_,'HRV_DFA_alpha1')}",
            f"    DFA α2   {g(nl_,'HRV_DFA_alpha2')}",
        ]
        if "QT_ms" in ivl.columns:
            lines += ["", "  ECG INTERVALS  (mean ± SD)"]
            for col in ["PR_ms", "QRS_ms", "QT_ms", "QTc_ms"]:
                if col in ivl.columns:
                    d2 = ivl[col].dropna()
                    if len(d2):
                        lines.append(
                            f"    {col:<16} {d2.mean():.1f} ± {d2.std():.1f} ms")
        lines += ["", "═" * 58]
        self._txt(self.txt_sum, "\n".join(lines))

    def _draw_results(self) -> None:
        """Orchestrate all result plots.

        Each sub-plot is scheduled in its own after() call so the UI
        remains responsive and the tab-bar stays interactive while
        heavier plots (PSD, radar, beat template) are rendering.
        """
        r = self._results
        if r is None:
            log.warning("_draw_results called with no results")
            return

        # Schedule each plot on its own event-loop tick.
        # Delay is cumulative so plots appear progressively (50 ms apart).
        tasks = [
            (  0, lambda: self._plot_rr(r)),
            ( 50, lambda: self._plot_hrv_tables(r)),
            (100, lambda: self._plot_nonlinear(r)),
            (150, lambda: self._plot_psd(r)),
            (200, lambda: self._plot_radar(r)),
            (250, lambda: self._plot_intervals(r)),
            (300, lambda: self._plot_beat(r)),
            (350, lambda: self._plot_summary(r)),
        ]
        for delay_ms, fn in tasks:
            self.after(delay_ms, fn)


    def _update_kpis(self):
        r  = self._results
        hr = r["hr"]
        rdf = r["rr_df"]
        td  = r["hrv_time"]
        def tv(k):
            try: return f"{float(td[k].values[0]):.1f}"
            except: return "—"
        self._kpi["hr_mean"].configure(text=f"{hr['mean']:.0f} bpm")
        self._kpi["hr_range"].configure(text=f"{hr['min']:.0f}–{hr['max']:.0f}")
        try:
            rr_mean_v = float(np.nanmean(r["rr_ms"]))
            self._kpi["rr_mean"].configure(text=f"{rr_mean_v:.0f} ms")
        except: self._kpi["rr_mean"].configure(text="—")
        self._kpi["n_beats"].configure(text=str(hr["n"]))
        self._kpi["sdnn"].configure(text=tv("HRV_SDNN"))
        self._kpi["rmssd"].configure(text=tv("HRV_RMSSD"))
        self._kpi["pnn50"].configure(text=tv("HRV_pNN50"))
        try:    self._kpi["dur"].configure(text=f"{rdf['Time_s'].iloc[-1]:.0f} s")
        except: self._kpi["dur"].configure(text="—")

    # ════════════════════════════════════════════════════════
    #  ACTIONS
    # ════════════════════════════════════════════════════════

    def _open_file(self):
        p = filedialog.askopenfilename(
            title="Open Spike2 .mat",
            filetypes=[("MATLAB files","*.mat"),("All","*.*")])
        if not p: return
        self._filepath = p
        self.lbl_file.configure(text=os.path.basename(p), text_color=GREEN)
        self._add_recent(p)
        # Auto-start preview immediately
        self._preview()

    def _show_channels(self):
        if not self._filepath:
            messagebox.showwarning("No file", "Open a .mat file first."); return
        try:    messagebox.showinfo("Channels", list_channels(self._filepath))
        except Exception as e: messagebox.showerror("Error", str(e))

    def _preview(self):
        """Load + filter + detect — fast, no HRV."""
        if not self._filepath:
            messagebox.showwarning("No file", "Open a .mat file first."); return
        self.btn_preview.configure(state="disabled", text="Loading…")
        self._status("Loading signal…", ORANGE)
        self.progress.pack(side="bottom", fill="x")
        self.progress.start()
        threading.Thread(target=self._preview_thread, daemon=True).start()

    def _preview_thread(self):
        try:
            channel = self.ent_channel.get().strip() or "ECG"
            self._fs = int(self._safe_float(self.ent_fs, 1000))
            t0 = self._safe_float(self.ent_t_start, 0.0)
            t1 = self._safe_float(self.ent_t_end,   0.0)

            sig, det_ch, keys = load_mat_signal(self._filepath, channel)

            if det_ch != channel:
                self.after(0, lambda: self.lbl_file.configure(
                    text=f"Auto: {det_ch}", text_color=ORANGE))

            # Segment
            i0 = int(t0 * self._fs) if t0 > 0 else 0
            i1 = int(t1 * self._fs) if t1 > 0 else len(sig)
            sig = sig[i0:i1]

            if sig.std() < 1e-10:
                raise ValueError("Signal is flat — wrong channel.")

            self._signal_raw = sig
            self._time       = np.arange(len(sig)) / self._fs
            self._nav_pos    = 0.0
            # Invalidate display cache
            self._ds_time = None
            self._ds_sig  = None

            self.after(0, lambda: self._status("Filtering…", ORANGE))
            self._prepare_signal()
            self._run_detection()

            self.after(0, self._on_preview_done)

        except Exception as e:
            import traceback; tb = traceback.format_exc()
            self.after(0, lambda: self._status(f"Error: {e}", RED))
            self.after(0, lambda: messagebox.showerror(
                "Error", str(e) + "\n\n" + tb))
            self.after(0, lambda: self.btn_preview.configure(
                state="normal", text="▶  Preview Detection"))
            self.after(0, lambda: (self.progress.stop(), self.progress.pack_forget()))

    def _on_preview_done(self):
        self.progress.stop(); self.progress.pack_forget()
        self.btn_preview.configure(state="normal", text="▶  Preview Detection")
        n = len(self._rpeaks_ok) if self._rpeaks_ok is not None else 0
        dur = self._time[-1] if self._time is not None else 0
        self._status(
            f"Signal ready — {n} peaks  |  {dur:.0f} s  |  {self._fs} Hz  "
            "→ adjust threshold then Run Full Analysis.", GREEN)
        self.tabs.set("Detection")
        self._draw_overview()
        self._draw_detail()

    def _run(self):
        if not NK_OK:
            messagebox.showerror("Missing", "pip install neurokit2"); return
        if self._signal_flt is None or self._rpeaks_ok is None:
            messagebox.showwarning("Not ready",
                                   "Click '▶ Preview Detection' first."); return
        if len(self._rpeaks_ok) < 5:
            messagebox.showwarning("Too few peaks",
                                   f"Only {len(self._rpeaks_ok)} peaks detected.\n"
                                   "Adjust threshold / detection settings."); return
        self.btn_run.configure(state="disabled", text="Analysing…")
        self._status("Running HRV analysis…", ORANGE)
        self.progress.pack(side="bottom", fill="x")
        self.progress.start()
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _run_thread(self):
        try:
            # Artifact correction on rpeaks if requested
            rp = self._rpeaks_ok.copy()
            if self.sw_art.get():
                try:
                    fixed, _ = nk.signal_fixpeaks(rp, sampling_rate=self._fs,
                                                   iterative=True, method="Kubios")
                    rp = np.array(fixed, dtype=int).flatten()
                except Exception as _e:
                    log.warning("Artifact correction failed: %s", _e)

            res = run_full_analysis(self._signal_flt, rp, self._fs)
            self._results = res
            self._rpeaks_ok = rp  # update with corrected peaks
            self.after(0, self._on_run_done)
        except Exception as e:
            import traceback; tb = traceback.format_exc()
            self.after(0, lambda: self._status(f"Analysis error: {e}", RED))
            self.after(0, lambda: messagebox.showerror(
                "Analysis Error", str(e) + "\n\n" + tb))
            self.after(0, lambda: self.btn_run.configure(
                state="normal", text="⚡  Run Full Analysis"))

    def _on_run_done(self):
        self.progress.stop(); self.progress.pack_forget()
        self.btn_run.configure(state="normal", text="⚡  Run Full Analysis")
        n = self._results["hr"]["n"]
        self._status(f"Analysis complete — {n} beats", GREEN)
        self._update_kpis()
        # Redraw overview & detail with final peaks
        self._draw_overview()
        self._draw_detail()
        # Draw all result tabs (each sub-plot is scheduled via after())
        try:
            self._draw_results()
        except Exception as _e:
            log.exception("_draw_results orchestration failed")

    # ════════════════════════════════════════════════════════
    #  EXPORT
    # ════════════════════════════════════════════════════════

    def _sheets(self):
        r  = self._results; hr = r["hr"]
        return {
            "HR": pd.DataFrame([{
                "HR_mean_bpm": round(hr["mean"],2),
                "HR_min_bpm":  round(hr["min"],2),
                "HR_max_bpm":  round(hr["max"],2),
                "HR_std_bpm":  round(hr["std"],2),
                "N_beats":     hr["n"],
            }]),
            "HRV_Time":      r["hrv_time"],
            "HRV_Frequency": r["hrv_freq"],
            "HRV_NonLinear": r["hrv_nonlin"],
            "RR_Timeseries": r["rr_df"],
            "ECG_Intervals": r["intervals"],
        }

    def _write_excel(self, dst):
        with pd.ExcelWriter(dst, engine="openpyxl") as wr:
            for nm, df in self._sheets().items():
                if df is not None and not df.empty:
                    df.to_excel(wr, sheet_name=nm, index=False)

    def _export_excel(self):
        if not self._results:
            messagebox.showwarning("No results","Run analysis first."); return
        sub = self.ent_subject.get()
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        p   = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=f"{sub}_ECG_{ts}.xlsx",
            filetypes=[("Excel","*.xlsx")])
        if not p: return
        self._write_excel(p)
        self._status("Excel saved", GREEN)
        messagebox.showinfo("Saved", p)

    def _export_zip(self):
        if not self._results:
            messagebox.showwarning("No results","Run analysis first."); return
        sub = self.ent_subject.get()
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        p   = filedialog.asksaveasfilename(
            defaultextension=".zip",
            initialfile=f"{sub}_ECG_full_{ts}.zip",
            filetypes=[("ZIP","*.zip")])
        if not p: return
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
            xl = io.BytesIO()
            self._write_excel(xl); xl.seek(0)
            zf.writestr(f"{sub}_ECG_{ts}.xlsx", xl.read())
            for key, name in [("overview","00_overview"),("detail","01_detail"),
                               ("rr","02_rr_hr"),("rr_hist","03_rr_hist"),
                               ("psd","04_psd"),("radar","05_radar"),
                               ("poincare","06_poincare"),
                               ("intervals","07_intervals"),("beat","08_beat")]:
                slot = self._slots.get(key)
                if slot and slot.fig:
                    try:
                        buf = io.BytesIO()
                        slot.fig.savefig(buf, format="png", dpi=300,
                                         bbox_inches="tight", facecolor=MP["bg"])
                        buf.seek(0)
                        zf.writestr(f"figures/{name}.png", buf.read())
                    except Exception as _e:
                        log.warning("Figure '%s' not saved to ZIP: %s", name, _e)
        self._status("ZIP saved", GREEN)
        messagebox.showinfo("Saved", p)

    # ════════════════════════════════════════════════════════
    #  HELPERS
    # ════════════════════════════════════════════════════════

    def _labeled_row(self, parent, items: list, pad: dict) -> None:
        """Render a horizontal row of (label, attr, default) entry pairs."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(**pad, fill="x", pady=(0, 6))
        for lbl, attr, val in items:
            col = ctk.CTkFrame(row, fg_color="transparent")
            col.pack(side="left", fill="x", expand=True, padx=(0, 6))
            ctk.CTkLabel(col, text=lbl, font=FSM, text_color=MUTED).pack(anchor="w")
            e = ctk.CTkEntry(col, font=FL, height=28, fg_color=BG,
                             border_color=BORDER2, text_color=TEXT)
            e.insert(0, val)
            e.pack(fill="x")
            setattr(self, f"ent_{attr}", e)

    def _entry(self, parent, label, attr, default, pad):
        ctk.CTkLabel(parent, text=label, font=FSM,
                     text_color=MUTED, anchor="w").pack(**pad, fill="x")
        e = ctk.CTkEntry(parent, font=FL, height=30, fg_color=BG,
                          border_color=BORDER2, text_color=TEXT)
        e.insert(0, default); e.pack(**pad, fill="x", pady=(2,8))
        setattr(self, f"ent_{attr}", e)

    def _sep(self, p):
        ctk.CTkFrame(p, height=1, fg_color=BORDER).pack(
            fill="x", padx=12, pady=8)

    def _hdr(self, p, text):
        ctk.CTkLabel(p, text=text, font=F(10,True),
                     text_color=RED, anchor="w").pack(
            padx=16, fill="x", pady=(4,2))

    def _textbox(self, parent, h=180, padx=0, expand=False):
        kw = dict(font=FM, fg_color="transparent", text_color=TEXT,
                  border_width=0, scrollbar_button_color=BORDER,
                  scrollbar_button_hover_color=BORDER2)
        if h > 0: kw["height"] = h
        t = ctk.CTkTextbox(parent, **kw)
        if h < 0 or expand:
            t.pack(fill="both", expand=True, padx=padx, pady=(0,6))
        else:
            t.pack(fill="x", padx=padx, pady=(0,4))
        return t

    def _txt(self, w, text):
        w.configure(state="normal")
        w.delete("1.0","end"); w.insert("1.0", text)
        w.configure(state="disabled")

    def _status(self, text, color=MUTED):
        self.lbl_status.configure(text=text, text_color=color)
        self.update_idletasks()

    def _safe_float(self, widget_or_val, default: float = 0.0) -> float:
        """Safely extract a float from a CTk widget or raw value.

        Returns *default* if the value is missing, empty, or non-numeric.
        Never raises.
        """
        try:
            raw = widget_or_val.get() if hasattr(widget_or_val, "get") else widget_or_val
            v = float(raw)
            if not np.isfinite(v):
                log.debug("_safe_float: non-finite value %r — using default %s", raw, default)
                return default
            return v
        except (ValueError, TypeError) as _e:
            log.debug("_safe_float: could not parse %r — using default %s: %s",
                      widget_or_val, default, _e)
            return default


    # ════════════════════════════════════════════════════════
    #  EPOCH ANALYSIS
    # ════════════════════════════════════════════════════════

    def _compute_epochs(self):
        if self._rpeaks_ok is None or len(self._rpeaks_ok) < 10:
            messagebox.showwarning("No data", "Run Preview Detection first."); return
        try:
            epoch_s   = max(10, self._safe_float(self.ent_epoch, 300))
            overlap_s = max(0,  self._safe_float(self.ent_overlap, 0))
            fs        = self._fs
            rp        = self._rpeaks_ok
            dur       = self._time[-1]

            step   = epoch_s - overlap_s
            starts = np.arange(0, dur - epoch_s + 1, step)
            if len(starts) < 2:
                short_msg = (f"Recording too short for {epoch_s:.0f}s epochs. "
                             f"Try a shorter epoch (e.g. {int(dur//3)}s).")
                messagebox.showwarning("Too few epochs", short_msg); return

            rows = []
            for t0 in starts:
                t1  = t0 + epoch_s
                ep_rp = rp[(rp/fs >= t0) & (rp/fs < t1)]
                if len(ep_rp) < 5: continue
                rr = np.diff(ep_rp).astype(float) / fs * 1000
                try:
                    hrv_e = nk.hrv_time(ep_rp, sampling_rate=fs, show=False)
                    sdnn  = float(hrv_e["HRV_SDNN"].values[0])
                    rmssd = float(hrv_e["HRV_RMSSD"].values[0])
                except Exception as _e:
                    log.warning("Epoch hrv_time failed, using manual calc: %s", _e)
                    sdnn  = float(rr.std())
                    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)))
                rows.append({
                    "Epoch_start_s": round(t0,1),
                    "Epoch_end_s":   round(t1,1),
                    "N_beats":       len(ep_rp),
                    "HR_mean":       round(float(60000/rr.mean()),1),
                    "MeanNN":        round(float(rr.mean()),1),
                    "SDNN":          round(sdnn, 2),
                    "RMSSD":         round(rmssd, 2),
                })

            if not rows:
                messagebox.showwarning("No epochs", "No valid epochs found."); return

            df = pd.DataFrame(rows)
            self._epoch_df = df

            # Draw
            t_mid_   = (df["Epoch_start_s"] + df["Epoch_end_s"]) / 2
            ep_rows_ = [("HR_mean","HR (bpm)","#E65100","Heart Rate per Epoch"),
                        ("SDNN",   "SDNN (ms)","#1565C0","SDNN per Epoch"),
                        ("RMSSD",  "RMSSD (ms)","#2E7D32","RMSSD per Epoch")]
            ep_vals_ = {col: df[col].values for col,*_ in ep_rows_}
            def draw_ep(fig):
                axes = fig.subplots(3, 1, sharex=True)
                fig.subplots_adjust(hspace=0.08)
                for ax, (col, ylabel, c, title) in zip(axes, ep_rows_):
                    _style_ax(ax)
                    ax.plot(t_mid_, ep_vals_[col], color=c, lw=1.5, marker="o", ms=4)
                    ax.fill_between(t_mid_, ep_vals_[col], alpha=0.08, color=c)
                    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10, loc="left")
                axes[-1].set_xlabel("Time (s)")
                fig.tight_layout(pad=1.5)
            self._slots["epochs"].update(draw_ep)
            self._txt(self.txt_epochs, df.to_string(index=False))

            n_ep = len(df)
            self.lbl_epoch_info.configure(
                text=f"{n_ep} epochs × {epoch_s:.0f}s", text_color=BLUE)
            self.tabs.set("Epochs")
            self._status(f"Epoch analysis done — {n_ep} epochs", GREEN)
        except Exception as _e:
            log.exception("Beat template draw failed")
            messagebox.showerror("Epoch error", str(e))

    # ════════════════════════════════════════════════════════
    #  RECENT FILES
    # ════════════════════════════════════════════════════════

    def _add_recent(self, path):
        if path in self._recent: self._recent.remove(path)
        self._recent.insert(0, path)
        self._recent = self._recent[:8]

    def _open_recent(self):
        if not self._recent:
            messagebox.showinfo("Recent files", "No recent files yet."); return
        win = ctk.CTkToplevel(self)
        win.title("Recent files"); win.geometry("520x280")
        win.configure(fg_color=BG); win.grab_set()
        ctk.CTkLabel(win, text="Recent files", font=FH,
                     text_color=TEXT).pack(padx=16, pady=(14,6), anchor="w")
        for p in self._recent:
            row = ctk.CTkFrame(win, fg_color=PANEL, corner_radius=6)
            row.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(row, text=os.path.basename(p), font=FL,
                         text_color=TEXT, anchor="w").pack(
                side="left", padx=10, pady=6, fill="x", expand=True)
            ctk.CTkButton(row, text="Open", width=64, height=28,
                          fg_color=BLUE, hover_color="#0D47A1",
                          text_color="white", font=FSM,
                          command=lambda p=p: (win.destroy(),
                                               self._load_path(p))
                          ).pack(side="right", padx=6)

    def _load_path(self, path):
        if not os.path.exists(path):
            messagebox.showerror("Not found", "File not found:\n" + path); return
        self._filepath = path
        self.lbl_file.configure(text=os.path.basename(path), text_color=GREEN)
        self._add_recent(path)
        self._preview()

    # ════════════════════════════════════════════════════════
    #  DARK MODE
    # ════════════════════════════════════════════════════════

    def _toggle_dark(self):
        self._dark_mode = not self._dark_mode
        mode = "dark" if self._dark_mode else "light"
        ctk.set_appearance_mode(mode)
        # Update matplotlib style
        if self._dark_mode:
            MP.update(bg="#1A1A2E", axes="#16213E", grid="#2A2A4A",
                      text="#E8E8F0", muted="#7A7A9A", border="#3A3A5A",
                      signal="#5B9BD5", raw="#607D8B")
            plt.rcParams.update({
                "figure.facecolor": "#1A1A2E",
                "axes.facecolor":   "#16213E",
                "grid.color":       "#2A2A4A",
                "axes.labelcolor":  "#E8E8F0",
                "xtick.color":      "#7A7A9A",
                "ytick.color":      "#7A7A9A",
                "text.color":       "#E8E8F0",
            })
        else:
            MP.update(bg="#FFFFFF", axes="#FAFAFA", grid="#EEEEEE",
                      text="#2A2A2A", muted="#9E9E9E", border="#D0D0D0",
                      signal="#1565C0", raw="#90A4AE")
            plt.rcParams.update({
                "figure.facecolor": "#FFFFFF",
                "axes.facecolor":   "#FAFAFA",
                "grid.color":       "#EEEEEE",
                "axes.labelcolor":  "#2A2A2A",
                "xtick.color":      "#9E9E9E",
                "ytick.color":      "#9E9E9E",
                "text.color":       "#2A2A2A",
            })
        # Redraw if data available
        if self._signal_flt is not None:
            self._draw_overview()
            self._draw_detail()
        if self._results is not None:
            self._draw_results()

    # ════════════════════════════════════════════════════════
    #  COPY / SAVE SUMMARY
    # ════════════════════════════════════════════════════════

    def _copy_summary(self):
        if not self._results:
            messagebox.showwarning("No results", "Run analysis first."); return
        text = self.txt_sum.get("1.0", "end")
        self.clipboard_clear(); self.clipboard_append(text)
        self._status("Summary copied to clipboard", GREEN)

    def _save_summary_txt(self):
        if not self._results:
            messagebox.showwarning("No results", "Run analysis first."); return
        sub = self.ent_subject.get()
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        p   = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"{sub}_ECG_summary_{ts}.txt",
            filetypes=[("Text","*.txt"),("All","*.*")])
        if not p: return
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.txt_sum.get("1.0", "end"))
        self._status(f"Summary saved: {os.path.basename(p)}", GREEN)

    # ════════════════════════════════════════════════════════
    #  DRAG AND DROP  (tkinterdnd2 optional)
    # ════════════════════════════════════════════════════════

    def _setup_dnd(self):
        try:
            self.drop_target_register("DND_Files")
            self.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass  # tkinterdnd2 not installed — drag-drop silently disabled

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")
        if path.lower().endswith(".mat"):
            self._load_path(path)

# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not NK_OK:
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        messagebox.showerror("Missing dependency",
                              "Install NeuroKit2:\n  pip install neurokit2")
        r.destroy()
    else:
        app = ECGApp()
        app.mainloop()
