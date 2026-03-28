"""
features.py — Peak-based feature extraction from 6-channel CZE-IT signals.

Extracts 399 features per sample: per-channel summary statistics, region-level
features, peak morphology metrics, and cross-channel interaction features.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.stats import skew, kurtosis

from .constants import REGIONS, CHANNELS


def _region_features(signal, prefix, region_name, start, end):
    """Extract summary features for a single signal region."""
    seg = signal[start:end + 1]
    rn = f'{prefix}_{region_name}'
    return {
        f'{rn}_auc':   float(np.trapz(np.abs(seg))),
        f'{rn}_mean':  float(np.mean(seg)),
        f'{rn}_max':   float(np.max(seg)),
        f'{rn}_std':   float(np.std(seg)),
        f'{rn}_sharp': float(np.max(np.abs(np.diff(seg, n=2)))) if len(seg) > 2 else 0.0,
    }


def extract_channel_features(signal, channel_name):
    """Extract all features from a single channel (300-point signal).

    Returns dict of ~55 features: global statistics, region-level features,
    peak morphology, and peak counts per region.
    """
    f = {}
    p = channel_name

    # Global statistics
    f[f'{p}_mean']      = float(np.mean(signal))
    f[f'{p}_std']       = float(np.std(signal))
    f[f'{p}_max']       = float(np.max(signal))
    f[f'{p}_range']     = float(np.ptp(signal))
    f[f'{p}_skew']      = float(skew(signal))
    f[f'{p}_kurt']      = float(kurtosis(signal))
    f[f'{p}_energy']    = float(np.sum(signal ** 2))
    f[f'{p}_auc_total'] = float(np.trapz(np.abs(signal)))

    # Region-level features
    for rn, (rs, re) in REGIONS.items():
        f.update(_region_features(signal, p, rn, rs, re))

    # Region ratios
    if f[f'{p}_auc_total'] > 0:
        f[f'{p}_gamma_ratio'] = f[f'{p}_gamma_auc'] / f[f'{p}_auc_total']
        f[f'{p}_mp_ratio']    = f[f'{p}_mprotein_auc'] / f[f'{p}_auc_total']
    else:
        f[f'{p}_gamma_ratio'] = 0
        f[f'{p}_mp_ratio']    = 0

    # Sharpness (second derivative)
    d2 = np.diff(signal, n=2)
    f[f'{p}_max_sharp']  = float(np.max(np.abs(d2)))
    f[f'{p}_mean_sharp'] = float(np.mean(np.abs(d2)))

    # Peak detection
    pthr = 10 if 'raw_ELP' in channel_name else 5
    pks, props = find_peaks(signal, height=0, distance=5, prominence=pthr)
    f[f'{p}_n_peaks'] = len(pks)

    if len(pks) > 0:
        h = props['peak_heights']
        mi = np.argmax(h)
        f[f'{p}_pk1_h']   = float(h[mi])
        f[f'{p}_pk1_pos'] = int(pks[mi])

        try:
            w, _, _, _ = peak_widths(signal, pks, rel_height=0.5)
            f[f'{p}_pk1_w']    = float(w[mi])
            f[f'{p}_mean_pkw'] = float(np.mean(w))
        except Exception:
            f[f'{p}_pk1_w'] = 0
            f[f'{p}_mean_pkw'] = 0

        try:
            pr, _, _ = peak_prominences(signal, pks)
            f[f'{p}_pk1_prom'] = float(pr[mi])
            f[f'{p}_max_prom'] = float(np.max(pr))
        except Exception:
            f[f'{p}_pk1_prom'] = 0
            f[f'{p}_max_prom'] = 0

        if len(pks) >= 2:
            si = np.argsort(h)[::-1]
            f[f'{p}_pk2_h']    = float(h[si[1]])
            f[f'{p}_pk_ratio'] = float(h[si[0]] / (h[si[1]] + 1e-8))
        else:
            f[f'{p}_pk2_h']    = 0
            f[f'{p}_pk_ratio'] = 0

        for rn, (rs, re) in REGIONS.items():
            rp = [pk for pk in pks if rs <= pk <= re]
            f[f'{p}_npk_{rn}']   = len(rp)
            f[f'{p}_maxpk_{rn}'] = float(max(signal[pk] for pk in rp)) if rp else 0
    else:
        for k in ['pk1_h', 'pk1_pos', 'pk1_w', 'mean_pkw', 'pk1_prom',
                   'max_prom', 'pk2_h', 'pk_ratio']:
            f[f'{p}_{k}'] = 0
        for rn in REGIONS:
            f[f'{p}_npk_{rn}']   = 0
            f[f'{p}_maxpk_{rn}'] = 0

    return f


def extract_cross_channel_features(sample_3d, channels=None):
    """Extract cross-channel interaction features from a single sample.

    Computes κ/λ ratios, immunoglobulin dominance, and combined features
    across electrophoretic regions.
    """
    if channels is None:
        channels = CHANNELS
    ci = {c: i for i, c in enumerate(channels)}
    f = {}

    kp  = sample_3d[ci['dif_Kappa']]
    lm  = sample_3d[ci['dif_Lambda']]
    igg = sample_3d[ci['dif_IgG']]
    iga = sample_3d[ci['dif_IgA']]
    igm = sample_3d[ci['dif_IgM']]

    # κ/λ ratios
    f['kl_auc'] = float(np.trapz(np.abs(kp)) / (np.trapz(np.abs(lm)) + 1e-8))
    f['kl_max'] = float(np.max(kp) / (np.max(lm) + 1e-8))

    for rn, (rs, re) in [('gamma', REGIONS['gamma']), ('mp', REGIONS['mprotein'])]:
        f[f'kl_{rn}']      = float(np.max(kp[rs:re+1]) / (np.max(lm[rs:re+1]) + 1e-8))
        f[f'kl_corr_{rn}'] = float(np.corrcoef(kp[rs:re+1], lm[rs:re+1])[0, 1])

    # Immunoglobulin dominance
    f['igg_v_iga'] = float(np.max(igg) - np.max(iga))
    f['igg_v_igm'] = float(np.max(igg) - np.max(igm))
    igmx = [np.max(igg), np.max(iga), np.max(igm)]
    f['dom_ig_max'] = float(max(igmx))
    f['dom_ig_idx'] = int(np.argmax(igmx))

    for rn, (rs, re) in [('gamma', REGIONS['gamma']), ('mp', REGIONS['mprotein']),
                          ('b2', REGIONS['beta2']), ('tr', REGIONS['transition'])]:
        ir = [np.max(igg[rs:re+1]), np.max(iga[rs:re+1]), np.max(igm[rs:re+1])]
        f[f'dom_ig_{rn}_max'] = float(max(ir))
        f[f'dom_ig_{rn}_idx'] = int(np.argmax(ir))

    # Combined ELP × immunotyping features
    elp = sample_3d[ci['raw_ELP']]
    ms, me = REGIONS['mprotein']
    es = np.max(np.abs(np.diff(elp[ms:me+1], n=2))) if (me - ms) > 2 else 0
    ds = np.max(igg[ms:me+1]) + np.max(iga[ms:me+1]) + np.max(igm[ms:me+1])
    f['elp_sh_x_diff'] = float(es * ds)
    imm = max(np.max(igg[ms:me+1]), np.max(iga[ms:me+1]), np.max(igm[ms:me+1]))
    f['any_mp_50']  = int(imm > 50)
    f['any_mp_100'] = int(imm > 100)

    return f


def extract_all_features(X_3d, channels=None, verbose=True):
    """Extract all peak + cross-channel features from a 3D signal array.

    Args:
        X_3d: ndarray of shape (n_samples, n_channels, n_timepoints)
        channels: list of channel names (default: CHANNELS)
        verbose: print progress every 500 samples

    Returns:
        DataFrame of shape (n_samples, n_features) — typically 399 features
    """
    if channels is None:
        channels = CHANNELS

    all_features = []
    for i in range(X_3d.shape[0]):
        if verbose and i % 500 == 0:
            print(f'  Extracting features: {i}/{X_3d.shape[0]}...')
        ft = {}
        for ci, ch in enumerate(channels):
            ft.update(extract_channel_features(X_3d[i, ci, :], ch))
        ft.update(extract_cross_channel_features(X_3d[i], channels))
        all_features.append(ft)

    df = pd.DataFrame(all_features)
    if verbose:
        print(f'  Extracted {df.shape[1]} features × {df.shape[0]} samples')
    return df


def extract_prealbumin_features(X_3d, channels=None):
    """Extract pre-albumin diffuse elevation features (timepoints 0–30).

    These features capture abnormal signal patterns in the cathodal
    application zone of the differential immunotyping channels.
    """
    if channels is None:
        channels = CHANNELS
    PRE_ALB = (0, 30)
    diff_chs = [ch for ch in channels if ch != 'raw_ELP']
    diff_idxs = {ch: i for i, ch in enumerate(channels)}

    all_f = []
    for i in range(X_3d.shape[0]):
        ft = {}
        for ch in diff_chs:
            ci = diff_idxs[ch]
            seg = X_3d[i, ci, PRE_ALB[0]:PRE_ALB[1] + 1]
            p = f'prealb_{ch}'
            ft[f'{p}_auc']  = float(np.trapz(np.abs(seg)))
            ft[f'{p}_mean'] = float(np.mean(seg))
            ft[f'{p}_max']  = float(np.max(seg))
            ft[f'{p}_std']  = float(np.std(seg))
        all_f.append(ft)
    return pd.DataFrame(all_f)
