"""
cds.py — Clinical Decision Support pipeline.

Single-sample CDS inference, confidence zone assignment,
reflex test recommendation engine, and structured report generation.
"""

import numpy as np
from .constants import L2_CLASSES


class CDSResult:
    """Container for a single CDS pipeline output."""
    def __init__(self):
        self.prediction = None
        self.l1_positive = None
        self.l1_proba = None
        self.l2_pred = None
        self.l2_proba = None
        self.l3_pred = None
        self.l3_proba = None
        self.l1_conf = None
        self.l2_conf = None
        self.l3_conf = None
        self.cascade_conf = None
        self.zone = None
        self.conformal_set = None
        self.action = None
        self.reflex_tests = []

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def cds_predict(x_peak, l1_models, l2_models, l3_models, config):
    """Run single-sample CDS pipeline: L1→L2→L3 → confidence → zone → reflex.

    Args:
        x_peak:     (F,) feature vector for one sample
        l1_models:  list of fitted L1 models
        l2_models:  list of fitted L2 models
        l3_models:  list of fitted L3 models
        config:     dict with l1_threshold, conf_high, conf_low

    Returns:
        CDSResult with prediction, confidence, zone, action, reflex tests
    """
    r = CDSResult()
    x = x_peak.reshape(1, -1)
    thr = config.get('l1_threshold', 0.5)

    # L1 — binary detection
    l1p = np.mean([m.predict_proba(x)[0, 1] for m in l1_models])
    r.l1_proba = float(l1p)
    r.l1_positive = l1p >= thr
    r.l1_conf = min(abs(l1p - thr) / max(thr, 1 - thr), 1.0)

    if not r.l1_positive:
        r.prediction = 'NEGATIVE'
        r.cascade_conf = r.l1_conf
    else:
        # L2 — heavy chain
        l2p = np.mean([m.predict_proba(x)[0] for m in l2_models], axis=0)
        r.l2_proba = l2p
        r.l2_pred = L2_CLASSES[np.argmax(l2p)]
        r.l2_conf = float(np.max(l2p))

        # L3 — light chain
        l3p = np.mean([m.predict_proba(x)[0, 1] for m in l3_models])
        r.l3_proba = float(l3p)
        r.l3_pred = 'LAMBDA' if l3p >= 0.5 else 'KAPPA'
        r.l3_conf = abs(l3p - 0.5) * 2

        r.prediction = f'{r.l2_pred}_{r.l3_pred}'
        r.cascade_conf = r.l1_conf * r.l2_conf * max(r.l3_conf, 0.01)

    # Zone assignment
    conf_high = config.get('conf_high', 0.7)
    conf_low = config.get('conf_low', 0.3)
    if r.cascade_conf >= conf_high:
        r.zone = 'HIGH'
        r.action = 'auto_report'
    elif r.cascade_conf >= conf_low:
        r.zone = 'MEDIUM'
        r.action = 'technician_review'
    else:
        r.zone = 'LOW'
        r.action = 'expert_referral'

    # Reflex tests
    r.reflex_tests = get_reflex_tests(r.prediction, r.zone)
    return r


def get_reflex_tests(prediction, zone):
    """Determine recommended reflex tests based on prediction and zone.

    Decision matrix follows IMWG diagnostic criteria, NCCN guidelines,
    and CAP/ADLM consensus recommendations.
    """
    tests = []
    if prediction == 'NEGATIVE':
        if zone == 'LOW':
            tests.append('sFLC')
        return tests

    heavy = prediction.split('_')[0]

    # Universal baseline for all positive predictions
    tests.extend(['Serum IFE', 'sFLC'])

    # Isotype-specific recommendations
    if heavy == 'FREE':
        tests.extend(['24h urine', 'UPEP', 'Renal panel'])
    elif heavy == 'IGA':
        tests.append('Nephelometric IgA quantification')
    elif heavy == 'IGM':
        tests.extend(['Cryoglobulin screen', 'Serum viscosity'])

    # Confidence-stratified escalation
    if zone == 'LOW':
        tests.append('Manual expert review')
    if zone == 'MEDIUM':
        tests.append('Tech verification')

    return tests
