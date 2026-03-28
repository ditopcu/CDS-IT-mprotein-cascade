"""
utils.py — General utilities: Timer, data loading, JSON serialization.
"""

import time
import pickle
import numpy as np
from pathlib import Path


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer() as t:
            model.fit(...)
        print(f'Elapsed: {t.elapsed:.1f}s')
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def load_pickle(path):
    """Load a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    """Save object to pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved: {path}')


def json_serializer(obj):
    """Custom JSON serializer for numpy types and Path objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
