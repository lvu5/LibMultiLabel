"""Opt-in process memory checkpoints (LML_PROFILE_MEMORY=1)."""

import logging
import os

import psutil


def log_memory(stage):
    if os.environ.get("LML_PROFILE_MEMORY") != "1":
        return
    rss = psutil.Process().memory_info().rss / 2**20
    # ru_maxrss is a lifetime high-water mark, not the peak of this phase.
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak /= 2**20 if sys.platform == "darwin" else 1024
        logging.info("MEMORY %s: RSS=%.1f MiB; process peak=%.1f MiB", stage, rss, peak)
    except ImportError:
        logging.info("MEMORY %s: RSS=%.1f MiB", stage, rss)
