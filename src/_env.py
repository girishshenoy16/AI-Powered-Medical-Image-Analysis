"""Silence TensorFlow/abseil startup noise.

Imported as the very first statement in every entry point so the environment
variables are set *before* TensorFlow's C++ layer initializes.
"""

import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress TensorFlow's Python-level log chatter (e.g. GPU-unavailable note).
logging.getLogger("tensorflow").setLevel(logging.ERROR)
