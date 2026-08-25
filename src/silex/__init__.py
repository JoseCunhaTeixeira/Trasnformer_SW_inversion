"""silex: model architecture, training, evaluation, and data generation for
the Silex dispersion-curve-to-soil-profile Transformer.

Sets a fallback Keras backend *before* anything in this package imports
keras, since Keras 3 picks a backend at first import and, left to its own
default detection, reaches for TensorFlow -- which has no wheels at all on
Python 3.14, and even where it's installed is a much heavier stack than
training likely needs. `torch` is a real pyproject.toml dependency for the
same reason: Keras 3 structurally needs *some* backend package installed
just to import at all, not merely selected, and torch is the one that
reliably has wheels across Python versions TensorFlow doesn't support.

`setdefault` only fills KERAS_BACKEND in if it isn't already set, so a real
GPU/TensorFlow training environment that sets it explicitly (and has
tensorflow installed) is fully respected -- this only exists so the
package works out of the box on whatever machine runs it otherwise,
`python -m pytest` included.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

if "tensorflow" not in sys.modules:
    # keras's own import machinery (keras.src.tree.optree_impl) and
    # keras-hub's top-level __init__ (what `keras_nlp` now re-exports, used
    # by model.py for TransformerEncoder/TransformerDecoder/etc.) both
    # unconditionally import tensorflow-only code paths even when the
    # active backend is torch. Traced and fixed the same way in sigpipe's
    # algorithms/inversion/dispersion_curve/petro/silex.py -- see that
    # file's comments for the full explanation of each of the four
    # attributes below. Remove once keras/keras-hub stop doing this or
    # TensorFlow ships wheels for your Python version.
    class _StubTensorShape:
        pass

    class _StubDType:
        pass

    class _StubTypeSpec:
        pass

    _tf_stub_spec = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    _tf_stub = importlib.util.module_from_spec(_tf_stub_spec)
    setattr(_tf_stub, "TensorShape", _StubTensorShape)  # noqa: B010
    setattr(_tf_stub, "DType", _StubDType)  # noqa: B010
    setattr(_tf_stub, "TypeSpec", _StubTypeSpec)  # noqa: B010
    setattr(_tf_stub, "dtypes", object())  # noqa: B010
    sys.modules["tensorflow"] = _tf_stub
