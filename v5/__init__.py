"""Return-prediction backbone: selective state-space model with a distributional head.

Submodules:

* :mod:`v5.mamba_ref` — CPU-executable Mamba block with ``mamba_ssm`` parameter parity.

The model definition, loss, and inference helpers live in the top-level
``v5_backbone`` module; the feature builder lives in ``features_v5``.
"""
