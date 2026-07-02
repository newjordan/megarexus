"""Calibration plumbing for the Gaussian Correction Depth (GCD) layer.

This package is part of the trusted LeWM core. It may contain deterministic
alignment and calibration math, but it must not contain live external-agent
clients or long-latency probes.

Planned:

- ``whitening.py`` — empirical whitening of a paired (auxiliary, latent)
  cache into zero-mean unit-cov form before alignment.
- ``procrustes.py`` — orthogonal Procrustes alignment from the whitened
  auxiliary space into the chassis latent basis.

Experimental goal-inference agents live under ``loom_layers`` until they have
their own tests, latency evidence, condition manifest, and promotion gate.
"""
