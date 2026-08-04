"""Tests for fail-fast neural training device policy."""

from __future__ import annotations

import pytest
import torch

from generals_bot.training.device_policy import DevicePolicyError, resolve_training_device


def test_resolve_rejects_explicit_cpu_when_cuda_required():
    policy = {
        "require_cuda_for_neural_training": True,
        "allow_cpu_training": False,
        "fail_on_silent_cpu_fallback": True,
        "default_cuda_device": "cuda:0",
    }
    with pytest.raises(DevicePolicyError, match="CPU neural training forbidden"):
        resolve_training_device("cpu", policy=policy, context="unit_test")


def test_resolve_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = {
        "require_cuda_for_neural_training": True,
        "allow_cpu_training": False,
        "fail_on_silent_cpu_fallback": True,
        "default_cuda_device": "cuda:0",
    }
    assert resolve_training_device("auto", policy=policy) == "cuda:0"
    assert resolve_training_device("cuda", policy=policy) == "cuda:0"


def test_resolve_allows_cpu_when_policy_permits():
    policy = {
        "require_cuda_for_neural_training": False,
        "allow_cpu_training": True,
        "fail_on_silent_cpu_fallback": False,
        "default_cuda_device": "cuda:0",
    }
    assert resolve_training_device("cpu", policy=policy) == "cpu"
