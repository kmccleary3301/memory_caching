from __future__ import annotations

import pytest
import torch

from memory_caching.baselines import LogLinearPP, LogLinearPPConfig


def test_loglinear_pp_builds_grm_logarithmic_mc_config() -> None:
    cfg = LogLinearPPConfig(d_model=16, num_heads=4)
    mc_cfg = cfg.build_mc_config()

    assert mc_cfg.aggregation == "grm"
    assert mc_cfg.segmentation == "logarithmic"
    assert mc_cfg.backend == "linear"


def test_loglinear_pp_rejects_non_grm_override() -> None:
    cfg = LogLinearPPConfig(
        d_model=16,
        num_heads=4,
        mc_config_overrides={"aggregation": "residual"},
    )
    with pytest.raises(ValueError, match="aggregation"):
        cfg.build_mc_config()


def test_loglinear_pp_forward_shape() -> None:
    module = LogLinearPP(LogLinearPPConfig(d_model=16, num_heads=4))
    x = torch.randn(2, 8, 16)
    y = module(x)
    assert y.shape == x.shape
