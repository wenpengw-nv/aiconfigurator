# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EPD three-stage sweep: encode pool joins the rate match analytically."""

import math

import pandas as pd
import pytest

from aiconfigurator.sdk import config
from aiconfigurator.sdk import sweep as sweep_mod
from aiconfigurator.sdk.sweep import _RATE_MATCH_ENCODER_DEGRADATION

pytestmark = pytest.mark.unit

_P_ROW = {
    "model": "test-model",
    "isl": 4000,
    "osl": 500,
    "prefix": 0,
    "concurrency": 1,
    "bs": 1,
    "global_bs": 1,
    "tp": 4,
    "pp": 1,
    "dp": 1,
    "moe_tp": 1,
    "moe_ep": 1,
    "parallel": "tp4pp1dp1",
    "ttft": 80.0,
    "tpot": 0.0,
    "seq/s": 10.0,
    "seq/s/gpu": 2.5,
    "num_total_gpus": 4,
    "tokens/s/user": 0.0,
    "gemm": "fp8",
    "kvcache": "fp8",
    "fmha": "fp8",
    "moe": "fp8",
    "comm": "half",
    "memory": 12.3,
    "backend": "trtllm",
    "version": "1.3.0",
    "system": "h200_sxm",
    "power_w": 500.0,
}

_D_ROW = dict(
    _P_ROW,
    concurrency=64,
    bs=64,
    global_bs=64,
    tp=2,
    parallel="tp2pp1dp1",
    ttft=0.0,
    tpot=25.0,
    seq_s=5.0,
    num_total_gpus=2,
)
_D_ROW["seq/s"] = 5.0
_D_ROW["seq/s/gpu"] = 2.5
del _D_ROW["seq_s"]

_E_CAND = {
    "latency": 40.0,
    "seq/s": 25.0,
    "bs": 2,
    "tp": 1,
    "pp": 1,
    "num_total_gpus": 1,
    "parallel": "tp1pp1",
    "memory": 4.5,
    "power_w": 300.0,
    "backend": "trtllm",
    "version": "1.3.0",
    "system": "h200_sxm",
}


def test_epd_three_stage_sweep_is_internally_consistent(monkeypatch):
    """The selected EPD row must satisfy the three-stage rate-match invariants."""

    def _fake_candidates(**kwargs):
        return pd.DataFrame([_P_ROW if kwargs["role"] == "prefill" else _D_ROW])

    monkeypatch.setattr(sweep_mod, "_get_disagg_worker_candidates", _fake_candidates)
    monkeypatch.setattr(sweep_mod, "build_encoder_worker_candidates", lambda **kwargs: [dict(_E_CAND)])

    rc = config.RuntimeConfig(batch_size=1, isl=4000, osl=500)
    rc.ttft = 1000.0
    rc.tpot = 50.0

    df = sweep_mod.sweep_disagg(
        model_path="test-model",
        runtime_config=rc,
        prefill_database=None,
        prefill_backend_name="trtllm",
        prefill_model_config=config.ModelConfig(),
        prefill_parallel_config_list=[(4, 1, 1, 1, 1, 1)],
        prefill_latency_correction=1.0,
        decode_database=None,
        decode_backend_name="trtllm",
        decode_model_config=config.ModelConfig(),
        decode_parallel_config_list=[(2, 1, 1, 1, 1, 1)],
        decode_latency_correction=1.0,
        prefill_num_worker_list=list(range(1, 33)),
        decode_num_worker_list=list(range(1, 33)),
        enable_epd=True,
    )

    assert len(df) > 0
    row = df.iloc[0]
    p_num, d_num, e_num = int(row["(p)workers"]), int(row["(d)workers"]), int(row["(e)workers"])
    assert e_num >= 1

    # Analytic encode pool: smallest count that is not the bottleneck.
    pd_rate = min(_P_ROW["seq/s"] * p_num * 0.9, _D_ROW["seq/s"] * d_num * 0.92)
    assert e_num == max(1, math.ceil(pd_rate / (_E_CAND["seq/s"] * _RATE_MATCH_ENCODER_DEGRADATION)))

    # Three-stage rate match and GPU accounting include the encode pool.
    expected_seq_s = min(pd_rate, _E_CAND["seq/s"] * e_num * _RATE_MATCH_ENCODER_DEGRADATION)
    assert row["seq/s"] == pytest.approx(expected_seq_s, rel=1e-3)
    assert row["num_total_gpus"] == 4 * p_num + 2 * d_num + e_num

    # TTFT = encode latency + corrected prefill ttft (no 1.8x on the E stage).
    assert row["ttft"] == pytest.approx(_E_CAND["latency"] + 1.8 * _P_ROW["ttft"], rel=1e-3)
