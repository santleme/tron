import os

import pytest

from training.kaggle.run_kaggle import constrain_cuda_visibility


def test_constrain_cuda_visibility_defaults_to_first_kaggle_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    constrain_cuda_visibility()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_constrain_cuda_visibility_keeps_only_first_configured_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 3")

    constrain_cuda_visibility()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_constrain_cuda_visibility_rejects_empty_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " , 3")

    with pytest.raises(RuntimeError, match="at least one visible CUDA device"):
        constrain_cuda_visibility()


def test_campaign_train_dispatches_registered_config_without_generic_evaluation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from training.kaggle import run_kaggle
    from training.ppo import train as train_module

    calls = {}
    monkeypatch.setattr(run_kaggle, "ensure_support_dependencies", lambda torch: None)

    def fake_train(config, output, **kwargs):
        calls.update(config=config, output=output, kwargs=kwargs)
        return {"training": {"updates": [{}] * config.total_updates}}

    monkeypatch.setattr(train_module, "train", fake_train)
    settings = {
        "run_id": "12345678-1234-4234-8234-123456789abc",
        "train_config": {"device": "cuda", "total_updates": 1, "checkpoint_every": 1},
        "g008_campaign_train": {"campaign_id": "g008-competitive-20261021-v2"},
    }
    runtime = {"format": "circuit-ai-kaggle-runtime-evidence/1", "run_id": settings["run_id"]}
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    report = run_kaggle.run_campaign_train(settings, tmp_path, runtime, torch)

    assert len(report["training"]["updates"]) == 1
    assert calls["kwargs"] == {"export_onnx": True, "run_evaluation": False,
                               "g008_campaign_train": settings["g008_campaign_train"],
                               "run_id": settings["run_id"]}
    assert (tmp_path / "kaggle-execution-evidence.json").is_file()
