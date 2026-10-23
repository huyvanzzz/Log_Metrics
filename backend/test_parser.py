import pytest
from pathlib import Path
from backend.parser import (
    normalize_model_name,
    calculate_overall,
    discover_models,
    merge_train_progress,
    get_model_metrics,
    get_model_benchmark,
)

BASE_DIR = Path("d:/AD/Visualize/Log")


def test_normalize_model_name():
    p1 = BASE_DIR / "EM_VLM4AD" / "Adapter"
    assert normalize_model_name(p1, BASE_DIR) == "EM_VLM4AD_Adapter"

    p2 = BASE_DIR / "EM_VLM4AD" / "Freeze"
    assert normalize_model_name(p2, BASE_DIR) == "EM_VLM4AD_Freeze"


def test_calculate_overall():
    # Overall = (BLEU4 + METEOR + ROUGE_L + CIDEr / 10) / 4
    metrics = {
        "BLEU-4": 40.0,
        "METEOR": 30.0,
        "ROUGE-L": 70.0,
        "CIDEr": 300.0  # -> 30.0
    }
    # (40 + 30 + 70 + 30) / 4 = 170 / 4 = 42.5
    assert calculate_overall(metrics) == 42.5


def test_discover_models():
    models = discover_models(BASE_DIR)
    model_ids = [m["model_id"] for m in models]
    assert "EM_VLM4AD_Adapter" in model_ids
    assert "EM_VLM4AD_Freeze" in model_ids
    assert "EM_VLM4AD_Base" in model_ids

    adapter_meta = next(m for m in models if m["model_id"] == "EM_VLM4AD_Adapter")
    assert adapter_meta["has_metrics"] is True
    assert adapter_meta["has_benchmark"] is True
    assert adapter_meta["has_train_progress"] is True

    base_meta = next(m for m in models if m["model_id"] == "EM_VLM4AD_Base")
    assert base_meta["has_metrics"] is True
    assert base_meta["has_benchmark"] is False


def test_merge_train_progress_adapter():
    adapter_dir = BASE_DIR / "EM_VLM4AD" / "Adapter"
    res = merge_train_progress(adapter_dir)
    assert res["total_steps"] > 0
    assert "finetune" in res["stages"]
    assert len(res["validations"]) >= 10
    # Steps should be monotonic within each stage
    for stage in res["stages"]:
        stage_steps = [s["step"] for s in res["steps"] if s["stage"] == stage]
        assert stage_steps == sorted(stage_steps)


def test_base_model_metrics():
    base_dir = BASE_DIR / "EM_VLM4AD" / "Base"
    metrics = get_model_metrics(base_dir)
    assert metrics is not None
    assert metrics["BLEU-4"] == 45.36
    assert metrics["METEOR"] == 34.49
    assert metrics["ROUGE-L"] == 71.98
    assert metrics["CIDEr"] == 320.0
    # Overall = (45.36 + 34.49 + 71.98 + 32.0) / 4 = 183.83 / 4 = 45.9575
    assert metrics["Overall"] == 45.9575
    assert metrics["BLEU-1"] is None


def test_merge_train_progress_freeze():
    freeze_dir = BASE_DIR / "EM_VLM4AD" / "Freeze"
    res = merge_train_progress(freeze_dir)
    assert "align" in res["stages"]
    assert "finetune" in res["stages"]
    assert len(res["validations"]) >= 6


def test_merge_train_progress_epoch_aggregation(tmp_path):
    import json
    log_file = tmp_path / "train_progress.jsonl"
    entries = [
        {"event": "epoch_start", "stage": "finetune", "epoch": 1, "global_step": 0},
        {"event": "train_step", "stage": "finetune", "epoch": 1, "global_step": 1, "loss": 1.2},
        {"event": "train_step", "stage": "finetune", "epoch": 1, "global_step": 2, "loss": 0.8},
        {"event": "epoch_end", "stage": "finetune", "epoch": 1, "train_loss": 1.0, "val_loss": 0.5},
        {"event": "epoch_start", "stage": "finetune", "epoch": 2, "global_step": 2},
        {"event": "train_step", "stage": "finetune", "epoch": 2, "global_step": 3, "loss": 0.6},
        {"event": "validation_end", "stage": "finetune", "epoch": 2, "global_step": 3, "val_loss": 0.4},
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    res = merge_train_progress(tmp_path)
    assert "epochs" in res
    epochs = res["epochs"]
    assert len(epochs) == 2
    # Epoch 1: avg loss = (1.2 + 0.8) / 2 = 1.0, val_loss = 0.5
    assert epochs[0]["epoch"] == 1
    assert epochs[0]["stage"] == "finetune"
    assert epochs[0]["train_loss"] == 1.0
    assert epochs[0]["val_loss"] == 0.5
    assert epochs[0]["steps_count"] == 2

    # Epoch 2: avg loss = 0.6, val_loss = 0.4
    assert epochs[1]["epoch"] == 2
    assert epochs[1]["train_loss"] == 0.6
    assert epochs[1]["val_loss"] == 0.4
    assert epochs[1]["steps_count"] == 1

