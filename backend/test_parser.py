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

    adapter_meta = next(m for m in models if m["model_id"] == "EM_VLM4AD_Adapter")
    assert adapter_meta["has_metrics"] is True
    assert adapter_meta["has_benchmark"] is True
    assert adapter_meta["has_train_progress"] is True

    freeze_meta = next(m for m in models if m["model_id"] == "EM_VLM4AD_Freeze")
    assert freeze_meta["has_metrics"] is True
    assert freeze_meta["has_benchmark"] is False  # Freeze doesn't have benchmark.json yet


def test_merge_train_progress_adapter():
    adapter_dir = BASE_DIR / "EM_VLM4AD" / "Adapter"
    res = merge_train_progress(adapter_dir)
    assert res["total_steps"] > 0
    assert "finetune" in res["stages"]
    # Total steps in Adapter finetune ends at 106690
    assert len(res["validations"]) >= 10
    # Steps should be monotonic within stage
    steps = [s["step"] for s in res["steps"]]
    assert steps == sorted(steps)


def test_merge_train_progress_freeze():
    freeze_dir = BASE_DIR / "EM_VLM4AD" / "Freeze"
    res = merge_train_progress(freeze_dir)
    assert "align" in res["stages"]
    assert "finetune" in res["stages"]
    assert len(res["validations"]) >= 6
