import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

BASE_LOG_DIR = Path("d:/AD/Visualize/Log")


def normalize_model_name(model_dir: Path, base_dir: Path = BASE_LOG_DIR) -> str:
    """
    Chuẩn hóa tên phương pháp theo folder, ví dụ:
    d:/AD/Visualize/Log/EM_VLM4AD/Adapter -> EM_VLM4AD_Adapter
    """
    rel = model_dir.resolve().relative_to(base_dir.resolve())
    parts = list(rel.parts)
    return "_".join(parts)


def calculate_overall(metrics: Dict[str, Any]) -> float:
    """
    Tính điểm Overall theo công thức bắt buộc:
    Overall = (BLEU4 + METEOR + ROUGE_L + CIDEr / 10) / 4
    """
    bleu4 = float(metrics.get("BLEU-4") or 0.0)
    meteor = float(metrics.get("METEOR") or 0.0)
    rouge_l = float(metrics.get("ROUGE-L") or 0.0)
    cider = float(metrics.get("CIDEr") or 0.0)
    overall = (bleu4 + meteor + rouge_l + (cider / 10.0)) / 4.0
    return round(overall, 4)


def discover_models(base_dir: Path = BASE_LOG_DIR) -> List[Dict[str, Any]]:
    """
    Quét đệ quy thư mục Log để tìm các thư mục chứa file log mô hình.
    """
    models = []
    if not base_dir.exists():
        return models

    for sub in sorted(base_dir.rglob("*")):
        if sub.is_dir():
            has_metrics = (sub / "metrics_display.json").exists()
            has_bench = (sub / "benchmark.json").exists()
            has_progress = len(list(sub.glob("train_progress*.jsonl"))) > 0
            has_preds = (sub / "predictions.jsonl").exists()

            if has_metrics or has_bench or has_progress:
                model_name = normalize_model_name(sub, base_dir)
                rel = sub.relative_to(base_dir)
                group_name = rel.parts[0] if len(rel.parts) > 1 else "Root"
                sub_name = "/".join(rel.parts[1:]) if len(rel.parts) > 1 else rel.parts[0]
                models.append({
                    "model_id": model_name,
                    "group": group_name,
                    "sub_name": sub_name,
                    "folder_path": str(sub),
                    "has_metrics": has_metrics,
                    "has_benchmark": has_bench,
                    "has_train_progress": has_progress,
                    "has_predictions": has_preds,
                })
    return models


def merge_train_progress(model_dir: Path) -> Dict[str, Any]:
    """
    Tự động gộp các file train_progress*.jsonl thông minh (Deduplication Merge):
    - Đọc theo thứ tự tự nhiên (train_progress.jsonl, train_progress1.jsonl, ...)
    - Sử dụng khóa duy nhất (stage, global_step). Nếu trùng, bản ghi từ file sau ghi đè file trước.
    - Trích xuất train_step, validation_end, checkpoint_saved và phân chia stage rõ ràng.
    """
    files = sorted(model_dir.glob("train_progress*.jsonl"), key=lambda p: (len(p.name), p.name))
    
    # Map (stage, global_step) -> step dict
    steps_dict: Dict[tuple, Dict[str, Any]] = {}
    validations: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    stages_order: List[str] = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                event = data.get("event")
                stage = data.get("stage", "default")
                if stage not in stages_order:
                    stages_order.append(stage)

                if event == "train_step":
                    step = data.get("global_step")
                    # Ghi đè nếu trùng để giữ trạng thái resume mới nhất
                    steps_dict[(stage, step)] = data
                elif event == "validation_end":
                    validations.append(data)
                elif event == "checkpoint_saved":
                    checkpoints.append(data)

    # Sort all steps chronologically respecting stages order
    def step_sort_key(item):
        (stage, step), d = item
        stage_idx = stages_order.index(stage) if stage in stages_order else 999
        return (stage_idx, step)

    sorted_steps = [d for k, d in sorted(steps_dict.items(), key=step_sort_key)]

    # Compute stage boundaries & metadata
    stage_boundaries = {}
    for stage in stages_order:
        stage_steps = [s for s in sorted_steps if s.get("stage") == stage]
        if stage_steps:
            stage_boundaries[stage] = {
                "start_step": stage_steps[0].get("global_step"),
                "end_step": stage_steps[-1].get("global_step"),
                "start_epoch": stage_steps[0].get("epoch"),
                "end_epoch": stage_steps[-1].get("epoch"),
                "count": len(stage_steps)
            }

    # Downsample points for smooth frontend rendering if steps are very dense
    # Keep 1 in every N steps if > 1500 steps, but always keep validation & last step
    max_render_steps = 1500
    if len(sorted_steps) > max_render_steps:
        stride = max(1, len(sorted_steps) // max_render_steps)
        downsampled = sorted_steps[::stride]
        if sorted_steps[-1] not in downsampled:
            downsampled.append(sorted_steps[-1])
    else:
        downsampled = sorted_steps

    clean_steps = []
    for s in downsampled:
        clean_steps.append({
            "step": s.get("global_step"),
            "step_in_epoch": s.get("step_in_epoch"),
            "epoch": s.get("epoch"),
            "stage": s.get("stage"),
            "loss": round(float(s.get("loss", 0.0)), 5) if s.get("loss") is not None else None,
            "lr": s.get("lr"),
            "speed": round(float(s.get("steps_per_second", 0.0)), 2) if s.get("steps_per_second") else None,
            "memory_allocated_mb": round(s.get("cuda_memory_allocated", 0) / (1024 * 1024), 1) if s.get("cuda_memory_allocated") else None,
            "memory_peak_mb": round(s.get("cuda_memory_peak", 0) / (1024 * 1024), 1) if s.get("cuda_memory_peak") else None,
            "timestamp": s.get("timestamp")
        })

    clean_validations = []
    for v in validations:
        clean_validations.append({
            "step": v.get("global_step"),
            "epoch": v.get("epoch"),
            "stage": v.get("stage"),
            "val_loss": round(float(v.get("val_loss", 0.0)), 5) if v.get("val_loss") is not None else None,
            "train_loss": round(float(v.get("train_loss", 0.0)), 5) if v.get("train_loss") is not None else None,
            "best_val_loss": round(float(v.get("best_val_loss", 0.0)), 5) if v.get("best_val_loss") is not None else None,
            "timestamp": v.get("timestamp")
        })

    return {
        "stages": stages_order,
        "stage_boundaries": stage_boundaries,
        "total_steps": len(sorted_steps),
        "steps": clean_steps,
        "validations": clean_validations,
        "checkpoints_count": len(checkpoints)
    }


def get_model_metrics(model_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Đọc metrics_display.json (hoặc metrics.json) và tính toán Overall score.
    Tự động chuẩn hóa về thang điểm 100 nếu dữ liệu gốc ở dạng tỷ lệ 0..1.
    Tự động chuẩn hóa CIDEr nếu nhập ở thang điểm 0..15 (ví dụ 3.20 -> 320.0).
    """
    path = model_dir / "metrics_display.json"
    if not path.exists():
        path = model_dir / "metrics.json"
        if not path.exists():
            return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        raw_b1 = float(raw["BLEU-1"]) if "BLEU-1" in raw and raw["BLEU-1"] is not None else None
        raw_b4 = float(raw["BLEU-4"]) if "BLEU-4" in raw and raw["BLEU-4"] is not None else None

        # Nếu BLEU <= 1.0, tự động nhân 100 để đưa về cùng thang đo chuẩn
        factor = 100.0 if ((raw_b1 and 0 < raw_b1 <= 1.0) or (raw_b4 and 0 < raw_b4 <= 1.0)) else 1.0

        def parse_metric(key: str) -> Optional[float]:
            if key not in raw or raw[key] is None:
                return None
            val = float(raw[key]) * factor
            if key == "CIDEr" and val < 15.0:
                val = val * 100.0
            return round(val, 3)

        normalized = {
            "BLEU-1": parse_metric("BLEU-1"),
            "BLEU-2": parse_metric("BLEU-2"),
            "BLEU-3": parse_metric("BLEU-3"),
            "BLEU-4": parse_metric("BLEU-4"),
            "METEOR": parse_metric("METEOR"),
            "ROUGE-L": parse_metric("ROUGE-L"),
            "CIDEr": parse_metric("CIDEr"),
        }
        overall = calculate_overall(normalized)
        normalized["Overall"] = overall
        return normalized
    except Exception:
        return None


def get_model_benchmark(model_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Đọc benchmark.json (an toàn nếu không tồn tại).
    """
    path = model_dir / "benchmark.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        peak_mb = round(raw.get("peak_cuda_memory", 0) / (1024 * 1024), 1) if raw.get("peak_cuda_memory") else None
        return {
            "e2e_mean_ms": round(float(raw.get("e2e_mean_ms", 0.0)), 2),
            "e2e_p50_ms": round(float(raw.get("e2e_p50_ms", 0.0)), 2),
            "e2e_p95_ms": round(float(raw.get("e2e_p95_ms", 0.0)), 2),
            "generation_mean_ms": round(float(raw.get("generation_mean_ms", 0.0)), 2),
            "generation_p50_ms": round(float(raw.get("generation_p50_ms", 0.0)), 2),
            "generation_p95_ms": round(float(raw.get("generation_p95_ms", 0.0)), 2),
            "average_generation_ms_per_output_token": round(float(raw.get("average_generation_ms_per_output_token", 0.0)), 3),
            "tokens_per_second": round(float(raw.get("tokens_per_second", 0.0)), 2),
            "gpu": raw.get("gpu", "N/A"),
            "device": raw.get("device", "N/A"),
            "peak_cuda_memory_mb": peak_mb,
            "python": raw.get("python", "N/A"),
            "torch": raw.get("torch", "N/A")
        }
    except Exception:
        return None
