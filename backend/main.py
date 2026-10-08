import sys
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.parser import (
    BASE_LOG_DIR,
    discover_models,
    merge_train_progress,
    get_model_metrics,
    get_model_benchmark,
    calculate_overall,
)

app = FastAPI(title="AD Model Visualizer API", version="1.0.0")

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
def list_models():
    """Liệt kê danh sách tất cả các mô hình tìm thấy trong thư mục Log."""
    return discover_models(BASE_LOG_DIR)


@app.get("/api/metrics")
def get_metrics_table():
    """
    Trả về bảng so sánh metrics của tất cả các mô hình,
    kèm tính toán điểm Overall = (BLEU4 + METEOR + ROUGE_L + CIDEr/10) / 4.
    """
    models = discover_models(BASE_LOG_DIR)
    results = []
    for m in models:
        metrics = get_model_metrics(Path(m["folder_path"]))
        if metrics:
            results.append({
                "model_id": m["model_id"],
                "folder_path": m["folder_path"],
                **metrics
            })
    return results


@app.get("/api/benchmark")
def get_benchmark_table():
    """
    Trả về bảng so sánh benchmark của tất cả các mô hình.
    Xử lý an toàn nếu mô hình chưa có file benchmark.json (như Freeze).
    """
    models = discover_models(BASE_LOG_DIR)
    results = []
    for m in models:
        bench = get_model_benchmark(Path(m["folder_path"]))
        results.append({
            "model_id": m["model_id"],
            "folder_path": m["folder_path"],
            "has_benchmark": bench is not None,
            "data": bench
        })
    return results


@app.get("/api/tradeoff")
def get_tradeoff_scatter():
    """
    Trả về dữ liệu Scatter Plot đánh đổi Tốc độ vs Chất lượng:
    - x: average_generation_ms_per_output_token (ms/token)
    - y: Overall score
    - Metadata: chi tiết metrics và benchmark của từng mô hình.
    """
    models = discover_models(BASE_LOG_DIR)
    points = []
    for m in models:
        folder = Path(m["folder_path"])
        metrics = get_model_metrics(folder)
        bench = get_model_benchmark(folder)
        
        overall = metrics.get("Overall") if metrics else None
        avg_ms_per_token = bench.get("average_generation_ms_per_output_token") if bench else None

        points.append({
            "model_id": m["model_id"],
            "x_latency_per_token": avg_ms_per_token,
            "y_overall": overall,
            "tokens_per_second": bench.get("tokens_per_second") if bench else None,
            "peak_vram_mb": bench.get("peak_cuda_memory_mb") if bench else None,
            "e2e_mean_ms": bench.get("e2e_mean_ms") if bench else None,
            "metrics": metrics,
            "has_benchmark": bench is not None
        })
    return points


@app.get("/api/loss")
def get_loss_data(models: Optional[str] = Query(None, description="Danh sách model IDs cách nhau dấu phẩy")):
    """
    Trả về chuỗi Loss sau khi gộp thông minh (deduplication) cho các model được chọn.
    Hỗ trợ cả 2 stage (align + finetune) trong cùng 1 tập dữ liệu.
    """
    discovered = {m["model_id"]: Path(m["folder_path"]) for m in discover_models(BASE_LOG_DIR)}
    
    if models:
        selected_ids = [s.strip() for s in models.split(",") if s.strip()]
    else:
        # Mặc định lấy tối đa 5 model đầu tiên
        selected_ids = list(discovered.keys())[:5]

    response = {}
    for mid in selected_ids:
        if mid in discovered:
            progress_data = merge_train_progress(discovered[mid])
            response[mid] = progress_data
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
