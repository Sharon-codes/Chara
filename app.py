import io
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr

from chara import CharaModel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "chara_model_4337.pkl"

# Initialize top-level FastAPI instance
app = FastAPI(
    title="Chara Survival Inference API",
    description="Thermodynamic Graph Laplacian Manifold Alignment for Survival Inference",
    version="0.1.6"
)

# Enable CORS for frontend web integration (Vercel & local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load frozen Chara model bundle if available
MODEL = None
if MODEL_PATH.exists():
    try:
        MODEL = CharaModel.load(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load {MODEL_PATH}: {e}")

def run_survival_inference(df: pd.DataFrame):
    """Core inference execution logic."""
    if MODEL is None:
        raise ValueError("Chara model bundle (chara_model_4337.pkl) is not loaded.")
    
    risk, x_scaled, aligned, _, alpha = MODEL.predict(df)
    curves, times = MODEL.survival_curves(x_scaled, alpha)
    
    horizons = np.array([365.0, 1095.0, 1825.0]) # 1-Yr, 3-Yr, 5-Yr (in days)
    survival = np.array([np.interp(horizons, times, row, left=1.0, right=row[-1]) for row in curves])
    
    return {
        "risk_mean": float(risk.mean()),
        "surv_1y_mean": float(survival[:, 0].mean()),
        "surv_3y_mean": float(survival[:, 1].mean()),
        "surv_5y_mean": float(survival[:, 2].mean()),
        "times": times.tolist(),
        "curves": curves[:100].tolist(),
        "num_patients": int(len(df))
    }

# Top-level FastAPI Routes
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Chara Survival Inference API",
        "version": "0.1.6",
        "institution": "CPG Lab, IIT Mandi",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "gradio": "/gradio"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }

@app.post("/predict")
async def predict_api_fastapi(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), index_col=0)
        results = run_survival_inference(df)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gradio API & Web Interface
def predict_gradio(file_obj):
    if file_obj is None:
        return {"error": "No file uploaded"}
    try:
        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        df = pd.read_csv(file_path, index_col=0)
        return run_survival_inference(df)
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks(title="Chara Survival API Backend") as demo:
    gr.Markdown("# Chara Survival API Backend")
    gr.Markdown("Computational & Physical Genomics Laboratory · Indian Institute of Technology Mandi")
    
    with gr.Row():
        file_in = gr.File(label="Upload Cohort CSV", file_types=[".csv"])
        json_out = gr.JSON(label="Inference Results (JSON)")
        
    btn = gr.Button("Run Inference API", variant="primary")
    btn.click(predict_gradio, inputs=[file_in], outputs=[json_out], api_name="predict")

# Mount Gradio app onto FastAPI app
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
