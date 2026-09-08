import io
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from chara import CharaModel
except ImportError:
    CharaModel = None

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "chara_model_4337.pkl"

# Initialize top-level FastAPI instance
app = FastAPI(
    title="Chara Platform API",
    description="Sparse Distance Reconstruction and Biophysical Validation Platform",
    version="0.2.9"
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
if CharaModel is not None and MODEL_PATH.exists():
    try:
        MODEL = CharaModel.load(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load {MODEL_PATH}: {e}")

def run_survival_inference(df):
    """Core inference execution logic."""
    if MODEL is None or np is None:
        raise ValueError("Chara model bundle or dependencies are not loaded.")
    
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

# Top-level Routes: Serve frontend index.html at root
@app.get("/")
def root():
    index_path = ROOT / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "status": "online",
        "service": "Chara Platform API",
        "version": "0.2.9",
        "institution": "CPG Lab, IIT Mandi"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }

@app.post("/predict")
async def predict_api_fastapi(file: UploadFile = File(...)):
    if pd is None:
        raise HTTPException(status_code=503, detail="Prediction backend dependencies not loaded.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), index_col=0)
        results = run_survival_inference(df)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static asset directories if they exist
for folder in ["data", "assets", "screenshots", "reports"]:
    dir_path = ROOT / folder
    if dir_path.is_dir():
        app.mount(f"/{folder}", StaticFiles(directory=str(dir_path)), name=folder)

# Gradio API & Web Interface (if gradio is installed)
if gr is not None:
    def predict_gradio(file_obj):
        if file_obj is None:
            return {"error": "No file uploaded"}
        if pd is None:
            return {"error": "Pandas not available"}
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

    app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
