import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from chara import CharaModel
import spaces

ROOT = Path(__file__).resolve().parent
MODEL = CharaModel.load(ROOT / "chara_model_4337.pkl")

@spaces.GPU
def predict_api(file_path):
    df = pd.read_csv(file_path, index_col=0)
    risk, x_scaled, aligned, _, alpha = MODEL.predict(df)
    curves, times = MODEL.survival_curves(x_scaled, alpha)
    
    horizons = np.array([365.0, 1095.0, 1825.0])
    survival = np.array([np.interp(horizons, times, row, left=1.0, right=row[-1]) for row in curves])
    
    return {
        "risk_mean": float(risk.mean()),
        "surv_1y_mean": float(survival[:,0].mean()),
        "surv_3y_mean": float(survival[:,1].mean()),
        "surv_5y_mean": float(survival[:,2].mean()),
        "times": times.tolist(),
        "curves": curves[:50].tolist()
    }

with gr.Blocks() as demo:
    gr.Markdown("# Chara API Backend")
    gr.Markdown("This Space is currently acting as a headless API backend. The UI is deployed separately on Vercel.")
    
    with gr.Row():
        file_in = gr.File(label="Upload CSV")
        json_out = gr.JSON(label="Result JSON")
        
    btn = gr.Button("Run Inference API")
    btn.click(predict_api, inputs=[file_in], outputs=[json_out], api_name="predict")

if __name__ == "__main__":
    demo.launch()
