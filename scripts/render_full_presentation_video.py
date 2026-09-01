# Full Presentation Video Generator (MP4)
import os
import cv2
import numpy as np
from pathlib import Path
import imageio.v2 as imageio

WIDTH, HEIGHT = 1920, 1080
FPS = 30

OUTPUT_DESKTOP = Path(r"C:\Users\Samsunh\Desktop\Chara_Full_Presentation_Video.mp4")
OUTPUT_WORKSPACE = Path(r"E:\Sharon\Chara_Full_Presentation_Video.mp4")
VIDEO_MD = Path(r"E:\Sharon\md_simulation.mp4")
ROOT = Path(r"E:\Sharon")

FIG1 = ROOT / "Fig1_OOD_Performance.png"
FIG2 = ROOT / "Fig2_KM_Survival.png"
FIG3 = ROOT / "Fig3_Ablation_Impact.png"
FIG4 = ROOT / "Fig4_Adversarial_Decay.png"
FIG5 = ROOT / "Fig5_Dirichlet_Energy.png"
FIG6 = ROOT / "Fig6_GSEA_Enrichment.png"
RMSD = ROOT / "RMSD_Replicas_300dpi.png"
RG = ROOT / "Rg_Replicas_300dpi.png"

def create_slide_canvas(title: str, step_category: str = "") -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), [248, 250, 252], dtype=np.uint8) # #f8fafc
    
    # Top Header Card
    cv2.rectangle(canvas, (80, 50), (WIDTH - 80, 150), (255, 255, 255), -1)
    cv2.rectangle(canvas, (80, 50), (WIDTH - 80, 150), (226, 232, 240), 2)
    
    # Text
    cat_text = step_category.upper() if step_category else "IIT MANDI · COMPUTATIONAL & PHYSICAL GENOMICS LAB"
    cv2.putText(canvas, cat_text, (110, 88), cv2.FONT_HERSHEY_DUPLEX, 0.65, (235, 99, 37), 1, cv2.LINE_AA) # Blue
    cv2.putText(canvas, title, (110, 130), cv2.FONT_HERSHEY_DUPLEX, 1.05, (15, 23, 42), 2, cv2.LINE_AA) # Navy
    
    # Footer
    cv2.putText(canvas, "Chara: Molecular-Dynamics-Guided Survival Generalization  |  IIT Mandi CPG Lab", (110, HEIGHT - 50), cv2.FONT_HERSHEY_DUPLEX, 0.55, (100, 116, 139), 1, cv2.LINE_AA)
    return canvas

def place_image_contained(canvas: np.ndarray, img_path: Path, x: int, y: int, w: int, h: int):
    if not img_path.exists(): return
    img = cv2.imread(str(img_path))
    if img is None: return
    
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Card background
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (226, 232, 240), 2)
    
    ox = x + (w - nw) // 2
    oy = y + (h - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = resized

def render_slides():
    slides_frames = []
    
    # Slide 1: Title
    s1 = np.full((HEIGHT, WIDTH, 3), [248, 250, 252], dtype=np.uint8)
    cv2.rectangle(s1, (180, 180), (WIDTH - 180, HEIGHT - 180), (255, 255, 255), -1)
    cv2.rectangle(s1, (180, 180), (WIDTH - 180, HEIGHT - 180), (226, 232, 240), 2)
    
    cv2.putText(s1, "COMPUTATIONAL & PHYSICAL GENOMICS LABORATORY · IIT MANDI", (WIDTH//2 - 440, 280), cv2.FONT_HERSHEY_DUPLEX, 0.75, (235, 99, 37), 2, cv2.LINE_AA)
    cv2.putText(s1, "Chara: Molecular-Dynamics-Guided Survival Generalization", (WIDTH//2 - 580, 370), cv2.FONT_HERSHEY_DUPLEX, 1.35, (15, 23, 42), 3, cv2.LINE_AA)
    cv2.putText(s1, "Zero-Retraining Cross-Platform Transcriptomic Oncology", (WIDTH//2 - 400, 440), cv2.FONT_HERSHEY_DUPLEX, 0.85, (100, 116, 139), 1, cv2.LINE_AA)
    
    cv2.putText(s1, "Lead Author: Sharon Melhi Nadar    |    PI: Dr. Kharerin Hungyo", (WIDTH//2 - 400, 560), cv2.FONT_HERSHEY_DUPLEX, 0.85, (15, 23, 42), 2, cv2.LINE_AA)
    cv2.putText(s1, "Indian Institute of Technology Mandi, Himachal Pradesh, India", (WIDTH//2 - 380, 620), cv2.FONT_HERSHEY_DUPLEX, 0.70, (100, 116, 139), 1, cv2.LINE_AA)
    
    slides_frames.append((s1, 3.0)) # 3 seconds
    
    # Slide 2: Generalization Crisis (Fig 1)
    s2 = create_slide_canvas("Cross-Platform Generalization Crisis & Baseline Collapse", "The Clinical Problem")
    place_image_contained(s2, FIG1, 950, 200, 890, 780)
    cv2.rectangle(s2, (80, 200), (900, 980), (255, 255, 255), -1)
    cv2.rectangle(s2, (80, 200), (900, 980), (226, 232, 240), 2)
    cv2.putText(s2, "Why Current Oncology AI Fails in Clinic:", (120, 270), cv2.FONT_HERSHEY_DUPLEX, 0.95, (38, 38, 220), 2, cv2.LINE_AA)
    cv2.putText(s2, "1. DeepSurv (MLP) drops to C = 0.5537 on Microarrays.", (120, 360), cv2.FONT_HERSHEY_DUPLEX, 0.75, (15, 23, 42), 1, cv2.LINE_AA)
    cv2.putText(s2, "2. Random Survival Forests (RSF) invert to C = 0.4041.", (120, 440), cv2.FONT_HERSHEY_DUPLEX, 0.75, (15, 23, 42), 1, cv2.LINE_AA)
    cv2.putText(s2, "3. RNA-seq models overfit to sequencing probe bias.", (120, 520), cv2.FONT_HERSHEY_DUPLEX, 0.75, (15, 23, 42), 1, cv2.LINE_AA)
    cv2.putText(s2, "Chara Solution: Invariant Molecular Dynamics Physics", (120, 640), cv2.FONT_HERSHEY_DUPLEX, 0.90, (74, 163, 22), 2, cv2.LINE_AA)
    cv2.putText(s2, "Achieves gold-standard C = 0.7311 with zero retraining.", (120, 710), cv2.FONT_HERSHEY_DUPLEX, 0.75, (15, 23, 42), 1, cv2.LINE_AA)
    slides_frames.append((s2, 3.5))

    # Slide 3: MD Verification (RMSD & Rg)
    s3 = create_slide_canvas("Triplicate MD Trajectory Convergence & Rg Compaction", "Step 1 Simulation Stability")
    place_image_contained(s3, RMSD, 80, 200, 860, 780)
    place_image_contained(s3, RG, 980, 200, 860, 780)
    slides_frames.append((s3, 3.0))
    
    # Slide 4: Dirichlet Energy Proof (Fig 5)
    s4 = create_slide_canvas("Empirical Dirichlet Energy Reduction in TCGA (p < 10^-15)", "Step 3 Proof")
    place_image_contained(s4, FIG5, 950, 200, 890, 780)
    cv2.rectangle(s4, (80, 200), (900, 980), (255, 255, 255), -1)
    cv2.rectangle(s4, (80, 200), (900, 980), (226, 232, 240), 2)
    cv2.putText(s4, "Topological Roughness Elimination:", (120, 270), cv2.FONT_HERSHEY_DUPLEX, 0.95, (74, 163, 22), 2, cv2.LINE_AA)
    cv2.putText(s4, "• Evaluated across n=503 TCGA lung cancer patients.", (120, 360), cv2.FONT_HERSHEY_DUPLEX, 0.75, (15, 23, 42), 1, cv2.LINE_AA)
    cv2.putText(s4, "• Paired Wilcoxon p = 2.41 x 10^-19 reduction.", (120, 440), cv2.FONT_HERSHEY_DUPLEX, 0.75, (15, 23, 42), 1, cv2.LINE_AA)
    cv2.putText(s4, "• Proves MD variances actively suppress platform noise.", (120, 520), cv2.FONT_HERSHEY_DUPLEX, 0.75, (15, 23, 42), 1, cv2.LINE_AA)
    slides_frames.append((s4, 3.0))

    # Slide 5: Kaplan-Meier Survival (Fig 2)
    s5 = create_slide_canvas("Monotonic Time-Horizon Scaling & Kaplan-Meier Strata", "Benchmark Results")
    place_image_contained(s5, FIG2, 950, 200, 890, 780)
    cv2.rectangle(s5, (80, 200), (900, 980), (255, 255, 255), -1)
    cv2.rectangle(s5, (80, 200), (900, 980), (226, 232, 240), 2)
    cv2.putText(s5, "Horizon Scaling Performance:", (120, 270), cv2.FONT_HERSHEY_DUPLEX, 0.95, (235, 99, 37), 2, cv2.LINE_AA)
    cv2.putText(s5, "• 1-Year AUC: 0.7463", (120, 360), cv2.FONT_HERSHEY_DUPLEX, 0.85, (15, 23, 42), 2, cv2.LINE_AA)
    cv2.putText(s5, "• 3-Year AUC: 0.7826", (120, 440), cv2.FONT_HERSHEY_DUPLEX, 0.85, (15, 23, 42), 2, cv2.LINE_AA)
    cv2.putText(s5, "• 5-Year AUC: 0.8195 (Peak Long-Term Calibration)", (120, 520), cv2.FONT_HERSHEY_DUPLEX, 0.85, (74, 163, 22), 2, cv2.LINE_AA)
    cv2.putText(s5, "• Kaplan-Meier Separation: Log-rank p < 10^-6", (120, 620), cv2.FONT_HERSHEY_DUPLEX, 0.85, (38, 38, 220), 2, cv2.LINE_AA)
    slides_frames.append((s5, 3.5))

    # Slide 6: Ablation & Adversarial (Fig 3 & Fig 4)
    s6 = create_slide_canvas("Biophysical Ablation & Adversarial Noise Robustness", "Scientific Dissection")
    place_image_contained(s6, FIG3, 80, 200, 860, 780)
    place_image_contained(s6, FIG4, 980, 200, 860, 780)
    slides_frames.append((s6, 3.5))

    # Slide 7: GSEA Pathways (Fig 6)
    s7 = create_slide_canvas("Biological Gene Set Enrichment & Clinical Independence", "Translation")
    place_image_contained(s7, FIG6, 950, 200, 890, 780)
    cv2.rectangle(s7, (80, 200), (900, 980), (255, 255, 255), -1)
    cv2.rectangle(s7, (80, 200), (900, 980), (226, 232, 240), 2)
    cv2.putText(s7, "Enriched MSigDB Hallmarks:", (120, 270), cv2.FONT_HERSHEY_DUPLEX, 0.95, (74, 163, 22), 2, cv2.LINE_AA)
    cv2.putText(s7, "• KRAS Signaling, EMT, Hypoxia (FDR < 0.05)", (120, 360), cv2.FONT_HERSHEY_DUPLEX, 0.75, (15, 23, 42), 1, cv2.LINE_AA)
    cv2.putText(s7, "Independent Multivariate Hazard:", (120, 480), cv2.FONT_HERSHEY_DUPLEX, 0.95, (235, 99, 37), 2, cv2.LINE_AA)
    cv2.putText(s7, "• HR = 3.81 (95% CI: 3.06-4.74, p = 4.48 x 10^-33)", (120, 560), cv2.FONT_HERSHEY_DUPLEX, 0.80, (15, 23, 42), 2, cv2.LINE_AA)
    cv2.putText(s7, "• Adjusted for Age, Gender, and Pathological TNM Stage.", (120, 630), cv2.FONT_HERSHEY_DUPLEX, 0.70, (100, 116, 139), 1, cv2.LINE_AA)
    slides_frames.append((s7, 3.5))

    return slides_frames

def compile_full_video():
    print("Building full presentation video sequence...")
    slides = render_slides()
    
    writer = imageio.get_writer(str(OUTPUT_WORKSPACE), fps=FPS, codec="libx264", quality=9)
    
    # Write slides
    for idx, (img, dur) in enumerate(slides):
        n_frames = int(dur * FPS)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for _ in range(n_frames):
            writer.append_data(rgb_frame)
            
        # If this is after Slide 2, insert the MD Simulation video seamlessly!
        if idx == 1 and VIDEO_MD.exists():
            print("Inserting live MD Trajectory video sequence...")
            cap = cv2.VideoCapture(str(VIDEO_MD))
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Fit frame in slide template
                md_slide = create_slide_canvas("Live Molecular Dynamics Trajectory (Switch I/II Cryptic Opening)", "Step 1 Simulation Video")
                fh, fw = frame.shape[:2]
                scale = min(1700 / fw, 760 / fh)
                nw, nh = int(fw * scale), int(fh * scale)
                rf = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                
                ox = 80 + (1760 - nw) // 2
                oy = 200 + (760 - nh) // 2
                md_slide[oy:oy+nh, ox:ox+nw] = rf
                
                writer.append_data(cv2.cvtColor(md_slide, cv2.COLOR_BGR2RGB))
            cap.release()
            
    writer.close()
    
    if OUTPUT_WORKSPACE.exists():
        import shutil
        shutil.copy2(OUTPUT_WORKSPACE, OUTPUT_DESKTOP)
        print(f"SUCCESS: Full Presentation Video generated at:\n -> {OUTPUT_DESKTOP}\n -> {OUTPUT_WORKSPACE}")

if __name__ == "__main__":
    compile_full_video()
