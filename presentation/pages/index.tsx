import React, { useState, useEffect, useRef } from 'react';
import Head from 'next/head';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronLeft,
  ChevronRight,
  Maximize,
  Minimize,
  Moon,
  Sun,
  Play,
  Pause,
  RotateCcw,
  BookOpen,
  Grid,
  Sparkles,
  Layers,
  Activity,
  CheckCircle2,
  ExternalLink,
  Copy,
  Sliders,
  ShieldCheck,
  TrendingUp,
  Cpu,
  Flame
} from 'lucide-react';

export default function PresentationDeck() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isDark, setIsDark] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showThumbnails, setShowThumbnails] = useState(false);
  const [showNotes, setShowNotes] = useState(false);
  const [copiedPip, setCopiedPip] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);

  const totalSlides = 15;

  // Presenter Timer
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = secs % 60;
    return `${String(mins).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' || e.code === 'ArrowRight' || e.code === 'KeyL') {
        e.preventDefault();
        nextSlide();
      } else if (e.code === 'ArrowLeft' || e.code === 'KeyH') {
        e.preventDefault();
        prevSlide();
      } else if (e.code === 'KeyF') {
        e.preventDefault();
        toggleFullscreen();
      } else if (e.code === 'KeyT') {
        e.preventDefault();
        setShowThumbnails((prev) => !prev);
      } else if (e.code === 'KeyN') {
        e.preventDefault();
        setShowNotes((prev) => !prev);
      } else if (e.code === 'KeyD') {
        e.preventDefault();
        setIsDark((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentSlide]);

  const nextSlide = () => {
    if (currentSlide < totalSlides - 1) {
      setCurrentSlide((prev) => prev + 1);
    }
  };

  const prevSlide = () => {
    if (currentSlide > 0) {
      setCurrentSlide((prev) => prev - 1);
    }
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  const slideTitles = [
    { title: "Title & Research Vision", cat: "Symposium Overview" },
    { title: "The Cross-Platform Generalization Crisis", cat: "The Problem" },
    { title: "The 4-Step Chara Architecture", cat: "Methodology" },
    { title: "Step 1: MARTINI 3 Coarse-Grained MD", cat: "Simulation Setup" },
    { title: "Step 1 Video: KRAS G12D Trajectory", cat: "Interactive Trajectory" },
    { title: "Triplicate RMSD & Radius of Gyration Stability", cat: "MD Verification" },
    { title: "Step 2: Contact Variance to Graph Weights", cat: "Biophysical Weighting" },
    { title: "Step 3: Interactive Spectral Heat Diffusion", cat: "Filtering Engine" },
    { title: "Empirical Dirichlet Energy Reduction (p < 10⁻¹⁹)", cat: "Topological Proof" },
    { title: "Step 4: 4,337-Gene Signature & 58 Biomarkers", cat: "Feature Space" },
    { title: "Benchmark 1: Zero-Shot OOD Validation (GSE31210)", cat: "OOD Results" },
    { title: "Benchmark 2: Monotonic Horizon AUCs & KM Strata", cat: "Survival Curves" },
    { title: "Benchmark 3: Biophysical Ablation & Adversarial Stress", cat: "Robustness" },
    { title: "MSigDB Hallmark Pathways & Key Biomarkers", cat: "Biology" },
    { title: "Clinical Risk Stratification & Software Ecosystem", cat: "Translation & Software" }
  ];

  const slideNotes = [
    "Introduce Chara as a zero-retraining survival generalization platform connecting atomistic biophysics with transcriptomic clinical cohorts.",
    "Explain why RNA-seq models fail when deployed onto historic microarray biobanks. High-density cross-platform benchmark breakdown.",
    "Walk through the 4-step pipeline: MD simulations -> dynamic contact variance -> continuous heat diffusion -> penalized Cox survival.",
    "Detail the 4 hallmark oncoproteins simulated: KRAS G12D, c-MYC, PTPN11, and mutant p53 in MARTINI 3 force fields.",
    "Highlight the Switch I/II loop cryptic pocket opening in laser-cyan. Show how dynamic flexibility defines signaling edge capacity.",
    "Show quantitative triplicate RMSD and radius of gyration convergence proving structural stability across all replicas.",
    "Explain how residue contact variance is transformed into exponential edge weights in the physical STRING network.",
    "Interactive network simulator: demonstrate how heat diffusion attenuates high-frequency technical noise while preserving pathway co-expression.",
    "Present Dirichlet energy proof on 503 TCGA patients: topological roughness drops with p = 2.41e-19.",
    "Describe the 4,337 conserved gene manifold and the 58 non-zero active biomarkers isolated by ElasticNet.",
    "Zero-shot OOD results: Chara achieves C = 0.7311 on held-out GSE31210 microarray cohort vs DeepSurv at 0.5537 and RSF at 0.4041.",
    "Show monotonic time-dependent AUC progression from 1-year (0.746) to 5-year (0.819) and Kaplan-Meier log-rank p < 10^-6.",
    "Ablation study: MD physics contributes +0.1191 C-index. Adversarial stress test proves robustness up to 50% Gaussian corruption.",
    "Biological validation: enriched MSigDB hallmarks (KRAS, EMT, Hypoxia) and key prognostic drivers (CCL20, DKK1, MS4A1, FAIM2).",
    "Clinical 4-tier stratification, independent hazard ratio HR = 3.81 (p = 4.48e-33), pip install chara-survival, and web platform."
  ];

  return (
    <div className={`h-screen w-screen flex flex-col justify-between overflow-hidden select-none font-sans ${isDark ? 'dark bg-obsidian text-slate-100' : 'bg-brand-bg text-slate-800'}`}>
      
      {/* ========================================================================= */}
      {/* TOP NAVBAR */}
      {/* ========================================================================= */}
      <header className="fixed top-0 left-0 right-0 z-50 px-6 py-3 flex items-center justify-between bg-white/90 dark:bg-obsidian/90 backdrop-blur-xl border-b border-slate-200 dark:border-white/10 shadow-xs transition-colors">
        <div className="flex items-center gap-3">
          <img src="/iit-mandi-logo.png" alt="IIT Mandi" className="w-8 h-8 object-contain rounded-lg p-0.5 border border-slate-200 dark:border-white/10 bg-white" />
          <div>
            <span className="text-[11px] font-bold text-slate-900 dark:text-white tracking-wider uppercase block">
              IIT Mandi · CPG Lab
            </span>
            <span className="text-xs font-semibold text-brand-blue dark:text-laser-cyan">
              Chara: Next.js Presentation Deck
            </span>
          </div>
        </div>

        {/* Center Topic Indicator */}
        <div className="hidden md:flex items-center gap-2 px-3.5 py-1 rounded-full bg-slate-100 dark:bg-white/5 border border-slate-200 dark:border-white/10 text-xs font-medium text-slate-700 dark:text-slate-300">
          <span className="w-2 h-2 rounded-full bg-brand-emerald animate-pulse"></span>
          <span>{slideTitles[currentSlide].cat}: {slideTitles[currentSlide].title}</span>
        </div>

        {/* Right Action Tools */}
        <div className="flex items-center gap-2">
          {/* Timer */}
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 dark:bg-white/5 text-xs font-mono text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-white/10">
            <Activity className="w-3.5 h-3.5 text-brand-blue dark:text-laser-cyan" />
            <span>{formatTime(elapsedTime)}</span>
          </div>

          {/* Slide Navigator */}
          <button
            onClick={prevSlide}
            disabled={currentSlide === 0}
            className="w-7 h-7 rounded-lg bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 disabled:opacity-30 flex items-center justify-center border border-slate-200 dark:border-white/10 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          
          <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-md bg-white dark:bg-white/5 text-slate-900 dark:text-laser-cyan border border-slate-200 dark:border-white/10 min-w-[60px] text-center shadow-xs">
            {String(currentSlide + 1).padStart(2, '0')} / {String(totalSlides).padStart(2, '0')}
          </span>

          <button
            onClick={nextSlide}
            disabled={currentSlide === totalSlides - 1}
            className="w-7 h-7 rounded-lg bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 disabled:opacity-30 flex items-center justify-center border border-slate-200 dark:border-white/10 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>

          {/* Thumbnails Grid Toggle */}
          <button
            onClick={() => setShowThumbnails(!showThumbnails)}
            title="Slide Grid (T)"
            className="w-7 h-7 rounded-lg bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 flex items-center justify-center border border-slate-200 dark:border-white/10 transition-colors"
          >
            <Grid className="w-3.5 h-3.5" />
          </button>

          {/* Presenter Notes Toggle */}
          <button
            onClick={() => setShowNotes(!showNotes)}
            title="Presenter Notes (N)"
            className={`w-7 h-7 rounded-lg flex items-center justify-center border transition-colors ${
              showNotes
                ? 'bg-brand-blue text-white border-brand-blue'
                : 'bg-slate-100 dark:bg-white/5 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-white/10 hover:bg-slate-200 dark:hover:bg-white/10'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
          </button>

          {/* Theme Toggle */}
          <button
            onClick={() => setIsDark(!isDark)}
            title="Toggle Light/Dark Theme (D)"
            className="w-7 h-7 rounded-lg bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 flex items-center justify-center border border-slate-200 dark:border-white/10 transition-colors"
          >
            {isDark ? <Sun className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5" />}
          </button>

          {/* Fullscreen Toggle */}
          <button
            onClick={toggleFullscreen}
            title="Fullscreen (F)"
            className="w-7 h-7 rounded-lg bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 flex items-center justify-center border border-slate-200 dark:border-white/10 transition-colors"
          >
            {isFullscreen ? <Minimize className="w-3.5 h-3.5" /> : <Maximize className="w-3.5 h-3.5" />}
          </button>
        </div>
      </header>

      {/* TOP PROGRESS BAR */}
      <div className="fixed top-[53px] left-0 right-0 z-50 h-[3px] bg-slate-200 dark:bg-white/5">
        <motion.div
          className="h-full bg-gradient-to-r from-brand-blue via-brand-emerald to-laser-cyan"
          animate={{ width: `${((currentSlide + 1) / totalSlides) * 100}%` }}
          transition={{ duration: 0.2, ease: "easeInOut" }}
        />
      </div>

      {/* ========================================================================= */}
      {/* MAIN SLIDE CONTAINER WITH FRAMER MOTION TRANSITIONS */}
      {/* ========================================================================= */}
      <main className="relative flex-1 w-full h-full pt-14 pb-12 flex items-center justify-center overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentSlide}
            initial={{ opacity: 0, scale: 0.98, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98, y: -8 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="w-full h-full max-w-6xl mx-auto px-6 py-4 flex flex-col justify-center"
          >
            {renderSlideContent(currentSlide, setCopiedPip, copiedPip)}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* ========================================================================= */}
      {/* FOOTER */}
      {/* ========================================================================= */}
      <footer className="fixed bottom-0 left-0 right-0 z-50 px-6 py-2.5 flex items-center justify-between bg-white/90 dark:bg-obsidian/90 backdrop-blur-xl border-t border-slate-200 dark:border-white/10 text-xs font-mono text-slate-500 dark:text-slate-400 shadow-xs">
        <div className="flex items-center gap-4">
          <span><kbd className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-white/10 text-slate-800 dark:text-slate-200 border border-slate-300 dark:border-white/10 font-bold">Space</kbd> / <kbd className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-white/10 text-slate-800 dark:text-slate-200 border border-slate-300 dark:border-white/10 font-bold">→</kbd> Next</span>
          <span><kbd className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-white/10 text-slate-800 dark:text-slate-200 border border-slate-300 dark:border-white/10 font-bold">←</kbd> Prev</span>
          <span><kbd className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-white/10 text-slate-800 dark:text-slate-200 border border-slate-300 dark:border-white/10 font-bold">T</kbd> Grid</span>
          <span><kbd className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-white/10 text-slate-800 dark:text-slate-200 border border-slate-300 dark:border-white/10 font-bold">N</kbd> Notes</span>
        </div>
        <div>CPG Lab · IIT Mandi © 2026</div>
      </footer>

      {/* ========================================================================= */}
      {/* THUMBNAIL DRAWER OVERLAY */}
      {/* ========================================================================= */}
      <AnimatePresence>
        {showThumbnails && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-md flex items-center justify-center p-8"
            onClick={() => setShowThumbnails(false)}
          >
            <motion.div
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.95 }}
              className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-white/10 rounded-2xl p-6 max-w-5xl w-full max-h-[85vh] overflow-y-auto shadow-2xl space-y-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-slate-200 dark:border-white/10 pb-3">
                <div className="flex items-center gap-2 font-bold text-slate-900 dark:text-white">
                  <Grid className="w-5 h-5 text-brand-blue dark:text-laser-cyan" />
                  <span>Slide Navigator ({totalSlides} Slides)</span>
                </div>
                <button
                  onClick={() => setShowThumbnails(false)}
                  className="text-xs font-semibold px-3 py-1 rounded-lg bg-slate-100 dark:bg-white/10 hover:bg-slate-200 dark:hover:bg-white/20 text-slate-700 dark:text-slate-300"
                >
                  Close (Esc)
                </button>
              </div>

              <div className="grid grid-cols-3 sm:grid-cols-5 gap-3.5">
                {slideTitles.map((s, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setCurrentSlide(idx);
                      setShowThumbnails(false);
                    }}
                    className={`p-3 rounded-xl border text-left transition-all relative flex flex-col justify-between h-24 ${
                      currentSlide === idx
                        ? 'border-brand-blue dark:border-laser-cyan bg-blue-50/50 dark:bg-cyan-950/40 ring-2 ring-brand-blue/30 dark:ring-laser-cyan/30'
                        : 'border-slate-200 dark:border-white/10 hover:border-slate-300 dark:hover:border-white/20 bg-slate-50 dark:bg-white/5'
                    }`}
                  >
                    <span className="text-[10px] font-mono font-bold text-brand-blue dark:text-laser-cyan">
                      Slide {String(idx + 1).padStart(2, '0')}
                    </span>
                    <span className="text-xs font-semibold text-slate-900 dark:text-white line-clamp-2 leading-snug">
                      {s.title}
                    </span>
                    <span className="text-[10px] text-slate-500 dark:text-slate-400 truncate">
                      {s.cat}
                    </span>
                  </button>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ========================================================================= */}
      {/* PRESENTER NOTES DRAWER */}
      {/* ========================================================================= */}
      <AnimatePresence>
        {showNotes && (
          <motion.div
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 100 }}
            className="fixed bottom-12 right-6 z-50 w-96 bg-white dark:bg-zinc-900 border border-slate-200 dark:border-white/10 rounded-2xl p-4 shadow-2xl space-y-2"
          >
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-white/10 pb-2">
              <div className="flex items-center gap-1.5 font-bold text-xs text-brand-blue dark:text-laser-cyan">
                <BookOpen className="w-4 h-4" />
                <span>Presenter Notes · Slide {currentSlide + 1}</span>
              </div>
              <button
                onClick={() => setShowNotes(false)}
                className="text-[11px] font-mono text-slate-400 hover:text-slate-600 dark:hover:text-white"
              >
                ✕
              </button>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-sans">
              {slideNotes[currentSlide]}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}

// =========================================================================
// SLIDE RENDERER FUNCTION
// =========================================================================
function renderSlideContent(index: number, setCopiedPip: (v: boolean) => void, copiedPip: boolean) {
  switch (index) {
    // -------------------------------------------------------------
    // SLIDE 01: TITLE
    // -------------------------------------------------------------
    case 0:
      return (
        <div className="space-y-6 text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-blue-50 dark:bg-cyan-950/60 border border-blue-200 dark:border-cyan-500/40 text-brand-blue dark:text-laser-cyan text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5 animate-spin" />
            <span>Computational &amp; Physical Genomics Lab · IIT Mandi</span>
          </div>

          <h1 className="text-4xl sm:text-5xl font-extrabold text-brand-navy dark:text-white tracking-tight leading-tight">
            Chara: Molecular-Dynamics-Guided Survival Generalization
          </h1>

          <p className="text-base sm:text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto font-light leading-relaxed">
            Zero-Retraining Cross-Platform Cancer Prognosis via Biophysically Weighted Spectral Diffusion Networks
          </p>

          <div className="grid sm:grid-cols-3 gap-3.5 max-w-3xl mx-auto text-left pt-2">
            <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-blue">
              <span className="text-xs font-bold text-brand-blue dark:text-laser-cyan uppercase block mb-1">01. Physics Prior</span>
              <p className="text-xs text-slate-600 dark:text-slate-300">Encodes atomistic residue fluctuations from MD simulations into protein graphs.</p>
            </div>
            <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-emerald">
              <span className="text-xs font-bold text-brand-emerald uppercase block mb-1">02. Zero-Shot OOD</span>
              <p className="text-xs text-slate-600 dark:text-slate-300">Evaluates across external cohorts (GSE31210) without retraining (C = 0.7311).</p>
            </div>
            <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-amber">
              <span className="text-xs font-bold text-brand-amber uppercase block mb-1">03. Clinical Utility</span>
              <p className="text-xs text-slate-600 dark:text-slate-300">Independent prognostic hazard (HR = 3.81, p &lt; 10⁻³²) adjusting for staging.</p>
            </div>
          </div>

          <div className="grid sm:grid-cols-2 gap-4 max-w-lg mx-auto pt-2 text-left">
            <div className="glass-card p-3 rounded-xl flex items-center gap-3">
              <img src="/sharon-melhi.png" alt="Sharon Melhi" className="w-10 h-10 rounded-lg object-cover border border-slate-200 dark:border-white/10" />
              <div>
                <p className="text-xs font-bold text-slate-900 dark:text-white">Sharon Melhi Nadar</p>
                <p className="text-[11px] text-brand-blue dark:text-laser-cyan font-medium">Lead Author &amp; Developer</p>
              </div>
            </div>
            <div className="glass-card p-3 rounded-xl flex items-center gap-3">
              <img src="/dr-kharerin-hungyo.png" alt="Dr. Kharerin Hungyo" className="w-10 h-10 rounded-lg object-cover border border-slate-200 dark:border-white/10" />
              <div>
                <p className="text-xs font-bold text-slate-900 dark:text-white">Dr. Kharerin Hungyo</p>
                <p className="text-[11px] text-brand-emerald font-medium">Principal Investigator, IIT Mandi</p>
              </div>
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 02: CRISIS & FIGURE 1
    // -------------------------------------------------------------
    case 1:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-crimson uppercase tracking-wider block">The Clinical Problem</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">Why 99% of Published Oncology AI Models Fail in Clinic</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-6 items-center">
            <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-crimson">
                <span className="text-xs font-bold text-brand-crimson block mb-1">Catastrophic Batch Effect Collapse</span>
                <p>
                  Models trained on <strong>Illumina RNA-seq (TCGA)</strong> memorize platform-specific negative binomial distributions. When deployed onto historic <strong>Affymetrix Microarray biobanks (GSE31210)</strong>, their predictive accuracy completely collapses.
                </p>
              </div>
              <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-emerald">
                <span className="text-xs font-bold text-brand-emerald block mb-1">The Biophysical Invariance Principle</span>
                <p>
                  Sequencing chemistries shift every decade, but <strong>protein allostery and physical binding kinetics never change</strong>. Chara uses physical simulations as an invariant inductive prior.
                </p>
              </div>
            </div>

            <div className="glass-card p-4 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg">
              <span className="text-xs font-bold text-slate-800 dark:text-white block mb-2">Figure 1: Cross-Platform Performance Breakdown</span>
              <img src="/Fig1_OOD_Performance.png" alt="Figure 1" className="w-full h-auto rounded-lg object-contain" />
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 03: 4-STEP PIPELINE
    // -------------------------------------------------------------
    case 2:
      return (
        <div className="space-y-5 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-blue dark:text-laser-cyan uppercase tracking-wider block">Methodology Architecture</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">The 4-Step Chara Computational Architecture</h2>
          </div>

          <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            {[
              { num: "01", title: "Molecular Dynamics", desc: "Simulate 4 oncogenic proteins across triplicate replicas in MARTINI 3 coarse-grained physics.", col: "text-brand-blue border-t-brand-blue" },
              { num: "02", title: "Contact Variance", desc: "Extract dynamic residue variances (sigma^2) and map them directly into protein interaction edge weights.", col: "text-brand-emerald border-t-brand-emerald" },
              { num: "03", title: "Spectral Diffusion", desc: "Apply continuous graph heat diffusion to filter high-frequency technical platform noise and smooth expression.", col: "text-brand-amber border-t-brand-amber" },
              { num: "04", title: "Zero-Shot Survival", desc: "Project patient expression onto the 4,337-gene conserved manifold for 5-year survival estimation and risk scoring.", col: "text-indigo-600 border-t-indigo-600" },
            ].map((st, i) => (
              <motion.div
                key={i}
                whileHover={{ y: -4 }}
                className={`glass-card p-5 rounded-2xl border-t-4 ${st.col} space-y-2`}
              >
                <div className="text-2xl font-bold font-mono">{st.num}</div>
                <h3 className="font-bold text-slate-900 dark:text-white text-sm">{st.title}</h3>
                <p className="text-slate-600 dark:text-slate-300">{st.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 04: MD SETUP
    // -------------------------------------------------------------
    case 3:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-blue dark:text-laser-cyan uppercase tracking-wider block">Step 1 · Biophysical Setup</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">MARTINI 3 Coarse-Grained Molecular Dynamics</h2>
          </div>

          <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-3.5 text-xs text-center">
            {[
              { code: "KRAS", name: "KRAS G12D (4OBE)", desc: "Captures Switch I/II loop opening and cryptic pocket exploration.", bg: "bg-blue-50 text-brand-blue" },
              { code: "MYC", name: "c-MYC / MAX (1NKP)", desc: "Models bHLH-LZ heterodimer interface rigidity & flanking loop motion.", bg: "bg-indigo-50 text-indigo-600" },
              { code: "SHP2", name: "PTPN11 (4DGP)", desc: "Simulates N-SH2 domain autoinhibitory cleft dynamics.", bg: "bg-emerald-50 text-brand-emerald" },
              { code: "p53", name: "Mutant TP53 (2J1X)", desc: "Captures DNA core domain destabilization & unfolding kinetics.", bg: "bg-red-50 text-brand-crimson" },
            ].map((tgt, i) => (
              <div key={i} className="glass-card p-4 rounded-2xl space-y-2">
                <span className={`w-10 h-10 rounded-full ${tgt.bg} flex items-center justify-center font-bold mx-auto text-xs`}>
                  {tgt.code}
                </span>
                <strong className="text-slate-900 dark:text-white block text-xs">{tgt.name}</strong>
                <p className="text-slate-500 dark:text-slate-400 text-[11px]">{tgt.desc}</p>
              </div>
            ))}
          </div>

          <div className="glass-card p-3 rounded-xl text-xs text-slate-700 dark:text-slate-300 grid sm:grid-cols-4 gap-2 text-center font-mono">
            <div>• Force Field: MARTINI 3</div>
            <div>• Ensemble: NPT (300K, 1 bar)</div>
            <div>• Timestep: 1.0 fs</div>
            <div>• Solvent: Explicit Water + 0.15M NaCl</div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 05: HD VIDEO CONTAINER
    // -------------------------------------------------------------
    case 4:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs font-mono text-brand-blue dark:text-laser-cyan uppercase tracking-wider block">Slide 05 · Trajectory Video</span>
              <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">KRAS G12D Conformational Switch &amp; Pocket Opening</h2>
            </div>
            <span className="px-3 py-1 rounded-full bg-cyan-100 dark:bg-cyan-950/60 border border-cyan-300 dark:border-cyan-500/40 text-brand-blue dark:text-laser-cyan text-xs font-mono font-semibold flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-brand-blue dark:bg-laser-cyan animate-ping"></span>
              <span>Residues 60–75 Cryptic Opening</span>
            </span>
          </div>

          <div className="glass-card p-3 rounded-2xl border border-slate-200 dark:border-white/10 relative overflow-hidden flex flex-col items-center">
            <video
              src="/md_simulation.mp4"
              controls
              autoPlay
              loop
              muted
              playsInline
              className="w-full max-h-[440px] object-contain rounded-xl shadow-2xl bg-black"
            />
            <div className="absolute bottom-6 left-8 bg-white/90 dark:bg-obsidian/90 backdrop-blur-md border border-slate-200 dark:border-white/10 px-4 py-2.5 rounded-xl shadow-xl flex items-center gap-3">
              <span className="w-2.5 h-2.5 rounded-full bg-brand-blue dark:bg-laser-cyan shadow-md"></span>
              <div>
                <p className="text-xs font-bold text-slate-900 dark:text-white">Switch I/II Dynamic Cryptic Opening</p>
                <p className="text-[10px] font-mono text-slate-500 dark:text-slate-400">Allosteric edge variance σ²(G12D, Q61H) = 0.841</p>
              </div>
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 06: RMSD & RG STABILITY CHARTS
    // -------------------------------------------------------------
    case 5:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-blue dark:text-laser-cyan uppercase tracking-wider block">Step 1 · Trajectory Verification</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">Triplicate RMSD &amp; Radius of Gyration (Rg) Stability</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-5">
            <div className="glass-card p-4 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg space-y-2">
              <div className="flex justify-between items-center text-xs font-bold">
                <span>RMSD Trajectory Convergence</span>
                <span className="text-brand-emerald font-mono">SD &lt; 0.04 nm</span>
              </div>
              <img src="/RMSD_Replicas_300dpi.png" alt="RMSD Plot" className="w-full h-auto rounded-lg object-contain" />
            </div>

            <div className="glass-card p-4 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg space-y-2">
              <div className="flex justify-between items-center text-xs font-bold">
                <span>Radius of Gyration (Rg) Compaction</span>
                <span className="text-brand-blue dark:text-laser-cyan font-mono">1.45–1.55 nm</span>
              </div>
              <img src="/Rg_Replicas_300dpi.png" alt="Rg Plot" className="w-full h-auto rounded-lg object-contain" />
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 07: CONTACT VARIANCE MATRIX
    // -------------------------------------------------------------
    case 6:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-blue dark:text-laser-cyan uppercase tracking-wider block">Step 2 · Graph Construction</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">Transforming Residue Fluctuations into Graph Weights</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-6 items-center">
            <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              <p>Static protein interaction databases (STRING) treat all edges uniformly. Chara introduces an <strong>allosteric weighting rule</strong>:</p>
              <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-blue">
                <strong className="text-slate-900 dark:text-white block text-xs">High Conformational Variance → Amplified Edge Weight</strong>
                <p>When residues undergo dynamic switching in MD, their inter-protein signaling capacity is exponentially boosted.</p>
              </div>
              <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-emerald">
                <strong className="text-slate-900 dark:text-white block text-xs">Rigid Hydrophobic Cores → Baseline Topology</strong>
                <p>Rigid structural scaffolds maintain standard baseline STRING confidence scores.</p>
              </div>
            </div>

            <div className="glass-card p-5 rounded-2xl text-center space-y-2">
              <span className="text-xs font-bold text-slate-800 dark:text-white">Dynamic Contact Variance Matrix (σ²_ij)</span>
              <div className="grid grid-cols-5 gap-1.5 max-w-[220px] mx-auto p-2 bg-slate-100 dark:bg-white/5 rounded-xl">
                {[
                  0.9, 0.6, 0.3, 0.1, 0.7,
                  0.6, 0.9, 0.7, 0.4, 0.3,
                  0.3, 0.7, 0.9, 0.6, 0.2,
                  0.1, 0.4, 0.6, 0.9, 0.6,
                  0.7, 0.3, 0.2, 0.6, 0.9
                ].map((val, i) => (
                  <div
                    key={i}
                    title={`Contact variance: ${val}`}
                    className="h-8 rounded flex items-center justify-center text-[10px] font-mono font-bold text-white transition-transform hover:scale-110 cursor-pointer"
                    style={{ backgroundColor: `rgba(37, 99, 235, ${val})` }}
                  >
                    {val}
                  </div>
                ))}
              </div>
              <span className="text-[11px] text-slate-500 dark:text-slate-400 font-mono">Darker Blue = Higher Dynamic Coupling</span>
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 08: INTERACTIVE HEAT DIFFUSION
    // -------------------------------------------------------------
    case 7:
      return <InteractiveHeatDiffusionSlide />;

    // -------------------------------------------------------------
    // SLIDE 09: DIRICHLET PROOF & FIGURE 5
    // -------------------------------------------------------------
    case 8:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-emerald uppercase tracking-wider block">Validation Metric</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">Empirical Proof: Dirichlet Energy Reduction in TCGA (p &lt; 10⁻¹⁹)</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-6 items-center">
            <div className="glass-card p-4 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg">
              <span className="text-xs font-bold text-slate-800 dark:text-white block mb-2">Figure 5: Dirichlet Topological Smoothness</span>
              <img src="/Fig5_Dirichlet_Energy.png" alt="Figure 5" className="w-full h-auto rounded-lg object-contain" />
            </div>

            <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-emerald">
                <span className="text-xs font-bold text-slate-900 dark:text-white block mb-1">What Dirichlet Energy Proves:</span>
                <p>Lower Dirichlet energy indicates that expression values across interacting protein partners are harmonized without artificial smoothing.</p>
              </div>
              <p className="text-slate-500 dark:text-slate-400 font-mono text-[11px]">
                Paired Wilcoxon signed-rank test confirmed that Chara significantly lowers topological roughness across all 503 clinical patients (p = 2.41 × 10⁻¹⁹).
              </p>
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 10: 4,337-GENE SIGNATURE
    // -------------------------------------------------------------
    case 9:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-blue dark:text-laser-cyan uppercase tracking-wider block">Step 4 · Feature Space</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">The 4,337-Gene Conserved Signature &amp; 58 Biomarkers</h2>
          </div>

          <div className="grid sm:grid-cols-3 gap-4 text-center">
            <div className="glass-card p-5 rounded-2xl space-y-1">
              <div className="text-3xl font-bold text-slate-400 font-mono">19,260</div>
              <span className="text-xs font-semibold text-slate-800 dark:text-white block">TCGA RNA-seq Transcriptome</span>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">Initial unconstrained protein-coding transcripts.</p>
            </div>
            <div className="glass-card p-5 rounded-2xl space-y-1">
              <div className="text-3xl font-bold text-slate-400 font-mono">12,488</div>
              <span className="text-xs font-semibold text-slate-800 dark:text-white block">STRING Physical Graph</span>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">High-confidence curated interaction network.</p>
            </div>
            <div className="glass-card p-5 rounded-2xl space-y-1 border-2 border-brand-blue dark:border-laser-cyan bg-blue-50/20 dark:bg-cyan-950/30">
              <div className="text-3xl font-bold text-brand-blue dark:text-laser-cyan font-mono">4,337</div>
              <span className="text-xs font-bold text-brand-blue dark:text-laser-cyan block">Conserved Manifold</span>
              <p className="text-[11px] text-slate-600 dark:text-slate-300">Exact overlap across RNA-seq and Affymetrix Microarrays.</p>
            </div>
          </div>

          <div className="glass-card p-4 rounded-xl text-xs text-slate-700 dark:text-slate-300 flex items-center justify-between">
            <div>
              <strong className="text-slate-900 dark:text-white block text-xs">Penalized ElasticNet Selection:</strong>
              <span className="text-slate-500 dark:text-slate-400">Isolates exactly <strong>58 non-zero active biomarkers</strong> from the diffused 4,337-gene manifold.</span>
            </div>
            <span className="px-3.5 py-1.5 rounded-full bg-brand-emerald text-white font-mono text-xs font-bold">58 Active Genes</span>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 11: BENCHMARK 1 (OOD VALIDATION)
    // -------------------------------------------------------------
    case 10:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-emerald uppercase tracking-wider block">Empirical Benchmark 1</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">Zero-Shot Out-of-Distribution Validation (GSE31210, n=226)</h2>
          </div>

          <div className="grid md:grid-cols-12 gap-6 items-center">
            <div className="md:col-span-7 glass-card p-5 rounded-2xl space-y-3">
              <span className="text-xs font-bold text-slate-800 dark:text-white block">Concordance Index (C-Index) on Held-Out Cohort:</span>
              <div className="space-y-2.5 font-mono text-xs">
                {[
                  { name: "Chara Framework (Ours)", c: "0.7311", w: "92%", col: "bg-brand-emerald text-white font-bold" },
                  { name: "DeepSurv (Neural Net)", c: "0.5537", w: "69%", col: "bg-amber-400 text-slate-900 font-semibold" },
                  { name: "Standard ElasticNet", c: "0.5248", w: "65%", col: "bg-slate-300 text-slate-800 font-medium" },
                  { name: "Clinical Baseline (Age, Stage)", c: "0.5000", w: "62%", col: "bg-slate-200 text-slate-700" },
                  { name: "Random Survival Forest (RSF)", c: "0.4041", w: "50%", col: "bg-red-400 text-white font-semibold" },
                ].map((row, i) => (
                  <div key={i} className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-700 dark:text-slate-300">
                      <span>{row.name}</span>
                      <span className="font-bold">{row.c}</span>
                    </div>
                    <div className="w-full bg-slate-100 dark:bg-white/5 rounded-full h-3 overflow-hidden">
                      <div className={`h-3 rounded-full ${row.col}`} style={{ width: row.w }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="md:col-span-5 space-y-3 text-xs text-slate-600 dark:text-slate-300">
              <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-emerald">
                <span className="text-xs font-bold text-slate-900 dark:text-white block mb-1">+0.267 Gain Over Deep Learning</span>
                <p>Deep neural networks collapse due to platform covariate shifts. Chara achieves gold-standard discrimination (C = 0.7311) with zero fine-tuning.</p>
              </div>
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 12: BENCHMARK 2 (TIME HORIZONS & FIGURE 2)
    // -------------------------------------------------------------
    case 11:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-emerald uppercase tracking-wider block">Empirical Benchmark 2</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">Monotonic Time-Horizon AUCs &amp; Kaplan-Meier Separation</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-5 items-center">
            <div className="glass-card p-4 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg">
              <span className="text-xs font-bold text-slate-800 dark:text-white block mb-2">Figure 2: Time Horizons &amp; KM Survival Curves</span>
              <img src="/Fig2_KM_Survival.png" alt="Figure 2" className="w-full h-auto rounded-lg object-contain" />
            </div>

            <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300">
              <div className="glass-card p-4 rounded-xl space-y-2 border-l-4 border-l-brand-blue">
                <strong className="text-xs font-bold text-slate-900 dark:text-white block">Monotonic AUC Progression:</strong>
                <div className="grid grid-cols-3 gap-2 text-center font-mono">
                  <div className="p-2 rounded bg-slate-50 dark:bg-white/5 border border-slate-200 dark:border-white/10">
                    <span className="text-[10px] text-slate-500 block">1-Year</span>
                    <strong className="text-brand-blue text-sm">0.7463</strong>
                  </div>
                  <div className="p-2 rounded bg-slate-50 dark:bg-white/5 border border-slate-200 dark:border-white/10">
                    <span className="text-[10px] text-slate-500 block">3-Year</span>
                    <strong className="text-brand-blue text-sm">0.7826</strong>
                  </div>
                  <div className="p-2 rounded bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-500/40">
                    <span className="text-[10px] text-emerald-600 block">5-Year</span>
                    <strong className="text-brand-emerald text-sm">0.8195</strong>
                  </div>
                </div>
              </div>

              <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-crimson">
                <strong className="text-xs font-bold text-brand-crimson block mb-1">Kaplan-Meier Separation:</strong>
                <p>Log-rank test p &lt; 10⁻⁶ confirms distinct survival divergence between Low (61.4% 5-yr) and High (&lt;18.5% 5-yr) risk patients.</p>
              </div>
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 13: BENCHMARK 3 (ABLATION & ADVERSARIAL)
    // -------------------------------------------------------------
    case 12:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-amber uppercase tracking-wider block">Empirical Benchmark 3</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">Biophysical Ablation &amp; Adversarial Noise Robustness</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-5">
            <div className="glass-card p-4 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg space-y-2">
              <div className="flex justify-between items-center text-xs font-bold">
                <span>Figure 3: Physics Ablation Impact</span>
                <span className="text-brand-emerald font-mono">+0.1191 Gain</span>
              </div>
              <img src="/Fig3_Ablation_Impact.png" alt="Figure 3" className="w-full h-auto rounded-lg object-contain" />
            </div>

            <div className="glass-card p-4 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg space-y-2">
              <div className="flex justify-between items-center text-xs font-bold">
                <span>Figure 4: 50% Gaussian Stress Test</span>
                <span className="text-brand-blue dark:text-laser-cyan font-mono">C &gt; 0.68</span>
              </div>
              <img src="/Fig4_Adversarial_Decay.png" alt="Figure 4" className="w-full h-auto rounded-lg object-contain" />
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 14: GSEA & BIOMARKERS (FIGURE 6)
    // -------------------------------------------------------------
    case 13:
      return (
        <div className="space-y-4 max-w-5xl mx-auto w-full">
          <div>
            <span className="text-xs font-bold text-brand-emerald uppercase tracking-wider block">Biological Interpretability</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">MSigDB Hallmark Pathways &amp; Key Biomarkers</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-5 items-center">
            <div className="glass-card p-4 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg">
              <span className="text-xs font-bold text-slate-800 dark:text-white block mb-2">Figure 6: MSigDB Hallmark Enrichment</span>
              <img src="/Fig6_GSEA_Enrichment.png" alt="Figure 6" className="w-full h-auto rounded-lg object-contain" />
            </div>

            <div className="space-y-2 text-xs">
              <span className="text-xs font-bold text-slate-800 dark:text-white block">Top Prognostic Effect Sizes (β):</span>
              <div className="space-y-1.5">
                <div className="p-2.5 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-500/30 flex justify-between items-center">
                  <span><strong>CCL20</strong> (Treg recruitment)</span>
                  <span className="font-mono text-brand-crimson font-bold">β = +0.0642</span>
                </div>
                <div className="p-2.5 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-500/30 flex justify-between items-center">
                  <span><strong>DKK1</strong> (Wnt inhibitor metastasis)</span>
                  <span className="font-mono text-brand-crimson font-bold">β = +0.0610</span>
                </div>
                <div className="p-2.5 rounded-xl bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-500/30 flex justify-between items-center">
                  <span><strong>MS4A1 / CD20</strong> (B-cell immunity)</span>
                  <span className="font-mono text-brand-emerald font-bold">β = -0.0708</span>
                </div>
                <div className="p-2.5 rounded-xl bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-500/30 flex justify-between items-center">
                  <span><strong>FAIM2</strong> (Cellular homeostasis)</span>
                  <span className="font-mono text-brand-emerald font-bold">β = -0.0524</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      );

    // -------------------------------------------------------------
    // SLIDE 15: STRATIFICATION & ECOSYSTEM
    // -------------------------------------------------------------
    case 14:
      return (
        <div className="space-y-5 max-w-4xl mx-auto w-full text-center">
          <img src="/iit-mandi-logo.png" alt="IIT Mandi" className="w-14 h-14 object-contain rounded-2xl mx-auto p-1 border border-slate-200 dark:border-white/10 shadow-sm bg-white" />

          <h2 className="text-3xl font-extrabold text-brand-navy dark:text-white">
            Computational &amp; Physical Genomics Laboratory
          </h2>
          <p className="text-xs font-mono text-brand-blue dark:text-laser-cyan font-semibold">Indian Institute of Technology Mandi · Himachal Pradesh, India</p>

          <div className="grid sm:grid-cols-2 gap-4 max-w-xl mx-auto text-left text-xs font-mono">
            <div className="glass-card p-4 rounded-xl space-y-1.5">
              <span className="text-slate-500 dark:text-slate-400 block text-[11px]">Official PyPI Library (v0.2.5):</span>
              <div className="flex items-center justify-between bg-slate-100 dark:bg-black/40 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-white/10">
                <span className="text-brand-emerald font-bold">pip install chara-survival</span>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText("pip install chara-survival");
                    setCopiedPip(true);
                    setTimeout(() => setCopiedPip(false), 2000);
                  }}
                  className="text-slate-500 hover:text-slate-900 dark:hover:text-white"
                >
                  {copiedPip ? <CheckCircle2 className="w-4 h-4 text-brand-emerald" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
              <span className="text-slate-400 text-[10px]">4,600+ community downloads</span>
            </div>

            <div className="glass-card p-4 rounded-xl space-y-1.5">
              <span className="text-slate-500 dark:text-slate-400 block text-[11px]">Live Web Platform:</span>
              <div className="flex items-center justify-between bg-slate-100 dark:bg-black/40 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-white/10">
                <span className="text-brand-blue dark:text-laser-cyan font-bold truncate">chara-frontend.vercel.app</span>
                <a href="https://chara-frontend.vercel.app" target="_blank" rel="noreferrer" className="text-slate-500 hover:text-slate-900 dark:hover:text-white">
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
              <span className="text-slate-400 text-[10px]">Zero-install WebAssembly client</span>
            </div>
          </div>

          <div className="flex items-center justify-center gap-4 pt-2">
            <a
              href="https://chara-frontend.vercel.app"
              target="_blank"
              rel="noreferrer"
              className="px-6 py-2.5 rounded-xl bg-brand-blue text-white font-semibold text-xs shadow-sm hover:bg-blue-700 transition-colors flex items-center gap-2"
            >
              <span>Launch Live Platform</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <a
              href="https://github.com/Sharon-codes/Chara"
              target="_blank"
              rel="noreferrer"
              className="px-6 py-2.5 rounded-xl bg-white dark:bg-white/10 text-slate-800 dark:text-white border border-slate-300 dark:border-white/10 font-semibold text-xs shadow-xs hover:bg-slate-50 dark:hover:bg-white/20 transition-colors"
            >
              GitHub Repository
            </a>
          </div>
        </div>
      );

    default:
      return null;
  }
}

// =========================================================================
// INTERACTIVE HEAT DIFFUSION SIMULATOR COMPONENT
// =========================================================================
function InteractiveHeatDiffusionSlide() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = (canvas.width = canvas.parentElement?.clientWidth || 550);
    let height = (canvas.height = 300);

    const nodes: { x: number; y: number; vx: number; vy: number; heat: number; radius: number }[] = [];
    for (let i = 0; i < 24; i++) {
      nodes.push({
        x: Math.random() * (width - 60) + 30,
        y: Math.random() * (height - 60) + 30,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        heat: Math.random() < 0.2 ? 1.0 : 0.0,
        radius: 4.5 + Math.random() * 3,
      });
    }

    const edges: { from: number; to: number; weight: number }[] = [];
    for (let i = 0; i < 24; i++) {
      for (let j = i + 1; j < 24; j++) {
        const dist = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y);
        if (dist < 100) edges.push({ from: i, to: j, weight: 1.0 - dist / 100 });
      }
    }

    const handleClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      nodes.forEach((n) => {
        if (Math.hypot(n.x - x, n.y - y) < 60) n.heat = 1.0;
      });
    };
    canvas.addEventListener('click', handleClick);

    let animId: number;
    function draw() {
      if (!ctx) return;
      ctx.fillStyle = 'rgba(248, 250, 252, 0.35)';
      ctx.fillRect(0, 0, width, height);

      edges.forEach((e) => {
        const n1 = nodes[e.from],
          n2 = nodes[e.to];
        const diff = (n1.heat - n2.heat) * 0.035 * e.weight;
        n1.heat -= diff;
        n2.heat += diff;

        ctx.strokeStyle = `rgba(37, 99, 235, ${0.15 + (n1.heat + n2.heat) * 0.5})`;
        ctx.lineWidth = 1 + (n1.heat + n2.heat) * 2.5;
        ctx.beginPath();
        ctx.moveTo(n1.x, n1.y);
        ctx.lineTo(n2.x, n2.y);
        ctx.stroke();
      });

      nodes.forEach((n) => {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 15 || n.x > width - 15) n.vx *= -1;
        if (n.y < 15 || n.y > height - 15) n.vy *= -1;
        n.heat = Math.max(0.01, n.heat * 0.995);

        ctx.fillStyle = n.heat > 0.3 ? '#2563eb' : '#94a3b8';
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius + n.heat * 3, 0, Math.PI * 2);
        ctx.fill();
      });
      animId = requestAnimationFrame(draw);
    }
    draw();

    return () => {
      cancelAnimationFrame(animId);
      canvas.removeEventListener('click', handleClick);
    };
  }, []);

  return (
    <div className="space-y-4 max-w-5xl mx-auto w-full">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs font-bold text-brand-blue dark:text-laser-cyan uppercase tracking-wider block">Step 3 · Filtering</span>
          <h2 className="text-2xl sm:text-3xl font-bold text-brand-navy dark:text-white">Interactive Spectral Heat Diffusion Simulator</h2>
        </div>
        <span className="text-xs font-mono text-slate-500 dark:text-slate-400 flex items-center gap-1">
          <Flame className="w-3.5 h-3.5 text-amber-500" /> Click network to inject heat
        </span>
      </div>

      <div className="grid md:grid-cols-12 gap-5 items-center">
        <div className="md:col-span-8 glass-card p-2 rounded-2xl border border-slate-200 dark:border-white/10 shadow-lg">
          <canvas ref={canvasRef} className="w-full h-[300px] rounded-xl bg-slate-50 dark:bg-black/40 cursor-pointer" />
        </div>

        <div className="md:col-span-4 space-y-3 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
          <div className="glass-card p-4 rounded-xl border-l-4 border-l-brand-blue space-y-1.5">
            <span className="text-xs font-bold text-slate-900 dark:text-white block">How Diffusion Cleans Expression:</span>
            <p>1. Microarray technical noise acts as high-frequency spikes.</p>
            <p>2. Heat diffusion smooths noise across neighboring pathway proteins.</p>
            <p>3. Preserves biological signal while removing platform noise.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
