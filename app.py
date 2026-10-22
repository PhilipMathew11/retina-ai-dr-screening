from __future__ import annotations

import json
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.gradcam import make_gradcam, overlay_heatmap
from src.preprocessing import prepare_model_input, preprocess_fundus

MODEL_PATH = Path("models/dr_screening.keras")
EVALUATION_PATH = Path("models/evaluation.json")
DISCLAIMER = (
    "Educational prototype only. Not a medical diagnostic tool. "
    "It does not replace professional ophthalmological assessment."
)

st.set_page_config(
    page_title="RetinaAI | Retinal Image Analysis",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #20243a;
        --muted: #74788c;
        --teal: #5b50cf;
        --cyan: #8d83ef;
        --line: #e8e5f1;
        --canvas: #f7f6fb;
        --coral: #ec806b;
        --white: #ffffff;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Aptos', sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--canvas); }
    [data-testid="stHeader"] { background: rgba(247,246,251,.94); }
    [data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    .block-container {
        max-width: 1440px;
        padding: 1.8rem 2.4rem 3rem;
    }
    h1, h2, h3 { color: var(--ink); font-family: 'Segoe UI', 'Aptos', sans-serif; }
    h1 { font-size: 2.05rem !important; letter-spacing: -.055em; }
    h2 { font-size: 1.48rem !important; letter-spacing: -.035em; }
    h3 { font-size: 1.08rem !important; letter-spacing: -.025em; }
    p, [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 15px 18px;
        box-shadow: 0 5px 18px rgba(19,42,58,.035);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stFileUploader"] section {
        background: #fff;
        border: 1px dashed #b8b1e8;
        border-radius: 18px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--teal);
        background: #fcfbff;
    }
    [data-testid="stButton"] button {
        border-radius: 12px;
        border: 1px solid #e0dded;
        font-weight: 700;
        min-height: 3rem;
        transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
    }
    [data-testid="stButton"] button[kind="primary"] {
        color: #fff;
        background: #5b50cf;
        border-color: #5b50cf;
        box-shadow: 0 7px 16px rgba(91,80,207,.2);
    }
    [data-testid="stButton"] button[kind="primary"]:hover {
        background: #493eb8;
        border-color: #493eb8;
        box-shadow: 0 10px 20px rgba(91,80,207,.27);
        transform: translateY(-1px);
    }
    [data-testid="stTabs"] button { font-weight: 700; }
    [data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid #edf2f3;
    }
    .brand {
        display:flex; align-items:center; gap:11px;
        padding: 9px 8px 21px;
        border-bottom:1px solid var(--line);
        margin-bottom: 18px;
    }
    .brand-mark {
        width:40px; height:40px; border-radius:13px;
        display:grid; place-items:center;
        color:#fff; font-size:21px;
        background:linear-gradient(145deg,#5b50cf,#887cf0);
        box-shadow:0 7px 16px rgba(91,80,207,.22);
    }
    .brand-name { font:800 18px 'Segoe UI',sans-serif; color:var(--ink); }
    .brand-sub { font-size:10px; color:var(--muted); margin-top:1px; }
    .side-label {
        color:#8a9ba5; text-transform:uppercase; letter-spacing:.12em;
        font-size:10px; font-weight:800; padding:9px 8px 7px;
    }
    .side-link {
        color:#536975; display:block; padding:9px 11px; border-radius:9px;
        text-decoration:none!important; font-size:13px; font-weight:600;
    }
    .side-link.active { color:var(--teal); background:#f0efff; }
    .side-note {
        margin-top:22px; padding:13px; border-radius:12px;
        background:#f6f4ff; border:1px solid #e9e6fb;
        font-size:11px; color:#526b75; line-height:1.5;
    }
    .hero {
        padding:24px 28px; border:1px solid var(--line); border-radius:20px;
        background:linear-gradient(105deg,#fff 0%,#fbfaff 66%,#f0efff 100%);
        box-shadow:0 8px 26px rgba(19,42,58,.04);
        margin: 0 0 20px;
    }
    .eyebrow {
        display:flex; align-items:center; gap:8px;
        color:var(--teal); text-transform:uppercase; letter-spacing:.13em;
        font-size:10px; font-weight:800; margin-bottom:8px;
    }
    .hero-title {
        color:var(--ink); font:800 28px 'Segoe UI',sans-serif;
        letter-spacing:-.055em; margin:0;
    }
    .hero-copy { color:var(--muted); font-size:13px; margin:7px 0 0; }
    .ready {
        display:inline-flex; align-items:center; gap:8px; padding:8px 11px;
        background:white; border:1px solid var(--line); border-radius:999px;
        color:#47706f; font-size:11px; font-weight:700; white-space:nowrap;
    }
    .ready-dot { width:7px;height:7px;border-radius:50%;background:#2aab83; }
    .section-kicker {
        color:#82949e; font-size:10px; font-weight:800; letter-spacing:.13em;
        text-transform:uppercase; margin:7px 0 4px;
    }
    .pipeline {
        display:grid; grid-template-columns:repeat(6,minmax(0,1fr));
        gap:8px; margin: 12px 0 26px;
    }
    .pipe-step {
        position:relative; padding:13px 10px 12px; border-radius:13px;
        border:1px solid var(--line); background:#fff; min-height:75px;
    }
    .pipe-step.done {     background:#f2f0ff; border-color:#d9d5fb; }
    .pipe-step.active {
        background:#fff; border-color:#8075e4;
        box-shadow:0 0 0 3px rgba(91,80,207,.11);
    }
    .pipe-step.wait { background:#fff; }
    .pipe-index { font-size:9px; font-weight:800; color:#94a6ae; letter-spacing:.12em; }
    .pipe-name { font-size:11px; font-weight:700; color:#38515e; margin-top:5px; }
    .pipe-step.done .pipe-name { color:#5b50cf; }
    .pipe-step.active .pipe-index, .pipe-step.active .pipe-name { color:#5b50cf; }
    .pipe-status {
        width:6px; height:6px; border-radius:50%; display:inline-block;
        background:#c8c8d5; margin-right:5px; vertical-align:middle;
    }
    .pipe-step.done .pipe-status { background:#7468df; }
    .pipe-step.active .pipe-status { background:#ec806b; box-shadow:0 0 0 3px #f9e7e2; }
    .upload-card {
        background:#fff; border:1px solid var(--line); border-radius:17px;
        padding:17px 19px; margin-bottom:10px;
        box-shadow:0 5px 18px rgba(19,42,58,.035);
    }
    .upload-title { font:700 15px 'Segoe UI',sans-serif; color:var(--ink); }
    .upload-copy { color:var(--muted); font-size:12px; margin-top:3px; }
    .image-card {
        background:#fff; border:1px solid var(--line); border-radius:16px;
        padding:15px; height:100%;
        box-shadow:0 5px 18px rgba(19,42,58,.035);
    }
    .image-title { color:var(--ink); font:700 14px 'Segoe UI',sans-serif; }
    .image-step {
        color:var(--teal); font-size:9px; font-weight:800;
        letter-spacing:.13em; text-transform:uppercase; margin-bottom:6px;
    }
    .image-copy { color:var(--muted); font-size:11px; margin:5px 0 0; }
    .ai-workspace {
        background:#fff; border:1px solid #e5e1f0; border-radius:20px;
        padding:22px; box-shadow:0 12px 32px rgba(44,37,94,.07);
        position:relative; overflow:hidden;
    }
    .ai-workspace:before {
        content:""; position:absolute; top:0; left:0; bottom:0; width:5px;
        background:linear-gradient(180deg,#6255db,#ec806b);
    }
    .ai-kicker { color:#827aab; font-size:10px; font-weight:800; letter-spacing:.15em; }
    .ai-heading { color:#20243a; font:800 21px 'Segoe UI',sans-serif; margin:5px 0 3px; }
    .ai-description { color:#74788c; font-size:12px; line-height:1.55; max-width:570px; }
    .ai-chip {
        display:inline-flex; align-items:center; gap:7px; margin-top:13px;
        background:#f2f0ff; border:1px solid #e4e0fb; color:#6258b9;
        border-radius:999px; padding:6px 10px; font-size:10px; font-weight:800;
    }
    .ai-chip-dot { width:6px;height:6px;border-radius:50%;background:#766be0; }
    .ai-action-note {
        background:#f8f7fd; border:1px solid #ebe8f5; color:#74788c;
        border-radius:12px; padding:11px 13px; font-size:11px; line-height:1.45;
    }
    .analysis-wait {
        display:flex; align-items:center; gap:10px; margin-top:12px;
        color:#6e668e; font-size:11px;
    }
    .analysis-wait-mark {
        width:24px;height:24px;border-radius:8px;background:#f0efff;color:#5b50cf;
        display:grid;place-items:center;font-size:12px;font-weight:800;
    }
    .result-card {
        background:linear-gradient(120deg,#292448,#423779);
        color:#fff; padding:22px 24px; border-radius:18px;
        box-shadow:0 13px 28px rgba(54,45,112,.18);
        border:1px solid rgba(255,255,255,.12);
    }
    .result-card * { color:#fff!important; }
    .result-label { color:#c9c3ff!important; font-size:10px; font-weight:800; letter-spacing:.14em; }
    .result-title { font:800 24px 'Segoe UI',sans-serif; margin:7px 0 3px; }
    .result-copy { color:#e0dcf4!important; font-size:12px; }
    .model-placeholder {
        background:#fbfaff; border:1px dashed #c6c0ec; border-radius:15px;
        padding:18px; color:#526b75; font-size:13px; line-height:1.55;
    }
    .disclaimer {
        padding:11px 14px; border-radius:11px; background:#fff8ea;
        border:1px solid #f0e0b9; color:#766139; font-size:11px;
    }
    .stat-pill {
        display:inline-block; padding:5px 9px; margin:3px 5px 2px 0;
        border-radius:999px; background:#f1efff; color:#5c54a8;
        font-size:10px; font-weight:700;
    }
    @media (max-width: 850px) {
        .block-container { padding:1rem 1rem 2rem; }
        .pipeline { grid-template-columns:repeat(3,minmax(0,1fr)); }
        .hero-title { font-size:23px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def pipeline_html(uploaded: bool, model_ready: bool, prediction_ready: bool) -> str:
    stages = [
        ("01", "Input", uploaded),
        ("02", "Preprocess", uploaded),
        ("03", "Enhance", uploaded),
        ("04", "Segment", uploaded),
        ("05", "AI Analysis", model_ready and prediction_ready),
        ("06", "Result", model_ready and prediction_ready),
    ]
    active_index = (
        4 if uploaded and model_ready and not prediction_ready else
        0 if not uploaded else None
    )
    cells = ""
    for index, (number, name, done) in enumerate(stages):
        state = "done" if done else "active" if index == active_index else "wait"
        state_text = "Complete" if done else "In progress" if state == "active" else "Waiting"
        cells += (
            f'<div class="pipe-step {state}">'
            f'<div class="pipe-index"><span class="pipe-status"></span>{number} · {state_text}</div>'
            f'<div class="pipe-name">{name}</div></div>'
        )
    return f'<div class="pipeline">{cells}</div>'


def clear_analysis() -> None:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_signature", None)
    st.session_state["upload_generation"] = (
        st.session_state.get("upload_generation", 0) + 1
    )


def image_panel(number: str, title: str, copy: str, image: np.ndarray, caption: str = "") -> None:
    with st.container(border=True):
        st.markdown(
            f'<div class="image-step">{number}</div>'
            f'<div class="image-title">{title}</div><div class="image-copy">{copy}</div>',
            unsafe_allow_html=True,
        )
        st.image(image, caption=caption, width="stretch")


@st.cache_data(show_spinner=False, max_entries=4)
def process_upload(image_bytes: bytes):
    """Cache the existing DIP result so UI reruns don't repeat image processing."""
    pil_image = Image.open(BytesIO(image_bytes))
    if pil_image.width * pil_image.height > 50_000_000:
        raise ValueError("Image is too large to process (limit: 50 megapixels).")
    image_rgb = np.asarray(pil_image.convert("RGB"))
    return image_rgb, preprocess_fundus(image_rgb)


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◉</div><div>'
        '<div class="brand-name">RetinaAI</div>'
        '<div class="brand-sub">IMAGE ANALYSIS WORKSPACE</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
    st.markdown('<a class="side-link active" href="#dashboard">◈ &nbsp; Dashboard</a>', unsafe_allow_html=True)
    st.markdown('<a class="side-link" href="#image-analysis">◉ &nbsp; Image Analysis</a>', unsafe_allow_html=True)
    st.markdown('<a class="side-link" href="#processing-pipeline">⌁ &nbsp; Processing Pipeline</a>', unsafe_allow_html=True)
    st.markdown('<a class="side-link" href="#model-information">▣ &nbsp; Model Information</a>', unsafe_allow_html=True)
    st.markdown('<a class="side-link" href="#about-project">ⓘ &nbsp; About Project</a>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-note"><b>Educational Prototype</b><br>'
        'For learning and demonstration only.<br><br>'
        'Not a medical diagnostic tool.</div>',
        unsafe_allow_html=True,
    )
    st.button(
        "＋  New image analysis",
        use_container_width=True,
        on_click=clear_analysis,
    )

model_ready = MODEL_PATH.is_file()
upload_generation = st.session_state.get("upload_generation", 0)
st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)
left, right = st.columns([5, 1.25], vertical_alignment="center")
with left:
    st.markdown(
        '<div class="hero"><div class="eyebrow">DIGITAL IMAGE PROCESSING + TRANSFER LEARNING</div>'
        '<div class="hero-title">AI-Assisted Retinal Image Analysis</div>'
        '<div class="hero-copy">Explore how one fundus photograph moves through image processing and model analysis.</div></div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div style="text-align:right;padding:0 0 20px">'
        '<span class="ready"><span class="ready-dot"></span> System Ready</span></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="disclaimer">⚕ &nbsp;<b>Educational prototype only.</b> '
    'Not a medical diagnostic tool and does not replace professional ophthalmological assessment.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div id="image-analysis"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Start an analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="upload-card"><div class="upload-title">Upload Fundus Image</div>'
    '<div class="upload-copy">JPG, JPEG or PNG · Choose a retinal fundus photograph to begin processing</div></div>',
    unsafe_allow_html=True,
)
uploaded = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key=f"fundus_upload_{upload_generation}",
)

processed = None
image_rgb = None
image_signature = None
if uploaded is not None:
    try:
        image_bytes = uploaded.getvalue()
        image_signature = hashlib.sha256(image_bytes).hexdigest()
        if st.session_state.get("analysis_signature") != image_signature:
            st.session_state.pop("analysis_result", None)
            st.session_state["analysis_signature"] = image_signature
        image_rgb, processed = process_upload(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Could not process this image: {error}")
        st.stop()

st.markdown('<div class="section-kicker">Processing pipeline</div>', unsafe_allow_html=True)
st.markdown(
    pipeline_html(
        uploaded is not None,
        model_ready,
        bool(
            image_signature
            and st.session_state.get("analysis_signature") == image_signature
            and st.session_state.get("analysis_result") is not None
        ),
    ),
    unsafe_allow_html=True,
)

if processed is not None and image_rgb is not None:
    st.markdown('<h2>Digital Image Processing</h2>', unsafe_allow_html=True)
    st.markdown(
        "Follow the same retinal image through ROI extraction, denoising, contrast enhancement, "
        "and a classical vessel-candidate mask."
    )

    original_col, roi_col = st.columns(2, gap="large")
    with original_col:
        image_panel(
            "01 · FUNDUS INPUT",
            "Original image",
            "Uploaded RGB fundus photograph — the starting point for the pipeline.",
            processed.original,
        )
    with roi_col:
        image_panel(
            "02 · PREPROCESSING",
            "ROI & noise reduction",
            "Dark border is cropped where detected; a mild Gaussian filter reduces noise.",
            processed.cropped,
        )

    enhanced_col, vessel_col = st.columns(2, gap="large")
    with enhanced_col:
        image_panel(
            "03 · CONTRAST ENHANCEMENT",
            "CLAHE enhanced image",
            "CLAHE on LAB lightness improves local retinal contrast.",
            processed.enhanced,
        )
    with vessel_col:
        image_panel(
            "04 · MORPHOLOGY & SEGMENTATION",
            "Vessel candidate mask",
            "Green-channel contrast, Otsu thresholding, and morphological cleanup.",
            processed.vessel_mask,
            "Approximate classical segmentation demonstration",
        )
    st.markdown(
        '<span class="stat-pill">RGB → LAB</span><span class="stat-pill">Gaussian filter</span>'
        '<span class="stat-pill">CLAHE</span><span class="stat-pill">Otsu threshold</span>'
        '<span class="stat-pill">Morphological opening / closing</span>',
        unsafe_allow_html=True,
    )
    with st.expander("Inspect extracted image features"):
        feature_cols = st.columns(len(processed.features))
        for column, (name, value) in zip(feature_cols, processed.features.items()):
            column.metric(name.replace("_", " ").title(), f"{value:.3f}")
    st.caption(
        "The vessel mask and simple statistics are classical DIP demonstrations; neither is a "
        "clinically validated segmentation or diagnosis."
    )

    st.markdown("---")
    st.markdown("<h2>AI Analysis</h2>", unsafe_allow_html=True)
    if model_ready:
        st.markdown(
            '<div class="ai-workspace"><div class="ai-kicker">MODEL INFERENCE</div>'
            '<div class="ai-heading">Ready for image analysis</div>'
            '<div class="ai-description">The processed image will be passed to the local '
            'EfficientNetB0 classifier. The analysis returns a model probability and an '
            'approximate Grad-CAM view.</div>'
            '<div class="ai-chip"><span class="ai-chip-dot"></span> LOCAL MODEL CONNECTED</div></div>',
            unsafe_allow_html=True,
        )
        action_col, note_col = st.columns([1, 1.75], vertical_alignment="center")
        analysis_result = st.session_state.get("analysis_result")
        with action_col:
            run_analysis = st.button(
                "✦  Re-run analysis" if analysis_result else "✦  Analyze this image",
                type="primary",
                use_container_width=True,
            )
        with note_col:
            st.markdown(
                '<div class="ai-action-note"><b>One image · one prediction</b><br>'
                'Uses the saved local model. This action does not train or modify it.</div>',
                unsafe_allow_html=True,
            )
        if run_analysis:
            try:
                import tensorflow as tf

                with st.spinner("Analyzing image and generating Grad-CAM…"):
                    model = tf.keras.models.load_model(MODEL_PATH)
                    model_input = prepare_model_input(image_rgb)
                    probability_dr = float(
                        model.predict(model_input[None, ...], verbose=0)[0][0]
                    )
                    predicted_dr = probability_dr >= 0.5
                    class_name = "Diabetic Retinopathy" if predicted_dr else "Normal"
                    class_probability = probability_dr if predicted_dr else 1.0 - probability_dr
                    heatmap = make_gradcam(
                        model, model_input[None, ...], class_index=int(predicted_dr)
                    )
                    gradcam_image = overlay_heatmap(model_input.astype(np.uint8), heatmap)
                st.session_state["analysis_result"] = {
                    "class_name": class_name,
                    "probability_dr": probability_dr,
                    "class_probability": class_probability,
                    "model_input": model_input.astype(np.uint8),
                    "gradcam_image": gradcam_image,
                }
                st.session_state["analysis_signature"] = image_signature
                st.rerun()
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Prediction could not be completed: {error}")

        analysis_result = st.session_state.get("analysis_result")
        if (
            analysis_result
            and image_signature
            and st.session_state.get("analysis_signature") == image_signature
        ):
            class_name = analysis_result["class_name"]
            class_probability = analysis_result["class_probability"]
            probability_dr = analysis_result["probability_dr"]
            st.markdown(
                f'<div class="result-card" style="margin-top:14px">'
                f'<div class="result-label">ANALYSIS RESULT · EDUCATIONAL ONLY</div>'
                f'<div class="result-title">{class_name}</div>'
                f'<div class="result-copy">Model probability for predicted class: {class_probability:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(class_probability, text=f"Predicted-class probability · {class_probability:.1%}")
            st.caption(
                f"Raw DR-class probability: {probability_dr:.1%}. This score may not be "
                "calibrated and is not clinical certainty."
            )
            explain_col, source_col = st.columns(2, gap="large")
            with source_col:
                image_panel(
                    "REFERENCE",
                    "Model input",
                    "Preprocessed image used by the classifier.",
                    analysis_result["model_input"],
                )
            with explain_col:
                image_panel(
                    "MODEL EXPLAINABILITY",
                    "Grad-CAM overlay",
                    "Highlighted regions approximately indicate image areas that influenced the prediction.",
                    analysis_result["gradcam_image"],
                )
            st.caption(
                "Grad-CAM is an approximate model explanation, not a confirmed lesion or medical finding."
            )
            st.markdown(
                '<div class="section-kicker">Retinal image analysis complete</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="result-card" style="margin-top:14px">'
                f'<div class="result-label">RETINAL IMAGE ANALYSIS COMPLETE</div>'
                f'<div class="result-copy" style="margin-top:7px">'
                f'Prediction: <b>{class_name}</b> &nbsp;·&nbsp; '
                f'Predicted-class probability: <b>{class_probability:.1%}</b>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="model-placeholder"><b>Model not yet trained — prediction will appear after model integration.</b><br>'
            'Image processing above is available now. No AI class or confidence is being shown.</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div id="model-information"></div>', unsafe_allow_html=True)
with st.expander("Model information & evaluation", expanded=False):
    st.markdown(
        "The model is a binary EfficientNetB0 transfer-learning classifier. A saved test evaluation "
        "describes model performance on a dataset; it does not predict the uploaded image."
    )
    if EVALUATION_PATH.is_file():
        try:
            evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
            if evaluation.get("evaluation_scope") == "preliminary_checkpoint":
                st.warning(
                    "Preliminary checkpoint test evaluation: training was interrupted before completion. "
                    "These are not final model performance results."
                )
            st.json(evaluation)
            matrix_path = MODEL_PATH.parent / "confusion_matrix.png"
            if matrix_path.is_file():
                st.image(
                    str(matrix_path),
                    caption="Held-out test confusion matrix (dataset-level evaluation)",
                    width="stretch",
                )
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Saved evaluation information could not be loaded: {error}")
    else:
        st.info("Evaluation metrics will appear after a test evaluation is saved.")

history_path = MODEL_PATH.parent / "training_curves.png"
if history_path.is_file():
    with st.expander("Training and validation curves"):
        st.image(str(history_path), caption="Training history; not test-set performance.")

st.markdown('<div id="about-project"></div>', unsafe_allow_html=True)
with st.expander("About this project"):
    st.markdown(
        "This academic prototype demonstrates image validation, ROI cropping, Gaussian noise "
        "reduction, LAB/CLAHE contrast enhancement, green-channel thresholding, morphology, "
        "simple feature extraction, transfer learning, and Grad-CAM."
    )
    st.markdown(
        '<div class="disclaimer">Educational prototype only. Not a medical diagnostic tool. '
        'It does not replace professional ophthalmological assessment.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Uploaded images are processed in the current app session and are not added to training data. "
    "Do not upload images unless you have permission to use them."
)
