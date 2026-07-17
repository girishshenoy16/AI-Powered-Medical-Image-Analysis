"""PowerBI-style Streamlit PACS workstation.

A single-page, single-tab clinical dashboard that mirrors a hospital
radiology workflow: KPI metric tiles, a sidebar with an uploader / metadata
filters / patient directory, a side-by-side PACS viewer (Original vs.
Preprocessed vs. Grad-CAM), a diagnostic findings card, and a searchable
historical case table. No charts are rendered (per the design spec) - all
evaluation plots live on disk under ``outputs/plots``.

Run with:  streamlit run src/dashboard.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import _env  # must be first: silence TF startup logs
except ModuleNotFoundError:
    import src._env as _env

from config import (
    CASES_DIR,
    CLASS_NAMES,
    DIAGNOSTIC_CSV,
    MODEL_PATH,
    PLOTS_DIR,
    PREPROCESSED_DIR,
)
from logger import get_logger
from predict import load_model, predict_and_record

logger = get_logger(__name__)

EVAL_JSON = PLOTS_DIR.parent / "evaluation_metrics.json"
N_SEED_CASES = 12  # balanced sample cases seeded on first launch


@st.cache_resource(show_spinner="Loading AI model...")
def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_model()


@st.cache_data(show_spinner="Reading diagnostic database...")
def load_cases() -> pd.DataFrame:
    if not DIAGNOSTIC_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(DIAGNOSTIC_CSV)
    for col in ("age", "confidence", "latency_ms"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_eval_metrics() -> dict | None:
    if not EVAL_JSON.exists():
        return None
    import json

    return json.loads(EVAL_JSON.read_text())


def seed_sample_cases(model) -> None:
    """Populate the CSV + case images from a balanced slice of the test set."""
    test_root = PREPROCESSED_DIR / "test"
    if not test_root.exists():
        return
    selected = []
    for class_name in CLASS_NAMES:
        files = sorted((test_root / class_name).glob("*.png"))
        step = max(1, len(files) // (N_SEED_CASES // 2))
        selected += files[::step][: N_SEED_CASES // 2]
    import random

    random.Random(7).shuffle(selected)
    for img_path in selected[:N_SEED_CASES]:
        try:
            predict_and_record(img_path, preprocessed=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Seed case failed for %s: %s", img_path, exc)


def load_case_images(patient_id: str) -> dict:
    case_dir = CASES_DIR / patient_id
    if not case_dir.exists():
        return {}
    out = {}
    for key in ("original", "preprocessed", "gradcam"):
        p = case_dir / f"{patient_id}_{key}.png"
        if p.exists():
            out[key] = Image.open(p)
    return out


def to_pil(array: np.ndarray) -> Image.Image:
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def render_pacs(original, preprocessed, gradcam) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1 · Original Scan**")
        st.image(to_pil(original), width="stretch")
    with c2:
        st.markdown("**2 · Preprocessed (CLAHE)**")
        st.image(to_pil(preprocessed), width="stretch", clamp=True)
    with c3:
        st.markdown("**3 · Grad-CAM Overlay**")
        st.image(to_pil(gradcam), width="stretch")


def recommendation_for(class_name: str, confidence: float) -> str:
    if class_name == "PNEUMONIA":
        return (
            "Opacity patterns consistent with pneumonia. Recommend correlating "
            "with clinical signs, inflammatory markers, and a follow-up review "
            "by a radiologist. Isolation precautions if infectious aetiology suspected."
        )
    return (
        "No significant opacification detected. Routine clinical correlation "
        "advised; repeat imaging if symptoms persist or worsen."
    )


def main() -> None:
    st.set_page_config(page_title="AI Medical Image Analysis", layout="wide")

    model = get_model()

    # ----- Header banner -----
    status = "ONLINE" if model is not None else "OFFLINE (no model)"
    backbone = "MobileNetV2 (transfer)" if model is not None else "-"
    st.title("AI-Powered Medical Image Analysis")
    st.caption(
        f"PACS Radiology Workstation  |  System State: **{status}**  |  "
        f"Model: **{backbone}**  |  Input: 160x160 grayscale"
    )
    st.divider()

    # ----- KPI metric bar -----
    df = load_cases()
    eval_metrics = load_eval_metrics()

    total_audited = len(df)
    accuracy = eval_metrics.get("accuracy") if eval_metrics else None
    positivity = (
        float((df["prediction"] == "PNEUMONIA").mean()) * 100 if total_audited else 0.0
    )
    latency = float(df["latency_ms"].mean()) if total_audited else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Audited", f"{total_audited}")
    k2.metric("System Accuracy", f"{accuracy*100:.1f}%" if accuracy else "N/A")
    k3.metric("Positivity Rate", f"{positivity:.1f}%")
    k4.metric("Avg. Latency", f"{latency:.1f} ms")
    st.divider()

    # On first launch, seed a few sample cases so the workstation is populated.
    if model is not None and total_audited == 0:
        with st.spinner("Seeding sample cases from the test set..."):
            seed_sample_cases(model)
        st.cache_data.clear()
        df = load_cases()
        total_audited = len(df)

    # ----- Sidebar -----
    st.sidebar.header("Patient Console")

    uploaded = st.sidebar.file_uploader(
        "Upload Chest X-Ray", type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        with st.spinner("Running diagnosis + Grad-CAM..."):
            tmp = CASES_DIR.parent / "_upload_tmp" / uploaded.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(uploaded.getbuffer())
            result = predict_and_record(tmp)
            st.session_state["current"] = result
            st.cache_data.clear()
            df = load_cases()
            total_audited = len(df)
        st.sidebar.success(
            f"Diagnosis: **{result['class_name']}** "
            f"({result['confidence']*100:.1f}%)"
        )

    st.sidebar.divider()

    # Metadata filters
    st.sidebar.subheader("Filters")
    if total_audited:
        ages = df["age"].dropna()
        age_range = st.sidebar.slider(
            "Age range", int(ages.min()), int(ages.max()),
            (int(ages.min()), int(ages.max())),
        )
        gender = st.sidebar.selectbox("Gender", ["All", "Male", "Female"])
        scanners = sorted(df["scanner"].dropna().unique().tolist())
        scanner_sel = st.sidebar.multiselect("Scanner", scanners, default=scanners)

        filtered = df[
            df["age"].between(*age_range)
            & (df["gender"].eq(gender) if gender != "All" else True)
            & (df["scanner"].isin(scanner_sel))
        ]
    else:
        filtered = df

    # Patient directory
    st.sidebar.divider()
    st.sidebar.subheader("Patient Case Directory")
    directory = filtered.drop_duplicates("patient_id") if not filtered.empty else filtered
    if not directory.empty:
        labels = [
            f"{r.patient_id} · {r.prediction} ({float(r.confidence)*100:.0f}%)"
            for r in directory.itertuples()
        ]
        choice = st.sidebar.radio("Select case", labels, key="dir")
        selected_id = directory.iloc[labels.index(choice)]["patient_id"]
    else:
        selected_id = None
        st.sidebar.info("No cases yet.")

    # ----- Main visualization pane -----
    current = st.session_state.get("current")
    if current is not None:
        cls = current["class_name"]
        conf = current["confidence"]
        col_img, col_find = st.columns([2, 1])
        with col_img:
            render_pacs(current["original"], current["preprocessed"], current["overlay"])
        with col_find:
            st.subheader("Diagnostic Findings")
            color = "inverse" if cls == "PNEUMONIA" else "normal"
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;background:"
                f"{'#5c1a1a' if cls=='PNEUMONIA' else '#143d2b'};"
                f"color:white;font-size:20px;text-align:center;'>"
                f"<b>{cls}</b></div>",
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{conf*100:.1f}%")
            st.write(recommendation_for(cls, conf))
    elif selected_id is not None:
        row = filtered[filtered["patient_id"] == selected_id].iloc[0]
        imgs = load_case_images(selected_id)
        if imgs:
            col_img, col_find = st.columns([2, 1])
            with col_img:
                render_pacs(imgs.get("original"), imgs.get("preprocessed"), imgs.get("gradcam"))
            with col_find:
                st.subheader("Diagnostic Findings")
                st.markdown(
                    f"<div style='padding:10px;border-radius:8px;background:"
                    f"{'#5c1a1a' if row.prediction=='PNEUMONIA' else '#143d2b'};"
                    f"color:white;font-size:20px;text-align:center;'>"
                    f"<b>{row.prediction}</b></div>",
                    unsafe_allow_html=True,
                )
                st.metric("Confidence", f"{float(row.confidence)*100:.1f}%")
                st.write(recommendation_for(row.prediction, float(row.confidence)))
                st.caption(f"Patient {row.patient_id} · {row.gender} · "
                           f"age {int(row.age)} · {row.scanner}")
        else:
            st.info("Case imagery not available for this patient.")
    else:
        st.info("Upload a chest X-ray or select a patient from the directory "
                "to begin diagnosis.")

    # ----- Bottom historical table -----
    st.divider()
    st.subheader("Clinical Historical Database")
    search = st.text_input("Search (patient id, filename, scanner, prediction)")
    table = filtered
    if search and not table.empty:
        mask = (
            table.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
        )
        table = table[mask]
    if not table.empty:
        st.dataframe(table, use_container_width=True, height=300)
    else:
        st.caption("No records match the current filters.")


if __name__ == "__main__":
    main()
