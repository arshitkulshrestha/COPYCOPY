"""AI-Based Help Line Triage — Hinglish helpline transcript demo (v2.1).

Operator-support dashboard that pairs a trained ML triage classifier
(``models2.0/``) with a transparent, rule-based evidence layer
(``triage_rules.py``). Transcripts can be typed or produced from a short audio
clip via Sarvam AI speech-to-text. The ML model gives the headline routing label
with an uncalibrated model score; the keyword rules give human-readable evidence,
location clues, and information still to verify. They are kept separate so an
operator can cross-check them.

Fictional/demo transcripts only. This is decision support, not a validated
emergency-risk model, and it must never dispatch services.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from triage_rules import analyze_transcript

import emergency_services as emsvc

AUDIO_TYPES = ["wav", "mp3", "aac", "aiff", "ogg", "opus", "flac", "mp4", "m4a", "amr", "wma", "webm"]

# -----------------------------------------------------------------------------
# Page setup and design system — warm ivory / sage / muted teal / champagne
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Helpline Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ivory: #f5efe3;        /* warm page background */
        --cream: #fffdf8;        /* elevated card surface */
        --cream-2: #faf5ea;      /* secondary surface */
        --champagne: #ece0c8;    /* beige highlight */
        --sage: #7e9b84;         /* primary accent */
        --sage-deep: #5f7d66;
        --teal: #4f8688;         /* secondary accent */
        --teal-deep: #3c6b6d;
        --ink: #3a382f;          /* soft charcoal text */
        --muted: #8c867a;        /* muted text */
        --border: #e6dcc8;       /* elegant warm border */
        --shadow: 0 6px 20px rgba(90, 77, 48, 0.08);
        --shadow-sm: 0 2px 8px rgba(90, 77, 48, 0.06);

        --priority: #b05f45;  --priority-bg: #f4e6dd;  /* muted clay */
        --support:  #a9843f;  --support-bg:  #f3ebd6;  /* muted gold */
        --unclear:  #5f7d76;  --unclear-bg:  #e6ede8;  /* sage-teal */
      }

      /* Force the warm theme regardless of system light/dark preference. */
      .stApp, [data-testid="stAppViewContainer"] { background: var(--ivory); }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] { background: var(--cream-2); border-right: 1px solid var(--border); }
      .block-container { padding-top: 1.1rem; padding-bottom: 2.6rem; max-width: 1180px; }

      html, body, [class*="css"], .stMarkdown, p, span, label, li { color: var(--ink); }
      h1, h2, h3, h4, h5 { color: var(--ink); font-weight: 700; letter-spacing: -0.01em; }
      h2 { font-size: 1.28rem !important; }
      h5 { font-size: 1.02rem !important; margin-bottom: 0.2rem; }

      .app-header {
        display: flex; align-items: center; gap: 0.85rem;
        background: linear-gradient(180deg, var(--cream) 0%, var(--cream-2) 100%);
        border: 1px solid var(--border); border-left: 5px solid var(--sage);
        border-radius: 0.9rem; padding: 0.85rem 1.1rem; box-shadow: var(--shadow-sm);
        margin-bottom: 0.7rem;
      }
      .app-header .mark { font-size: 1.7rem; }
      .app-header .title { font-size: 1.32rem; font-weight: 700; color: var(--ink); line-height: 1.15; }
      .app-header .sub { font-size: 0.82rem; color: var(--muted); }

      [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--cream); border: 1px solid var(--border) !important;
        border-radius: 0.9rem; box-shadow: var(--shadow);
        padding: 0.3rem 0.25rem;
      }
      [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] { box-shadow: var(--shadow-sm); }

      .eyebrow {
        display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--sage-deep); margin-bottom: 0.1rem;
      }
      .card-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin: 0 0 0.3rem; }

      .triage-badge {
        display: inline-flex; align-items: center; gap: 0.45rem;
        padding: 0.42rem 0.9rem; border-radius: 0.65rem;
        font-weight: 700; font-size: 1rem; border: 1px solid transparent;
      }
      .badge-sm { font-size: 0.82rem; padding: 0.28rem 0.65rem; }
      .b-priority { background: var(--priority-bg); color: var(--priority); border-color: var(--priority); }
      .b-support  { background: var(--support-bg);  color: var(--support);  border-color: var(--support); }
      .b-unclear  { background: var(--unclear-bg);  color: var(--unclear);  border-color: var(--unclear); }
      .b-neutral  { background: var(--champagne); color: var(--ink); border-color: var(--border); }

      .score-row { margin: 0.4rem 0; }
      .score-label { font-size: 0.86rem; color: var(--ink); display: flex; justify-content: space-between; margin-bottom: 0.22rem; font-weight: 600; }
      .score-track { background: var(--champagne); border-radius: 7px; height: 0.55rem; overflow: hidden; }
      .score-fill { height: 100%; border-radius: 7px; }
      .f-priority { background: linear-gradient(90deg, #c07558, var(--priority)); }
      .f-support  { background: linear-gradient(90deg, #c2a05a, var(--support)); }
      .f-unclear  { background: linear-gradient(90deg, #8aa79c, var(--unclear)); }

      .chip {
        display: inline-block; background: var(--cream-2); color: var(--ink);
        border: 1px solid var(--border); border-radius: 0.55rem;
        padding: 0.18rem 0.55rem; margin: 0.14rem 0.22rem 0.14rem 0; font-size: 0.82rem;
      }
      .chip-sage { background: var(--unclear-bg); border-color: var(--sage); color: var(--sage-deep); }

      .kv { margin: 0.22rem 0; color: var(--ink); font-size: 0.93rem; }
      .kv b { color: var(--muted); font-weight: 600; }
      .muted { color: var(--muted); font-size: 0.84rem; }
      .verify-item {
        background: var(--cream-2); border-left: 3px solid var(--sage);
        border-radius: 0.4rem; padding: 0.4rem 0.6rem; margin: 0.3rem 0; font-size: 0.9rem;
      }

      .stButton > button {
        border-radius: 0.6rem; font-weight: 600; border: 1px solid var(--border);
        background: var(--cream); color: var(--ink); transition: all 0.15s ease;
      }
      .stButton > button:hover { border-color: var(--sage); color: var(--sage-deep); background: var(--cream-2); }
      .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, var(--sage) 0%, var(--sage-deep) 100%);
        color: #fff; border: 1px solid var(--sage-deep);
      }
      .stButton > button[kind="primary"]:hover { background: var(--sage-deep); color: #fff; }
      [data-testid="stDownloadButton"] > button { border-radius: 0.6rem; border: 1px solid var(--border); background: var(--cream); color: var(--ink); font-weight: 600; }
      [data-testid="stDownloadButton"] > button:hover { border-color: var(--teal); color: var(--teal-deep); }

      textarea, [data-baseweb="select"] > div, .stTextArea textarea {
        background: var(--cream) !important; border-radius: 0.6rem !important;
        border-color: var(--border) !important; color: var(--ink) !important;
      }
      [data-testid="stDataFrame"] { border-radius: 0.7rem; overflow: hidden; border: 1px solid var(--border); }
      [data-testid="stNotification"], .stAlert { border-radius: 0.7rem; }
      hr { border-color: var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
      <div class="mark">🛡️</div>
      <div>
        <div class="title">Helpline Triage — Operator Console</div>
        <div class="sub">Ten Seconds to Read a Call · Hinglish helpline triage · decision-support prototype</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "**Demo only — use fictional or appropriately authorised audio/text.** The ML label and keyword "
    "rules are decision support, not a validated urgency model; they can miss danger or flag harmless "
    "text. A trained operator must verify the transcript and follow local protocol. This tool must "
    "not independently dispatch emergency services.",
    icon="⚠️",
)


# -----------------------------------------------------------------------------
# models2.0: trained ML triage classifier (loaded by file path)
# -----------------------------------------------------------------------------
_MODELS2_DIR = Path(__file__).resolve().parent / "models2.0"
_TRIAGE_SPEC = importlib.util.spec_from_file_location(
    "triage_model_v2", _MODELS2_DIR / "triage_model.py"
)
triage_model_v2 = importlib.util.module_from_spec(_TRIAGE_SPEC)
sys.modules[_TRIAGE_SPEC.name] = triage_model_v2  # so dataclasses/typing resolve the module
try:
    _TRIAGE_SPEC.loader.exec_module(triage_model_v2)  # type: ignore[union-attr]
    _TRIAGE_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - defensive
    _TRIAGE_IMPORT_ERROR = str(exc)


@st.cache_resource(show_spinner="Loading triage model…")
def get_ml_model():
    if _TRIAGE_IMPORT_ERROR is not None:
        raise RuntimeError(_TRIAGE_IMPORT_ERROR)
    return triage_model_v2.load_model()


def ml_predict(text: str) -> dict[str, Any] | None:
    try:
        model = get_ml_model()
        return triage_model_v2.predict_transcript(text, model)
    except Exception:
        return None


ML_LABEL_META = {
    "PRIORITY_REVIEW": {"title": "Priority review", "emoji": "🔴", "badge": "b-priority", "fill": "f-priority"},
    "SUPPORT_REVIEW": {"title": "Support review", "emoji": "🟠", "badge": "b-support", "fill": "f-support"},
    "UNCLEAR_REVIEW": {"title": "Unclear — clarify", "emoji": "⚪", "badge": "b-unclear", "fill": "f-unclear"},
}


def ml_meta(label: str) -> dict[str, Any]:
    return ML_LABEL_META.get(
        label, {"title": label, "emoji": "•", "badge": "b-unclear", "fill": "f-unclear"}
    )


# -----------------------------------------------------------------------------
# Semantic Evidence Voting (SEV) — experimental alternative model
# -----------------------------------------------------------------------------
# NOTE (honest result): on the same untouched held-out split, the Logistic
# Regression baseline scores macro F1 0.939 vs SEV 0.748 (see
# models2.0/evaluation_semantic_vs_logreg.json). The multilingual encoder embeds
# this heavily romanized-Hinglish corpus weakly. LogReg is the default; SEV is
# offered for transparency and comparison, not as an improvement.
METHOD_LOGREG = "Logistic Regression"
METHOD_SEV = "Semantic Evidence Voting"
METHOD_FUSION = "Evidence Fusion"

_SEV_SPEC = importlib.util.spec_from_file_location(
    "sev_v2", _MODELS2_DIR / "semantic_evidence_voting.py"
)
sev_module = importlib.util.module_from_spec(_SEV_SPEC)
sys.modules[_SEV_SPEC.name] = sev_module  # required so @dataclass can resolve its module
try:
    _SEV_SPEC.loader.exec_module(sev_module)  # type: ignore[union-attr]
    _SEV_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - defensive
    _SEV_IMPORT_ERROR = str(exc)

_SEV_INDEX_PATH = _MODELS2_DIR / "sev_index.npz"

# Evidence Fusion module + its validation-tuned config.
_FUSION_SPEC = importlib.util.spec_from_file_location(
    "evidence_fusion_v2", _MODELS2_DIR / "evidence_fusion.py"
)
fusion_module = importlib.util.module_from_spec(_FUSION_SPEC)
sys.modules[_FUSION_SPEC.name] = fusion_module  # required so @dataclass can resolve its module
try:
    _FUSION_SPEC.loader.exec_module(fusion_module)  # type: ignore[union-attr]
    _FUSION_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - defensive
    _FUSION_IMPORT_ERROR = str(exc)

_FUSION_CONFIG_PATH = _MODELS2_DIR / "fusion_config.json"


def get_fusion_config():
    """Load the validation-tuned fusion config (weight + review thresholds)."""
    cfg = fusion_module.FusionConfig()
    try:
        import json
        data = json.loads(_FUSION_CONFIG_PATH.read_text())
        cfg.weight = float(data.get("weight", cfg.weight))
        cfg.low_confidence = float(data.get("low_confidence", cfg.low_confidence))
        cfg.low_margin = float(data.get("low_margin", cfg.low_margin))
    except Exception:
        pass
    return cfg


def sev_available() -> bool:
    """True only if sentence-transformers is importable and the index exists."""
    if _SEV_IMPORT_ERROR is not None or not _SEV_INDEX_PATH.exists():
        return False
    return importlib.util.find_spec("sentence_transformers") is not None


def fusion_available() -> bool:
    """Fusion needs the LogReg model, SEV, and the fusion module."""
    return _FUSION_IMPORT_ERROR is None and sev_available()


@st.cache_resource(show_spinner="Loading Semantic Evidence Voting index…")
def get_sev_voter():
    return sev_module.SemanticEvidenceVoter.load(_SEV_INDEX_PATH)


def predict_with_method(text: str, method: str) -> dict[str, Any] | None:
    """Unified prediction dict for either model (keys are model-agnostic)."""
    if method == METHOD_FUSION:
        try:
            lr = ml_predict(text)
            sev = get_sev_voter().predict_one(text)
            if not lr:
                return None
            fused = fusion_module.fuse(lr["class_scores"], sev["vote_proportions"], get_fusion_config())
            meta_labels = {
                "logreg": fused["logreg_label"], "sev": fused["sev_label"],
            }
            note = (
                f"Fused score = {fused['weight']:.0%} Logistic Regression + "
                f"{1 - fused['weight']:.0%} SEV (weight tuned on validation). "
                "Not a probability of real-world correctness."
            )
            return {
                "method": METHOD_FUSION,
                "predicted_label": fused["predicted_label"],
                "scores": fused["scores"],
                "confidence": fused["confidence"],
                "score_label": "fused score",
                "score_note": note,
                "supporting_examples": sev["supporting_examples"],
                "description": triage_model_v2.LABEL_DESCRIPTIONS.get(fused["predicted_label"], ""),
                "human_review_note": "Verify the transcript, current danger, caller location, and appropriate response with a trained operator.",
                "fusion": {**fused, "component_labels": meta_labels},
            }
        except Exception:
            return None
    if method == METHOD_SEV:
        try:
            r = get_sev_voter().predict_one(text)
            return {
                "method": METHOD_SEV,
                "predicted_label": r["predicted_label"],
                "scores": r["vote_proportions"],
                "confidence": r["confidence"],
                "score_label": "vote proportion",
                "score_note": (
                    f"Vote proportions are the normalised share of similarity mass among the "
                    f"top-{r['neighbours_used']} nearest training examples — not probabilities "
                    f"of real-world correctness."
                ),
                "supporting_examples": r["supporting_examples"],
                "description": triage_model_v2.LABEL_DESCRIPTIONS.get(r["predicted_label"], ""),
                "human_review_note": "Verify the transcript, current danger, caller location, and appropriate response with a trained operator.",
            }
        except Exception:
            return None
    # Default: Logistic Regression baseline.
    r = ml_predict(text)
    if not r:
        return None
    return {
        "method": METHOD_LOGREG,
        "predicted_label": r["predicted_label"],
        "scores": r["class_scores"],
        "confidence": r["model_confidence"],
        "score_label": "model score",
        "score_note": r.get("confidence_note", "Uncalibrated model score; not a probability of real-world correctness."),
        "supporting_examples": None,
        "description": r.get("description", ""),
        "human_review_note": r.get("human_review_note", ""),
    }


def card_header(eyebrow: str, title: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="card-title">{title}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sarvam AI speech-to-text (optional audio input)
# -----------------------------------------------------------------------------
def get_sarvam_api_key() -> str | None:
    """Streamlit Community Cloud secrets first; environment variable as fallback."""
    try:
        value = st.secrets.get("SARVAM_API_KEY", None)
        if value:
            return str(value).strip()
    except Exception:
        pass
    value = os.environ.get("SARVAM_API_KEY")
    return value.strip() if value else None


def get_geoapify_api_key() -> str | None:
    """Streamlit Community Cloud secrets first; environment variable as fallback."""
    try:
        value = st.secrets.get("GEOAPIFY_API_KEY", None)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return emsvc.get_api_key()  # reads GEOAPIFY_API_KEY from the environment


def get_google_places_api_key() -> str | None:
    """Streamlit Secrets first; GOOGLE_PLACES_API_KEY environment variable as fallback."""
    try:
        value = st.secrets.get("GOOGLE_PLACES_API_KEY", None)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return emsvc.get_google_api_key()


@st.cache_data(show_spinner=False, ttl=900)
def cached_emergency_search(
    location: str, resource_types: tuple[str, ...], radius_m: int,
    api_key: str, google_api_key: str | None = None,
):
    """Cache lookups for 15 min to respect API limits (keyed incl. both keys)."""
    result = emsvc.find_nearby(
        location, resource_types=resource_types, radius_m=radius_m,
        api_key=api_key, google_api_key=google_api_key,
    )
    # Return a plain dict so Streamlit can hash/serialise the cached value cleanly.
    return {
        "query_location": result.query_location,
        "coordinates": {
            "lat": result.coordinates.lat,
            "lon": result.coordinates.lon,
            "formatted": result.coordinates.formatted,
        },
        "radius_m": result.radius_m,
        "resources": {k: [r.as_dict() for r in v] for k, v in result.resources.items()},
        "errors": result.errors,
    }


def transcribe_audio(audio_bytes: bytes, filename: str, mode: str) -> tuple[str, str | None]:
    """Transcribe a short audio clip with Sarvam Saaras v4."""
    api_key = get_sarvam_api_key()
    if not api_key:
        raise RuntimeError("Sarvam API key is not configured. Add SARVAM_API_KEY to Streamlit Secrets or the environment.")

    from sarvamai import SarvamAI  # lazy import so the app runs without the package

    suffix = Path(filename).suffix or ".wav"
    temp_path = None
    client = SarvamAI(api_subscription_key=api_key)
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        with open(temp_path, "rb") as audio_file:
            response = client.speech_to_text.transcribe(file=audio_file, model="saaras:v4", mode=mode)
        transcript = getattr(response, "transcript", None)
        language_code = getattr(response, "language_code", None)
        if not transcript or not str(transcript).strip():
            raise RuntimeError("Sarvam returned an empty transcript. Try a clearer short clip.")
        return str(transcript).strip(), language_code
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# -----------------------------------------------------------------------------
# Sample data and case records
# -----------------------------------------------------------------------------
SAMPLE_TRANSCRIPTS = {
    "Following near metro": "A man is following me near the metro station. He is threatening me and I am scared right now.",
    "Being followed — help": "I am being followed by a man. Please send help.",
    "Threat at home": "My partner is threatening me at home. I do not feel safe right now.",
    "Repeated harassment": "Someone keeps sending unwanted messages and following me after college.",
    "Information request": "I need information about reporting harassment. There is no immediate danger that I know of.",
    "Hinglish example": "Woh mera peecha kar raha hai aur dhamki de raha hai. Mujhe darr lag raha hai abhi.",
    "Negation test": "He is not following me and he has not threatened me. I am asking how to report an old incident.",
    "Short / ambiguous": "Help please. I am scared.",
}


def make_case(case_id: str, transcript: str, created_at: str, source: str = "Analysed transcript",
              method: str = METHOD_LOGREG) -> dict[str, Any]:
    analysis = analyze_transcript(transcript)
    prediction = predict_with_method(transcript, method)
    if prediction and "predicted_label" in prediction:
        ml_label = prediction["predicted_label"]
        ml_score = f"{prediction['confidence'] * 100:.0f}%"
    else:
        ml_label = "Model unavailable"
        ml_score = "—"
    return {
        "case_id": case_id,
        "transcript": transcript,
        "model": method,
        "ml_label": ml_label,
        "ml_score": ml_score,
        "rule_cue": analysis["urgency"],
        "threat": analysis["threat"],
        "location": ", ".join(analysis["location_clues"]) if analysis["location_clues"] else "Not identified",
        "suggested_queue": analysis["suggested_queue"],
        "created_at": created_at,
        "review_status": "Pending review",
        "operator_notes": "",
        "source": source,
    }


SEED_SAMPLE_KEYS = ["Following near metro", "Being followed — help", "Repeated harassment", "Information request"]


def seed_cases() -> list[dict[str, Any]]:
    # Seeds always use the fast Logistic Regression baseline so startup never
    # depends on the heavy Sentence Transformer download.
    return [
        make_case(f"DEMO-{i:03d}", SAMPLE_TRANSCRIPTS[key], "Fictional example",
                  source="Built-in fictional example", method=METHOD_LOGREG)
        for i, key in enumerate(SEED_SAMPLE_KEYS, start=1)
    ]


for key, default in [
    ("cases", None), ("last_analysis", None), ("last_prediction", None),
    ("last_transcript", ""), ("last_case_id", None), ("case_counter", 1),
    ("last_detected_language", None),
]:
    if key not in st.session_state:
        st.session_state[key] = seed_cases() if key == "cases" else default


# -----------------------------------------------------------------------------
# Related-complaint matching: lexical similarity with a justified threshold
# -----------------------------------------------------------------------------
RELATED_THRESHOLD = 0.30
STRONG_THRESHOLD = 0.55


def _strength_band(score: float) -> str:
    if score >= STRONG_THRESHOLD:
        return "Strong"
    if score >= RELATED_THRESHOLD:
        return "Moderate"
    return "Weak"


def find_related_complaints(
    transcript: str,
    cases: list[dict[str, Any]],
    exclude_case_id: str | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, float]:
    columns = ["Case ID", "Match strength", "Lexical overlap", "Previous transcript", "ML label"]
    candidates = [
        case for case in cases
        if case.get("case_id") != exclude_case_id and str(case.get("transcript", "")).strip()
    ]
    if not transcript.strip() or not candidates:
        return pd.DataFrame(columns=columns), 0.0

    documents = [transcript] + [case["transcript"] for case in candidates]
    try:
        word_matrix = TfidfVectorizer(
            ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=30000,
        ).fit_transform(documents)
        char_matrix = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode",
            sublinear_tf=True, max_features=40000,
        ).fit_transform(documents)
        word_sim = cosine_similarity(word_matrix[0:1], word_matrix[1:]).ravel()
        char_sim = cosine_similarity(char_matrix[0:1], char_matrix[1:]).ravel()
        scores = (0.5 * word_sim) + (0.5 * char_sim)
    except ValueError:
        return pd.DataFrame(columns=columns), 0.0

    best = float(scores.max()) if len(scores) else 0.0
    rows = []
    for case, score in zip(candidates, scores):
        s = float(score)
        if s < RELATED_THRESHOLD:
            continue
        rows.append({
            "Case ID": case.get("case_id", "Unknown"),
            "Match strength": _strength_band(s),
            "Lexical overlap": f"{s * 100:.0f}%",
            "_s": s,
            "Previous transcript": case.get("transcript", ""),
            "ML label": ml_meta(case.get("ml_label", "")).get("title", case.get("ml_label", "")),
        })
    rows.sort(key=lambda r: r["_s"], reverse=True)
    frame = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "_s"} for r in rows[:top_n]], columns=columns
    )
    return frame, best


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🛡️ Operator Console")
st.sidebar.caption("Decision-support prototype · fictional examples only.")

PAGE_TRIAGE = "🩺  Triage Dashboard"
PAGE_PERFORMANCE = "📊  Model Performance"
PAGE_EMERGENCY = "📍  Emergency Resources"
nav = st.sidebar.radio(
    "Navigate",
    [PAGE_TRIAGE, PAGE_PERFORMANCE, PAGE_EMERGENCY],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("**System status**")
if ml_predict("status check") is not None:
    st.sidebar.success("ML triage model: loaded", icon="✅")
else:
    st.sidebar.error("ML model unavailable — keyword rules only.", icon="⚠️")
if get_sarvam_api_key():
    st.sidebar.success("Sarvam speech-to-text: configured", icon="🎙️")
else:
    st.sidebar.info("Sarvam key not set — typed transcripts still work.", icon="🎙️")
if get_geoapify_api_key():
    st.sidebar.success("Geoapify (maps/search): configured", icon="📍")
else:
    st.sidebar.info("Geoapify key not set — nearby-resources lookup off.", icon="📍")
if get_google_places_api_key():
    st.sidebar.success("Google Places (ratings/reviews): configured", icon="⭐")
else:
    st.sidebar.info("Google Places key not set — ratings/reviews off.", icon="⭐")

st.sidebar.divider()
st.sidebar.caption(
    "Decision support only — this tool must never auto-dispatch emergency services. "
    "Class scores / vote proportions are uncalibrated model outputs on synthetic data, "
    "not probabilities of real-world correctness."
)


# -----------------------------------------------------------------------------
# Page: Model Performance & Comparison (reads saved held-out evaluations)
# -----------------------------------------------------------------------------
def _load_eval_json(name: str) -> dict[str, Any] | None:
    import json
    try:
        return json.loads((_MODELS2_DIR / name).read_text())
    except Exception:
        return None


def _metrics_table(metric_block: dict[str, Any]) -> pd.DataFrame:
    labels = ["PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"]
    rows = []
    for lbl in labels:
        pc = (metric_block.get("per_class") or {}).get(lbl, {})
        rows.append({
            "Class": ml_meta(lbl)["title"],
            "Precision": f"{pc.get('precision', float('nan')):.3f}" if pc else "—",
            "Recall": f"{pc.get('recall', float('nan')):.3f}" if pc else "—",
            "F1": f"{pc.get('f1-score', float('nan')):.3f}" if pc else "—",
            "Support": int(pc.get("support", 0)) if pc else 0,
        })
    return pd.DataFrame(rows)


def _confusion_df(metric_block: dict[str, Any]) -> pd.DataFrame | None:
    cm = metric_block.get("confusion_matrix")
    if not cm:
        return None
    short = ["PRIORITY", "SUPPORT", "UNCLEAR"]
    return pd.DataFrame(cm, index=[f"true {s}" for s in short], columns=[f"pred {s}" for s in short])


def render_performance() -> None:
    with st.container(border=True):
        card_header("Evaluation", "Model performance & comparison")
        st.markdown(
            '<div class="muted">Metrics are from saved, leakage-free held-out evaluations '
            "(grouped train/val/test split by scenario family) on synthetic data. They are "
            "not evidence of real-world performance.</div>",
            unsafe_allow_html=True,
        )

    fusion = _load_eval_json("evaluation_fusion.json")
    if not fusion:
        st.warning("No saved evaluation found (models2.0/evaluation_fusion.json). Run the evaluation script to populate this page.")
        return

    models = [
        ("Logistic Regression", fusion.get("logistic_regression", {})),
        ("Semantic Evidence Voting", fusion.get("semantic_evidence_voting", {})),
        ("Evidence Fusion", fusion.get("evidence_fusion", {})),
    ]
    with st.container(border=True):
        card_header("Headline", f"Three-way comparison · {fusion.get('test_rows', '?')} held-out test rows")
        cols = st.columns(3)
        for col, (name, m) in zip(cols, models):
            with col:
                st.markdown(f"**{name}**")
                st.metric("Macro F1", f"{m.get('macro_f1', float('nan')):.3f}")
                st.metric("Accuracy", f"{m.get('accuracy', float('nan')):.3f}")
                st.metric("PRIORITY recall", f"{m.get('priority_recall', float('nan')):.3f}")
        best = max(models, key=lambda kv: kv[1].get("macro_f1", 0.0))[0]
        st.caption(
            f"Fusion weight (LogReg share) tuned on validation: {fusion.get('fusion_weight_logreg_share', '?')}. "
            f"Highest macro F1: **{best}** — but the Fusion/LogReg gap here is within one test example, "
            "so Fusion is best read as ≈ tied with LogReg, not a validated improvement."
        )

    for name, m in models:
        with st.container(border=True):
            card_header("Per class", name)
            st.dataframe(_metrics_table(m), use_container_width=True, hide_index=True)
            cm = _confusion_df(m)
            if cm is not None:
                st.markdown('<div class="eyebrow" style="margin-top:0.5rem;">Confusion matrix (rows = true)</div>', unsafe_allow_html=True)
                st.dataframe(cm, use_container_width=True)

    note = fusion.get("note")
    if note:
        st.caption(note)


# -----------------------------------------------------------------------------
# Page: Emergency Assistance & Nearby Resources (Geoapify + optional Google)
# -----------------------------------------------------------------------------
RESOURCE_LABELS = {
    "police": "🚓 Police stations",
    "hospital": "🏥 Hospitals",
    "clinic": "🩺 Clinics",
    "pharmacy": "💊 Pharmacies",
    "ambulance": "🚑 Emergency / ambulance (hospital ER)",
}
# Marker colours (functional, distinct) — RGB for pydeck + hex for the legend.
RESOURCE_COLORS = {
    "police": ((60, 107, 109), "#3c6b6d"),
    "hospital": ((176, 95, 69), "#b05f45"),
    "clinic": ((169, 132, 63), "#a9843f"),
    "pharmacy": ((126, 155, 132), "#7e9b84"),
    "ambulance": ((192, 74, 58), "#c04a3a"),
}
ORIGIN_COLOR = (58, 56, 47)  # soft charcoal for the searched location


def _render_map(data: dict, chosen: list[str]) -> None:
    """Interactive pydeck map: colour-coded, hoverable markers + searched origin."""
    try:
        import pydeck as pdk
    except Exception:
        st.caption("Map unavailable (pydeck not installed) — see the cards below.")
        return

    coords = data["coordinates"]
    points = []
    for rtype in chosen:
        rgb = RESOURCE_COLORS.get(rtype, ((100, 100, 100), "#666"))[0]
        for r in data["resources"].get(rtype, []):
            if r["lat"] is None or r["lon"] is None:
                continue
            rating = r.get("rating")
            points.append({
                "lat": r["lat"], "lon": r["lon"],
                "name": r["name"],
                "kind": RESOURCE_LABELS.get(rtype, rtype),
                "distance": "—" if r["distance_km"] is None else f"{r['distance_km']} km",
                "address": r["address"] or "Not available",
                "phone": r["phone"] or "Not available",
                "rating": f"{rating}★ ({r.get('reviews_count') or 0})" if rating else "Not available",
                "color": list(rgb),
            })
    points.append({
        "lat": coords["lat"], "lon": coords["lon"],
        "name": "📍 Searched location", "kind": data["query_location"],
        "distance": "—", "address": coords.get("formatted") or "", "phone": "—",
        "rating": "—", "color": list(ORIGIN_COLOR),
    })
    df = pd.DataFrame(points)
    layer = pdk.Layer(
        "ScatterplotLayer", df, get_position="[lon, lat]", get_fill_color="color",
        get_radius=60, radius_min_pixels=7, radius_max_pixels=22, pickable=True,
        stroked=True, get_line_color=[255, 253, 248], line_width_min_pixels=1,
    )
    view = pdk.ViewState(latitude=coords["lat"], longitude=coords["lon"], zoom=12)
    tooltip = {
        "html": "<b>{name}</b><br/>{kind}<br/>📏 {distance}<br/>📍 {address}<br/>📞 {phone}<br/>⭐ {rating}",
        "style": {"backgroundColor": "#3a382f", "color": "#fffdf8", "fontSize": "12px"},
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip,
                             map_style="light"), use_container_width=True)
    # Legend
    legend = "".join(
        f'<span class="chip" style="border-color:{RESOURCE_COLORS[t][1]};">'
        f'<span style="color:{RESOURCE_COLORS[t][1]};font-weight:700;">●</span> {RESOURCE_LABELS[t]}</span>'
        for t in chosen if t in RESOURCE_COLORS
    )
    st.markdown(f'<div style="margin-top:0.4rem;">{legend}</div>', unsafe_allow_html=True)


def _stars(rating: float | None) -> str:
    if not rating:
        return ""
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def _resource_card(r: dict) -> None:
    """One scannable card: name, type, distance, address, phone, rating/reviews, links."""
    with st.container(border=True):
        dist = "" if r["distance_km"] is None else f'<span class="chip">📏 {r["distance_km"]} km</span>'
        hexcol = RESOURCE_COLORS.get(r["type"], ((0, 0, 0), "#666"))[1]
        type_chip = f'<span class="chip" style="border-color:{hexcol};color:{hexcol};font-weight:700;">{RESOURCE_LABELS.get(r["type"], r["type"])}</span>'
        st.markdown(
            f'<div class="card-title">{r["name"]}</div><div>{type_chip}{dist}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="kv"><b>Address:</b> {r["address"] or "Not available"}</div>', unsafe_allow_html=True)

        phone = r["phone"]
        if phone:
            tel = re.sub(r"[^0-9+]", "", phone)
            st.markdown(f'<div class="kv"><b>Phone:</b> <a href="tel:{tel}">{phone}</a></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kv"><b>Phone:</b> Not available</div>', unsafe_allow_html=True)

        rating = r.get("rating")
        if rating:
            st.markdown(
                f'<div class="kv"><b>Rating:</b> <span style="color:#a9843f;">{_stars(rating)}</span> '
                f'{rating} ({r.get("reviews_count") or 0} reviews)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="kv"><b>Rating / reviews:</b> Not available</div>', unsafe_allow_html=True)

        links = []
        if r.get("directions_url"):
            links.append(f'<a href="{r["directions_url"]}" target="_blank">🧭 Directions</a>')
        if r.get("google_url"):
            links.append(f'<a href="{r["google_url"]}" target="_blank">🗺️ Google Maps listing</a>')
        if links:
            st.markdown('<div class="kv">' + " · ".join(links) + "</div>", unsafe_allow_html=True)

        reviews = r.get("reviews") or []
        if reviews:
            with st.expander(f"Read {len(reviews)} Google review(s)"):
                for rv in reviews:
                    head = f"**{rv.get('author') or 'Anonymous'}**"
                    if rv.get("rating"):
                        head += f" · {_stars(rv['rating'])} {rv['rating']}"
                    if rv.get("relative_time"):
                        head += f" · {rv['relative_time']}"
                    st.markdown(head)
                    if rv.get("text"):
                        st.caption(rv["text"])


def render_emergency() -> None:
    with st.container(border=True):
        card_header("Emergency", "Nearby emergency resources")
        st.markdown(
            '<div class="muted">Informational lookup only — these listings do not dispatch '
            "services and may be out of date. Always verify before acting.</div>",
            unsafe_allow_html=True,
        )

    if not get_geoapify_api_key():
        st.info(
            "Nearby-resources lookup is disabled. Add `GEOAPIFY_API_KEY` to Streamlit Secrets "
            "or the environment to enable it (free key at geoapify.com). Add `GOOGLE_PLACES_API_KEY` "
            "too for phone numbers, ratings and reviews.",
            icon="📍",
        )
        return

    default_location = ""
    if st.session_state.get("last_analysis"):
        default_location = str(st.session_state.last_analysis.get("location") or "")
        if default_location in ("Not identified", "Not identified."):
            default_location = ""

    with st.container(border=True):
        with st.form("emergency_resources_form"):
            loc = st.text_input(
                "Location (city, area, or address)",
                value=default_location,
                placeholder="e.g. Connaught Place, New Delhi",
            )
            fc1, fc2 = st.columns([1, 1.4])
            with fc1:
                radius_km = st.slider("Search radius (km)", 1, 25, 5)
            with fc2:
                chosen = st.multiselect(
                    "Resource types",
                    options=list(RESOURCE_LABELS.keys()),
                    default=list(emsvc.DEFAULT_RESOURCE_TYPES),
                    format_func=lambda k: RESOURCE_LABELS.get(k, k),
                )
            submitted = st.form_submit_button("🔎 Find nearby resources", type="primary", use_container_width=True)

    if not submitted:
        return
    if not loc.strip():
        st.warning("Enter a location to search.", icon="⚠️")
        return
    if not chosen:
        st.warning("Select at least one resource type.", icon="⚠️")
        return

    try:
        with st.spinner("Searching nearby resources…"):
            data = cached_emergency_search(
                loc.strip(), tuple(chosen), int(radius_km) * 1000,
                get_geoapify_api_key(), get_google_places_api_key(),
            )
    except emsvc.MissingApiKeyError:
        st.error("Geoapify rejected the API key. Check GEOAPIFY_API_KEY.", icon="🔑")
        return
    except emsvc.LocationNotFoundError as exc:
        st.error(str(exc), icon="🗺️")
        return
    except emsvc.ApiRequestError as exc:
        st.error(f"Lookup failed: {exc}", icon="📡")
        return
    except emsvc.EmergencyServiceError as exc:
        st.error(f"Lookup error: {exc}", icon="⚠️")
        return

    coords = data["coordinates"]
    st.caption(
        f"Matched **{coords['formatted'] or loc.strip()}** "
        f"({coords['lat']:.4f}, {coords['lon']:.4f}) · radius {radius_km} km"
    )
    if data["errors"].get("_google"):
        st.warning(f"Ratings/reviews unavailable: {data['errors']['_google']}", icon="⭐")
    elif not get_google_places_api_key():
        st.caption("Add GOOGLE_PLACES_API_KEY for phone numbers, ratings and reviews. Showing map/listing data only.")

    total = sum(len(v) for v in data["resources"].values())
    if total == 0:
        st.info("No matching resources found within this radius. Try a wider radius or a different location.", icon="🔍")
        return

    with st.container(border=True):
        card_header("Map", "Resource map")
        _render_map(data, chosen)

    for rtype in chosen:
        items = data["resources"].get(rtype, [])
        err = data["errors"].get(rtype)
        st.markdown(f"#### {RESOURCE_LABELS.get(rtype, rtype)}")
        if err:
            st.caption(f"⚠️ {err}")
        elif not items:
            st.caption("No results in range.")
        else:
            card_cols = st.columns(2)
            for i, r in enumerate(items):
                with card_cols[i % 2]:
                    _resource_card(r)

    st.caption(
        "Listings are OpenStreetMap-derived via Geoapify; phone, ratings and reviews (where shown) "
        "come from Google Places and are matched by name and location — they may be incomplete or "
        "imperfectly matched. Fields without a confident match read 'Not available'. Informational "
        "only — this tool never contacts or dispatches emergency services."
    )


# -----------------------------------------------------------------------------
# Page dispatch — other pages render then stop; triage is the fall-through
# -----------------------------------------------------------------------------
if nav == PAGE_PERFORMANCE:
    render_performance()
    st.stop()
if nav == PAGE_EMERGENCY:
    render_emergency()
    st.stop()


# --- Triage Dashboard controls (sidebar) -----------------------------------
sample_choice = st.sidebar.selectbox(
    "Load an example transcript",
    ["Custom transcript", *SAMPLE_TRANSCRIPTS.keys()],
    key="sample_choice",
)
if st.session_state.get("_last_sample_choice") != sample_choice:
    st.session_state.transcript_input = SAMPLE_TRANSCRIPTS.get(sample_choice, "")
    st.session_state._last_sample_choice = sample_choice

_sev_ok = sev_available()
_fusion_ok = fusion_available()
_method_options = [METHOD_LOGREG]
if _sev_ok:
    _method_options.append(METHOD_SEV)
if _fusion_ok:
    _method_options.append(METHOD_FUSION)
triage_method = st.sidebar.radio(
    "Triage model",
    _method_options,
    index=0,
    help=(
        "Logistic Regression (word+char TF-IDF) is the default. Semantic Evidence Voting "
        "is an experimental multilingual-embedding method. Evidence Fusion blends the two "
        "class-score vectors with a validation-tuned weight and flags disagreement / "
        "low-confidence cases for human review."
    ),
)
if not _sev_ok:
    st.sidebar.caption("Semantic Evidence Voting / Fusion unavailable (needs sentence-transformers + built index).")
st.sidebar.caption(
    "Held-out macro F1 — LogReg 0.772 · SEV 0.615 · Fusion 0.776. See the Model Performance "
    "page for the full comparison. No method is claimed as a validated improvement."
)


# -----------------------------------------------------------------------------
# 1. Incoming complaint (audio → Sarvam → transcript, or typed)
# -----------------------------------------------------------------------------
def clear_transcript() -> None:
    """Reset input + results. Runs as a callback (before the widget re-instantiates)."""
    st.session_state.transcript_input = ""
    st.session_state.last_analysis = None
    st.session_state.last_prediction = None
    st.session_state.last_transcript = ""
    st.session_state.last_case_id = None
    st.session_state.last_detected_language = None


with st.container(border=True):
    card_header("Step 1", "Incoming complaint")
    st.markdown(
        '<div class="muted">Upload a short audio clip (transcribed via Sarvam AI) or type a transcript directly.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    audio_col, mode_col = st.columns([2, 1])
    with audio_col:
        audio_file = st.file_uploader(
            "Audio recording (short clips, ~30s, for synchronous transcription)",
            type=AUDIO_TYPES,
            help="Use a fictional or specifically authorised clip. Audio is sent to Sarvam AI only when you request transcription.",
        )
    with mode_col:
        sarvam_mode = st.selectbox(
            "Transcription style",
            ["transcribe", "codemix", "translit"],
            index=0,
            help="Try transcribe first; compare codemix/translit on fictional Hinglish examples if needed.",
        )

    consent = st.checkbox(
        "I confirm this is fictional or authorised audio and may be sent to Sarvam AI for transcription.",
        key="audio_consent",
    )

    t_col, _ = st.columns([1.3, 2.4])
    with t_col:
        transcribe_clicked = st.button(
            "🎙️ Transcribe with Sarvam",
            use_container_width=True,
            disabled=audio_file is None,
        )

    if transcribe_clicked:
        if not consent:
            st.error("Please confirm the audio is fictional or authorised before sending it to Sarvam.")
        elif not get_sarvam_api_key():
            st.error("Sarvam API key is missing. Configure SARVAM_API_KEY in Streamlit Secrets or the environment.")
        elif audio_file is not None:
            try:
                with st.spinner("Transcribing audio with Sarvam…"):
                    transcript_result, detected_language = transcribe_audio(
                        audio_file.getvalue(), audio_file.name, sarvam_mode
                    )
                st.session_state.transcript_input = transcript_result
                st.session_state.last_detected_language = detected_language
                st.success("Transcription complete. Review and correct the text below before analysis.")
                st.rerun()
            except Exception as exc:
                st.error(f"Transcription failed: {type(exc).__name__}: {exc}")
                st.info("You can still enter or paste a transcript manually below.")

    if st.session_state.get("last_detected_language"):
        st.caption(f"Sarvam detected language: {st.session_state.last_detected_language}")

    transcript = st.text_area(
        "Call transcript — review before analysis",
        key="transcript_input",
        height=140,
        placeholder="Enter a fictional transcript in English, Hindi, or Romanized Hinglish…",
        help="Original text is kept for operator review. ML classifies routing; keyword rules add transparent evidence.",
        label_visibility="collapsed",
    )
    col_analyse, col_clear, _ = st.columns([1.1, 1, 2.4])
    with col_analyse:
        analyze_clicked = st.button("🔍 Analyse transcript", type="primary", use_container_width=True)
    with col_clear:
        st.button("Clear", use_container_width=True, on_click=clear_transcript)

if analyze_clicked:
    if not transcript.strip():
        st.warning("Enter a fictional transcript before analysing.")
    else:
        result = analyze_transcript(transcript)
        with st.spinner(f"Analysing with {triage_method}…"):
            prediction = predict_with_method(transcript.strip(), triage_method)
        case_id = f"HL-{st.session_state.case_counter:04d}"
        st.session_state.case_counter += 1
        created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        st.session_state.cases.append(
            make_case(case_id, transcript.strip(), created_at, method=triage_method)
        )
        st.session_state.last_analysis = result
        st.session_state.last_prediction = prediction
        st.session_state.last_transcript = transcript.strip()
        st.session_state.last_case_id = case_id


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
def render_score_bars(class_scores: dict[str, float]) -> None:
    rows_html = []
    for label in ["PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"]:
        if label not in class_scores:
            continue
        meta = ml_meta(label)
        pct = class_scores[label] * 100
        rows_html.append(
            f'<div class="score-row"><div class="score-label">'
            f'<span>{meta["emoji"]} {meta["title"]}</span><span>{pct:.0f}%</span></div>'
            f'<div class="score-track"><div class="score-fill {meta["fill"]}" '
            f'style="width:{pct:.0f}%"></div></div></div>'
        )
    st.markdown("".join(rows_html), unsafe_allow_html=True)


if st.session_state.last_analysis is not None:
    result = st.session_state.last_analysis
    prediction = st.session_state.last_prediction
    case_id = st.session_state.last_case_id
    current_case = next((c for c in st.session_state.cases if c["case_id"] == case_id), None)
    review_status = current_case.get("review_status", "Pending review") if current_case else "Pending review"

    method_used = prediction.get("method", METHOD_LOGREG) if prediction else METHOD_LOGREG
    score_label = prediction.get("score_label", "model score") if prediction else "model score"
    with st.container(border=True):
        card_header("Step 2", f"Triage result · case {case_id}")
        st.markdown(
            f'<div class="muted">Model: <b>{method_used}</b> · routing label with rule-based evidence · '
            f'uncalibrated {score_label}, not a validated risk score</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        head_l, head_r = st.columns([1.5, 1])
        with head_l:
            if prediction and "predicted_label" in prediction:
                meta = ml_meta(prediction["predicted_label"])
                conf = prediction["confidence"] * 100
                st.markdown(
                    f'<span class="triage-badge {meta["badge"]}">{meta["emoji"]} {meta["title"]}</span>'
                    f'<span class="muted" style="margin-left:0.6rem;">{score_label} {conf:.0f}%</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="muted" style="margin-top:0.45rem;">{prediction.get("description","")}</div>', unsafe_allow_html=True)
                if result["urgency"] == "Critical" and prediction["predicted_label"] != "PRIORITY_REVIEW":
                    st.error(
                        "⚠️ Keyword rules flagged possible CRITICAL indicators while the ML label is "
                        f"'{meta['title']}'. Treat as priority and verify the transcript immediately."
                    )
            else:
                st.error("Selected model unavailable — showing keyword-rule evidence only.")
        with head_r:
            st.markdown(
                '<div class="eyebrow">Operator review</div>'
                f'<span class="triage-badge b-neutral badge-sm">📋 {review_status}</span>',
                unsafe_allow_html=True,
            )

        # Evidence Fusion: show component agreement and the human-review flag.
        if prediction and prediction.get("fusion"):
            fz = prediction["fusion"]
            st.write("")
            lr_m, sev_m = ml_meta(fz["logreg_label"]), ml_meta(fz["sev_label"])
            agree_badge = ("b-unclear", "✅ Models agree") if fz["models_agree"] else ("b-support", "⚠️ Models disagree")
            st.markdown(
                '<div class="eyebrow">Component models</div>'
                f'<div class="kv"><span class="chip">Logistic Regression → {lr_m["emoji"]} {lr_m["title"]}</span>'
                f'<span class="chip">SEV → {sev_m["emoji"]} {sev_m["title"]}</span>'
                f'<span class="triage-badge {agree_badge[0]} badge-sm">{agree_badge[1]}</span></div>',
                unsafe_allow_html=True,
            )
            if fz["review_flag"]:
                st.warning("🚩 Flagged for human review: " + " ".join(fz["review_reasons"]))
            else:
                st.caption("No fusion review flag raised (models agree and the fused score is confident). Human verification is still required.")

        if prediction and prediction.get("scores"):
            st.write("")
            if method_used == METHOD_SEV:
                bars_label = "Vote proportions"
            elif method_used == METHOD_FUSION:
                bars_label = "Fused class scores"
            else:
                bars_label = "Model class scores (uncalibrated)"
            st.markdown(f'<div class="eyebrow">{bars_label}</div>', unsafe_allow_html=True)
            render_score_bars(prediction["scores"])
            st.caption(prediction.get("score_note", "Uncalibrated output; not a probability of real-world correctness."))

        # Semantic Evidence Voting / Fusion: show the nearest training examples transparently.
        if prediction and prediction.get("supporting_examples"):
            st.write("")
            st.markdown('<div class="eyebrow">Supporting training examples (nearest neighbours)</div>', unsafe_allow_html=True)
            for ex in prediction["supporting_examples"]:
                ex_meta = ml_meta(ex["label"])
                snippet = ex["transcript"] if len(ex["transcript"]) <= 110 else ex["transcript"][:107] + "…"
                st.markdown(
                    f'<div class="kv"><span class="triage-badge {ex_meta["badge"]} badge-sm">{ex_meta["emoji"]} {ex_meta["title"]}</span>'
                    f'<span class="muted" style="margin-left:0.5rem;">similarity {ex["similarity"]*100:.0f}%</span><br>'
                    f'<span style="font-size:0.88rem;">“{snippet}”</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption("These are the labelled training transcripts most similar to this input. Similarity is not proof of a shared caller or incident; a human must verify.")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            card_header("Evidence", "🔎 Detected cues (rules)")
            st.markdown(f'<div class="kv"><b>Possible concern:</b> {result["threat"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kv"><b>Suggested queue:</b> {result["suggested_queue"]}</div>', unsafe_allow_html=True)
            loc = ", ".join(result["location_clues"]) if result["location_clues"] else "Not identified"
            st.markdown(f'<div class="kv"><b>Location clue(s):</b> {loc}</div>', unsafe_allow_html=True)
            st.caption("A location clue is unverified text, not a confirmed current location or address.")
            st.markdown('<div class="eyebrow" style="margin-top:0.4rem;">Matched text indicators</div>', unsafe_allow_html=True)
            if result["evidence"]:
                st.markdown("".join(f'<span class="chip chip-sage">{p}</span>' for p in result["evidence"]), unsafe_allow_html=True)
            else:
                st.markdown('<span class="muted">No configured indicator phrase matched.</span>', unsafe_allow_html=True)
            st.caption("A matched phrase is a review cue, not proof that an event occurred.")
    with c2:
        with st.container(border=True):
            card_header("Verify", "✅ Information to confirm")
            for item in result["missing"]:
                st.markdown(f'<div class="verify-item">{item}</div>', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow" style="margin-top:0.4rem;">Suggested operator next step</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kv">{result["next_step"]}</div>', unsafe_allow_html=True)
            if prediction and "human_review_note" in prediction:
                st.caption(prediction["human_review_note"])

    with st.container(border=True):
        card_header("Step 3", "Operator review")
        st.markdown(
            '<div class="muted">Distinguish model/rule output above from operator-confirmed findings below. '
            'Saving does not contact services or dispatch help.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if current_case is not None:
            status_key = f"review_status_{case_id}"
            notes_key = f"review_notes_{case_id}"
            st.session_state.setdefault(status_key, current_case.get("review_status", "Pending review"))
            st.session_state.setdefault(notes_key, current_case.get("operator_notes", ""))
            rc1, rc2 = st.columns([1, 1.4])
            with rc1:
                status = st.selectbox(
                    "Operator-confirmed outcome",
                    [
                        "Pending review",
                        "Reviewed — more information needed",
                        "Reviewed — follow-up required",
                        "Reviewed — no further action recorded",
                        "Escalated according to local protocol",
                    ],
                    key=status_key,
                )
            with rc2:
                notes = st.text_area(
                    "Operator notes (fictional demo only)",
                    key=notes_key,
                    placeholder="Record what was verified, without adding real personal data.",
                    height=90,
                )
            if st.button("💾 Save operator review", key=f"save_review_{case_id}"):
                for case in st.session_state.cases:
                    if case["case_id"] == case_id:
                        case["review_status"] = status
                        case["operator_notes"] = notes.strip()
                        break
                st.success("Operator review saved to this browser session.")

    with st.container(border=True):
        card_header("Step 4", "Related transcripts")
        matches, best = find_related_complaints(
            st.session_state.last_transcript, st.session_state.cases, exclude_case_id=case_id
        )
        if matches.empty:
            st.info(
                f"No strong matches found (highest lexical overlap was {best * 100:.0f}%, "
                f"below the {RELATED_THRESHOLD * 100:.0f}% threshold for a related candidate).",
                icon="🔗",
            )
        else:
            st.dataframe(matches, use_container_width=True, hide_index=True)
        st.caption(
            "Lexical overlap is a rough word/character comparison, not proof of a shared caller, "
            "person, or incident. Any potential link must be checked by an authorised human."
        )


# -----------------------------------------------------------------------------
# 5. Session case history
# -----------------------------------------------------------------------------
with st.container(border=True):
    card_header("Step 5", "Session case history")
    st.markdown('<div class="muted">Kept in Streamlit session state only — not a permanent database.</div>', unsafe_allow_html=True)
    st.write("")
    if st.session_state.cases:
        history = pd.DataFrame(st.session_state.cases)
        display_columns = [
            "case_id", "model", "ml_label", "ml_score", "rule_cue", "threat", "location",
            "suggested_queue", "review_status", "created_at", "source",
        ]
        export_columns = [
            "case_id", "transcript", "model", "ml_label", "ml_score", "rule_cue", "threat", "location",
            "suggested_queue", "created_at", "review_status", "operator_notes", "source",
        ]
        for col in set(display_columns + export_columns):
            if col not in history.columns:
                history[col] = ""
        st.dataframe(history[display_columns], use_container_width=True, hide_index=True)

        csv_bytes = history[export_columns].to_csv(index=False).encode("utf-8-sig")
        hc1, hc2 = st.columns([1, 1])
        with hc1:
            st.download_button(
                "⬇️ Download session history (CSV)", data=csv_bytes,
                file_name="helpline_triage_session_history.csv", mime="text/csv",
                use_container_width=True,
            )
        with hc2:
            if st.button("↺ Reset to built-in examples", use_container_width=True):
                st.session_state.cases = seed_cases()
                st.session_state.last_analysis = None
                st.session_state.last_prediction = None
                st.session_state.last_transcript = ""
                st.session_state.last_case_id = None
                st.session_state.last_detected_language = None
                st.session_state.case_counter = 1
                st.rerun()
    else:
        st.info("No cases recorded in this session yet.")

st.caption(
    "Prototype v2.1 — trained ML triage (models2.0: word+char TF-IDF → Logistic Regression) with a "
    "transparent keyword-rule evidence layer, optional Sarvam AI speech-to-text, basic location clues, "
    "thresholded lexical similarity, and temporary session history. Scores are uncalibrated model "
    "outputs on synthetic data. BHASHINI, semantic embeddings, calibrated probabilities, and "
    "real-world validation are future work."
)
