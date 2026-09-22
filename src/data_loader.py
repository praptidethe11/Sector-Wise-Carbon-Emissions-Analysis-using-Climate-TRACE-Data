import pickle
import json
import pandas as pd
import streamlit as st

MODELS_DIR = "models"


@st.cache_resource
def load_assets():
    with open(f"{MODELS_DIR}/final_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(f"{MODELS_DIR}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open(f"{MODELS_DIR}/options.json") as f:
        options = json.load(f)

    with open(f"{MODELS_DIR}/sector_subsector_map.json") as f:
        sector_map = json.load(f)

    sector_shap = pd.read_csv(f"{MODELS_DIR}/sector_shap_summary.csv")
    state_shap = pd.read_csv(f"{MODELS_DIR}/state_shap_summary.csv")
    asset_count_reference = pd.read_csv(f"{MODELS_DIR}/asset_count_reference.csv")

    return {
        "model": model,
        "encoders": encoders,
        "options": options,
        "sector_map": sector_map,
        "sector_shap": sector_shap,
        "state_shap": state_shap,
        "asset_count_reference": asset_count_reference,
    }