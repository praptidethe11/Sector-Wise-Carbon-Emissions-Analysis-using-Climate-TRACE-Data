import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import encode_input

MODELS_PRESENT = os.path.exists("models/encoders.pkl")


@pytest.mark.skipif(not MODELS_PRESENT, reason="models/encoders.pkl not present yet")
def test_encode_input_shape():
    import pickle
    with open("models/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    sample_sector = encoders["sector"].classes_[0]
    sample_subsector = encoders["subsector"].classes_[0]
    sample_gas = encoders["gas"].classes_[0]
    sample_state = encoders["state"].classes_[0]

    row = encode_input(encoders, sample_sector, sample_subsector, sample_gas, sample_state, 2024, 5)

    assert row.shape == (1, 6)
    assert list(row.columns) == [
        "sector_enc", "subsector_enc", "gas_enc", "state_enc", "year", "asset_count"
    ]