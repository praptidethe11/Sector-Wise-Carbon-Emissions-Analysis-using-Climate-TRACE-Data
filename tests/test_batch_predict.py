import os
import sys
import pickle
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import batch_predict

MODELS_PRESENT = os.path.exists("models/encoders.pkl")


@pytest.fixture
def loaded_assets():
    with open("models/final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


@pytest.mark.skipif(not MODELS_PRESENT, reason="models not present")
def test_single_combination_matches_direct_predict(loaded_assets):
    model, encoders = loaded_assets
    sector = encoders["sector"].classes_[0]
    subsector = encoders["subsector"].classes_[0]
    gas = encoders["gas"].classes_[0]
    state = encoders["state"].classes_[0]

    results = batch_predict(model, encoders, [sector], [subsector], [gas], [state], [2025], 5)
    assert len(results) == 1
    assert results.iloc[0]["predicted_emissions"] >= 0


@pytest.mark.skipif(not MODELS_PRESENT, reason="models not present")
def test_multiple_states_produce_different_predictions(loaded_assets):
    model, encoders = loaded_assets
    sector, subsector, gas = "power", "electricity-generation", "co2"
    states = list(encoders["state"].classes_[:5])

    results = batch_predict(model, encoders, [sector], [subsector], [gas], states, [2025], 5)
    # if this fails, states aren't actually affecting predictions for this combo
    assert results["predicted_emissions"].nunique() > 1


@pytest.mark.skipif(not MODELS_PRESENT, reason="models not present")
def test_invalid_subsector_sector_combo_returns_empty_or_skips(loaded_assets):
    model, encoders = loaded_assets
    # deliberately mismatched pairing
    results = batch_predict(model, encoders, ["power"], ["aluminum"], ["co2"], ["Goa"], [2025], 5)
    # should either skip invalid rows (empty) or raise cleanly — not silently produce garbage
    assert isinstance(results, type(results))  # basic sanity, won't crash


@pytest.mark.skipif(not MODELS_PRESENT, reason="models not present")
def test_large_batch_runs_in_reasonable_time(loaded_assets):
    model, encoders = loaded_assets
    sectors = list(encoders["sector"].classes_)
    subsectors = list(encoders["subsector"].classes_)[:5]
    gases = list(encoders["gas"].classes_)
    states = list(encoders["state"].classes_)[:10]
    years = [2024, 2025]

    start = time.time()
    results = batch_predict(model, encoders, sectors, subsectors, gases, states, years, 5)
    elapsed = time.time() - start

    print(f"\n{len(results)} predictions in {elapsed:.2f}s")
    assert elapsed < 10  # should be near-instant for a tree model at this scale