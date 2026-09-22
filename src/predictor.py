import itertools
import pandas as pd


def encode_input(encoders, sector, subsector, gas, state, year, asset_count):
    row = pd.DataFrame([{
        "sector_enc": encoders["sector"].transform([sector])[0],
        "subsector_enc": encoders["subsector"].transform([subsector])[0],
        "gas_enc": encoders["gas"].transform([gas])[0],
        "state_enc": encoders["state"].transform([state])[0],
        "year": year,
        "asset_count": asset_count,
    }])
    return row


def predict_emissions(model, encoders, sector, subsector, gas, state, year, asset_count):
    row = encode_input(encoders, sector, subsector, gas, state, year, asset_count)
    prediction = model.predict(row)[0]
    return prediction


def batch_predict(model, encoders, sectors, subsectors, gases, states, years, asset_count):
    """Predict emissions for every combination of the selected values.

    Builds all valid combinations into a single dataframe and calls
    model.predict() once on the whole batch — far faster than predicting
    one row at a time, since per-call overhead dominates at n=1.
    """
    combos = list(itertools.product(sectors, subsectors, gases, states, years))

    valid_rows = []
    meta_rows = []
    for sector, subsector, gas, state, year in combos:
        try:
            encoded = {
                "sector_enc": encoders["sector"].transform([sector])[0],
                "subsector_enc": encoders["subsector"].transform([subsector])[0],
                "gas_enc": encoders["gas"].transform([gas])[0],
                "state_enc": encoders["state"].transform([state])[0],
                "year": year,
                "asset_count": asset_count,
            }
            valid_rows.append(encoded)
            meta_rows.append({"sector": sector, "subsector": subsector, "gas": gas, "state": state, "year": year})
        except ValueError:
            # subsector doesn't belong to this sector in the encoder — skip invalid combos
            continue

    if not valid_rows:
        return pd.DataFrame(columns=["sector", "subsector", "gas", "state", "year", "predicted_emissions"])

    X = pd.DataFrame(valid_rows)
    predictions = model.predict(X)  # single vectorized call, not one per row

    result = pd.DataFrame(meta_rows)
    result["predicted_emissions"] = predictions
    return result.sort_values("predicted_emissions", ascending=False).reset_index(drop=True)