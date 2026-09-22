import streamlit as st
from src.data_loader import load_assets
from src.predictor import batch_predict
from src.charts import sector_shap_series, top_states_shap_series

st.set_page_config(page_title="India Carbon Emission Driver Explorer", layout="wide")

assets = load_assets()
model = assets["model"]
encoders = assets["encoders"]
options = assets["options"]
sector_map = assets["sector_map"]
sector_shap = assets["sector_shap"]
state_shap = assets["state_shap"]
asset_count_reference = assets["asset_count_reference"]

st.title("India Carbon Emission Driver Explorer")
st.caption("Extending a machine learning carbon driver framework from China to India")

tab1, tab2 = st.tabs(["Predict emissions", "Explore drivers"])

with tab1:
    st.subheader("Predict emissions across one or more combinations")

    col1, col2, col3 = st.columns(3)

    with col1:
        sectors = st.multiselect("Sector(s)", options["sectors"], default=[options["sectors"][0]])
        available_subsectors = sorted(set(
            sub for s in sectors for sub in sector_map.get(s, [])
        ))
        subsectors = st.multiselect("Subsector(s)", available_subsectors, default=available_subsectors[:1])
    with col2:
        gases = st.multiselect("Gas(es)", options["gases"], default=[options["gases"][0]])
        states = st.multiselect("State(s)", options["states"], default=[options["states"][0]])
    with col3:
        years = st.multiselect("Year(s)", options["years"], default=[options["years"][-1]])
        asset_count = st.number_input("Asset count (applied to all combinations)", min_value=1, value=5)

    total_combos = len(sectors) * len(subsectors) * len(gases) * len(states) * len(years)

    if total_combos > 500:
        st.warning(f"{total_combos} combinations selected — that's a lot. Consider narrowing your selection.")
    else:
        st.caption(f"{total_combos} combination(s) will be predicted")

    if st.button("Predict"):
        if not (sectors and subsectors and gases and states and years):
            st.error("Select at least one option in every category.")
        else:
            results = batch_predict(model, encoders, sectors, subsectors, gases, states, years, asset_count)

            if results.empty:
                st.warning("No valid combinations (check that selected subsectors belong to selected sectors).")
            elif len(results) == 1:
                st.metric("Predicted emissions", f"{results.iloc[0]['predicted_emissions']:,.2f}")
            else:
                results = results.sort_values("predicted_emissions", ascending=False).reset_index(drop=True)
                st.dataframe(results, use_container_width=True)

                label_cols = []
                for col in ["sector", "subsector", "gas", "state", "year"]:
                    if results[col].nunique() > 1:
                        label_cols.append(col)

                if not label_cols:
                    label_cols = ["state"]

                chart_labels = results[label_cols].astype(str).agg(" / ".join, axis=1)
                chart_data = results.set_index(chart_labels)["predicted_emissions"]
                st.bar_chart(chart_data)

            st.caption("Model: tuned Extra-Trees | R²=0.9967 on 2025 held-out test data")

with tab2:
    st.subheader("What's driving India's emissions?")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**By sector** (mean SHAP contribution)")
        st.bar_chart(sector_shap_series(sector_shap))
        st.caption("Power sector dominates, consistent with India's coal-dependent grid")

    with col2:
        st.write("**Top emitting states** (mean SHAP contribution)")
        st.bar_chart(top_states_shap_series(state_shap))
        st.caption("Chhattisgarh, Bihar, UP, and MP lead — India's coal/heavy-industry belt")

st.divider()
st.caption("Data: Climate TRACE (2021-2025) | Anchor study: Yu, Xia & Cao (2024), Scientific Reports")