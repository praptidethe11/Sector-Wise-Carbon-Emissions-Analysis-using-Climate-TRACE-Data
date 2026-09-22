# India Carbon Emission Driver Explorer

A Streamlit app extending a machine learning carbon-emission-driver framework
(originally applied to Chinese cities by Yu, Xia & Cao, 2024) to India, using
Climate TRACE data (2021-2025).

## What it does

- **Predict tab**: pick a sector, subsector, gas, state, year, and asset count
  to get a predicted emissions value from a tuned Extra-Trees model (R²=0.9967).
- **Explore tab**: view precomputed SHAP driver breakdowns showing which
  sectors and states dominate India's emissions.

## Setup

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Place the following exported files (from the training notebook) into `models/`
before running:
`final_model.pkl`, `encoders.pkl`, `options.json`, `sector_subsector_map.json`,
`sector_shap_summary.csv`, `state_shap_summary.csv`.

## Run

\`\`\`bash
streamlit run app.py
\`\`\`

## Data source

Climate TRACE (climatetrace.org) — India, Power/Transportation/Manufacturing
sectors, 2021-2025, state and subsector level.
