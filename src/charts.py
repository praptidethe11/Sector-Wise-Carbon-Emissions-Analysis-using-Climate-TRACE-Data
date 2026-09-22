import pandas as pd


def sector_shap_series(sector_shap: pd.DataFrame) -> pd.Series:
    """Return a sector -> mean SHAP value series, ready for st.bar_chart."""
    return sector_shap.set_index("sector")["shap_value"]


def top_states_shap_series(state_shap: pd.DataFrame, n: int = 10) -> pd.Series:
    """Return the top-n states by mean SHAP value, ready for st.bar_chart."""
    top = state_shap.nlargest(n, "shap_value")
    return top.set_index("state")["shap_value"]