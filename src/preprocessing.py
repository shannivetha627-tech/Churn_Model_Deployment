
"""Shared data-cleaning functions for training and prediction."""

import numpy as np
import pandas as pd


def clean_frame(frame):
    """Convert charge values to numbers and normalize blank values."""
    result = frame.copy()

    if "customerID" in result.columns:
        result = result.drop(columns=["customerID"])

    if "TotalCharges" in result.columns:
        result["TotalCharges"] = pd.to_numeric(
            result["TotalCharges"].astype(str).str.strip(),
            errors="coerce",
        )

    return result.replace(r"^\s*$", np.nan, regex=True)
