"""Load and format student personas from the CSV survey data."""

import os
import random
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "cs222_classmate_agentbank.csv")


def format_persona(row: pd.Series) -> str:
    """Turn one survey row into a natural-language persona string."""
    parts: list[str] = []

    def add(text):
        if pd.notna(text) and str(text).strip() not in ("nan", ""):
            parts.append(str(text).strip())

    add(f"age {row.get('q1', '')}")
    add(f"{row.get('q2', '')}")
    add(f"from a {row.get('q3', '')} area")
    add(f"values {row.get('q5', '')}")
    add(f"{row.get('q6', '')} person")
    add(f"{row.get('q7', '')} in nature")
    add(f"enjoys {row.get('q8', '')}")
    add(f"interested in {row.get('q9', '')}")
    add(f"politically {row.get('q10', '')}")
    add(f"has {row.get('q11', '')} close friends")

    if pd.notna(row.get("q14")):
        add(f"Myers-Briggs type {row['q14']}")

    add(f"seeks {row.get('q15', '')}")
    add(f"concerned about {row.get('q16', '')}")
    add(f"approaches problems through {row.get('q21', '')}")
    add(f"makes decisions by {row.get('q25', '')}")

    if pd.notna(row.get("q22")):
        add(f"is {row['q22']}")
    add(f"values {row.get('q23', '')}")
    add(f"aspires to {row.get('q24', '')}")
    add(f"values {row.get('q26', '')} in others")

    if pd.notna(row.get("q29")) and pd.notna(row.get("q30")):
        add(f"identifies as {row['q29']}, speaks {row['q30']}")

    return " ".join(parts)


def load_players(n: int = 5) -> list[dict]:
    """Return *n* randomly selected players as [{name, persona}, ...]."""
    df = pd.read_csv(CSV_PATH)
    selected = df.sample(n=min(n, len(df)))
    return [
        {"name": row["sunet"], "persona": format_persona(row)}
        for _, row in selected.iterrows()
    ]
