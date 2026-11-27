"""
MP215 data cleaning and the seven required data-quality checks.

Run from the repository root:
    python src/mp215/data_cleaning.py

The raw CSV files are only read, never modified.
"""

from pathlib import Path
import json

import pandas as pd

# These paths match the required MP215 repository layout.
TRAIN_FILE = Path("data/raw/train.csv")
LATER_FILE = Path("data/raw/later.csv")
HOLDOUT_FILE = Path("data/raw/holdout_features.csv")
CLEAN_FILE = Path("data/processed/train_clean.csv")
REPORT_FILE = Path("reports/data_quality_report.json")

TARGET = "pick_seconds"
STUCK_MIN_ROWS = 10
STUCK_MIN_DAYS = 7
TARGET_TYPO_LIMIT = 1000
EXPERIENCE_FACTOR = 30

# Abbreviations found in train.csv; later.csv contains the full spellings.
ZONE_ALIASES = {
    "chi.": "chilled",
    "bul.": "bulky",
    "amb.": "ambient",
}


def find_stuck_runs(data):
    """Find long runs where a numeric feature has exactly the same reading."""
    ordered = data.copy()
    ordered["_original_index"] = ordered.index
    ordered["_date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values(["_date", "record_id"]).reset_index(drop=True)

    runs = []

    for column in data.select_dtypes(include="number").columns:
        if column == TARGET:
            continue

        values = ordered[column].to_numpy()
        start = 0

        for end in range(1, len(ordered) + 1):
            changed = end == len(ordered) or values[end] != values[start]

            if changed:
                count = end - start
                first_date = ordered.loc[start, "_date"]
                last_date = ordered.loc[end - 1, "_date"]
                days = (last_date - first_date).days

                if count >= STUCK_MIN_ROWS and days >= STUCK_MIN_DAYS:
                    run = ordered.iloc[start:end]
                    runs.append({
                        "column": column,
                        "value": float(values[start]),
                        "rows": int(count),
                        "start_date": first_date.strftime("%Y-%m-%d"),
                        "end_date": last_date.strftime("%Y-%m-%d"),
                        "first_record_id": str(run.iloc[0]["record_id"]),
                        "last_record_id": str(run.iloc[-1]["record_id"]),
                        "indices": run["_original_index"].tolist(),
                    })

                start = end

    return runs


def main():
    # 1. Read the three source files.
    for file in [TRAIN_FILE, LATER_FILE, HOLDOUT_FILE]:
        if not file.exists():
            raise FileNotFoundError(
                f"Cannot find {file}. Run this script from the repository root."
            )

    train = pd.read_csv(TRAIN_FILE)
    later = pd.read_csv(LATER_FILE)
    holdout = pd.read_csv(HOLDOUT_FILE)

    # 2. Basic structure checks.
    required = {
        "record_id", "date", "items_in_order", "aisle_distance_m",
        "picker_exp_months", "shelf_height_m", "shift_hour",
        "order_weight_kg", "zone_type", "shift", "pick_seconds",
    }
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")

    # CHECK 1: Leaking column.
    # Exclude pick_seconds because it is the target and should be absent
    # from holdout_features.csv.
    input_columns = set(train.columns) - {TARGET}
    leak_candidates = sorted(input_columns - set(holdout.columns))

    # CHECK 2: Unit mix-up.
    evening = train["shift"].eq("Evening")
    evening_experience = train.loc[evening, "picker_exp_months"]
    other_experience = train.loc[~evening, "picker_exp_months"]

    is_multiple_of_30 = (
        evening_experience.mod(EXPERIENCE_FACTOR).eq(0).all()
    )
    unit_mixup = (
        is_multiple_of_30
        and evening_experience.max() > 60
        and other_experience.max() <= 60
        and later.loc[later["shift"].eq("Evening"),
                      "picker_exp_months"].max() <= 60
    )

    # CHECK 3: Sentinel values.
    sentinel_numbers = {-1, -999, -9999, 999, 9999}
    sentinels = {}
    for column in train.select_dtypes(include="number").columns:
        matches = {
            str(number): int((train[column] == number).sum())
            for number in sentinel_numbers
            if (train[column] == number).any()
        }
        if matches:
            sentinels[column] = matches

    # CHECK 4: Exact duplicate rows.
    duplicate_count = int(train.duplicated().sum())

    # CHECK 5: Stuck sensor.
    stuck_runs = find_stuck_runs(train)
    stuck_report = [
        {key: value for key, value in run.items() if key != "indices"}
        for run in stuck_runs
    ]

    # CHECK 6: Target typos (values are 10 times the normal range).
    typo_mask = train[TARGET] >= TARGET_TYPO_LIMIT
    typo_rows = train.loc[typo_mask, ["record_id", "date", TARGET]].copy()
    typo_rows["corrected_value"] = typo_rows[TARGET] / 10
    typo_report = [
        {
            "record_id": str(row["record_id"]),
            "date": str(row["date"]),
            "original_value": float(row[TARGET]),
            "corrected_value": float(row["corrected_value"]),
        }
        for _, row in typo_rows.iterrows()
    ]

    # CHECK 7: Spelling variants.
    canonical_zones = set(
        later["zone_type"].astype(str).str.strip().str.lower().unique()
    )
    normalized_zones = (
        train["zone_type"].astype(str).str.strip().str.lower()
        .replace(ZONE_ALIASES)
    )
    unknown_zones = sorted(set(normalized_zones.unique()) - canonical_zones)
    if unknown_zones:
        raise ValueError(f"Unrecognized zone_type values: {unknown_zones}")

    findings = {
        "leaking_column": {
            "present": bool(leak_candidates),
            "columns": leak_candidates,
        },
        "unit_mixup": {
            "present": bool(unit_mixup),
            "group_column": "shift",
            "group_value": "Evening",
            "column": "picker_exp_months",
            "source_unit": "days",
            "expected_unit": "months",
            "factor": EXPERIENCE_FACTOR,
            "affected_rows": int(evening.sum()),
        },
        "sentinel_value": {
            "present": bool(sentinels),
            "values_found": sentinels,
        },
        "duplicate_rows": {
            "present": duplicate_count > 0,
            "count": duplicate_count,
        },
        "stuck_sensor": {
            "present": bool(stuck_runs),
            "runs": stuck_report,
        },
        "target_typos": {
            "present": bool(typo_report),
            "count": len(typo_report),
            "rows": typo_report,
        },
        "spelling_variants": {
            "present": train["zone_type"].nunique() > len(canonical_zones),
            "column": "zone_type",
            "true_levels": len(canonical_zones),
            "raw_levels": train["zone_type"].value_counts().to_dict(),
            "clean_levels": sorted(canonical_zones),
        },
    }

    # 3. Apply the confirmed corrections to a COPY of train.csv.
    clean = train.copy()

    # Evening picker experience: days -> months, using the dataset's factor 30.
    if unit_mixup:
        clean.loc[evening, "picker_exp_months"] = (
            clean.loc[evening, "picker_exp_months"] / EXPERIENCE_FACTOR
        )

    # Replace only the detected stuck-sensor run, then use the unaffected
    # readings' median for imputation.
    stuck_indices_by_column = {}
    for run in stuck_runs:
        stuck_indices_by_column.setdefault(run["column"], []).extend(run["indices"])

    for column, indices in stuck_indices_by_column.items():
        clean.loc[indices, column] = pd.NA
        clean[column] = clean[column].fillna(clean[column].median())

    # Remove the extra zero from the nine target typos.
    clean.loc[typo_mask, TARGET] = clean.loc[typo_mask, TARGET] / 10

    # Standardize category labels.
    clean["zone_type"] = normalized_zones

    # No duplicates exist in this file; this also handles exact duplicates
    # should they appear in a later copy of the dataset.
    rows_before = len(clean)
    clean = clean.drop_duplicates().reset_index(drop=True)

    # Final checks.
    if clean.isna().any().any():
        raise ValueError("Missing values remain in train_clean.csv.")
    if set(clean["zone_type"].unique()) != canonical_zones:
        raise ValueError("zone_type does not match later.csv categories.")
    if clean.loc[clean["shift"].eq("Evening"),
                 "picker_exp_months"].max() > 60:
        raise ValueError("Evening picker experience was not converted correctly.")

    # 4. Save cleaned data and an audit report. Raw files remain untouched.
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    clean.to_csv(CLEAN_FILE, index=False)

    report = {
        "findings": findings,
        "cleaning_summary": {
            "rows_before": int(len(train)),
            "rows_after": int(len(clean)),
            "duplicate_rows_removed": int(rows_before - len(clean)),
            "evening_experience_rows_converted": int(evening.sum())
                if unit_mixup else 0,
            "stuck_sensor_runs_corrected": stuck_report,
            "target_typos_corrected": len(typo_report),
            "zone_type_levels_after_cleaning": sorted(clean["zone_type"].unique()),
            "pick_seconds_min_after_cleaning": float(clean[TARGET].min()),
            "pick_seconds_max_after_cleaning": float(clean[TARGET].max()),
        },
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nDATA QUALITY FINDINGS")
    for name, details in findings.items():
        print(f"{name}: {'PRESENT' if details['present'] else 'NOT PRESENT'}")

    print("\nCLEANING COMPLETE")
    print(f"Rows: {len(train)} -> {len(clean)}")
    print(f"Cleaned dataset: {CLEAN_FILE}")
    print(f"Quality report: {REPORT_FILE}")
    print("Original raw CSV files were not changed.")


if __name__ == "__main__":
    main()
