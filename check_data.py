"""
Pre-check script: validates a bucketized 30-day-window input file BEFORE
running it through any of the embedding-generation notebooks (cpu/*.ipynb,
gpu/*.ipynb). Catches the exact failure modes discussed while designing
this pipeline:

  - a `_30_days` cell that's a hard NULL (not a list at all, not even an
    empty one) -- this crashes every notebook immediately with
    "TypeError: 'NoneType' object is not iterable"
  - a `_30_days` list whose length doesn't match `n_hits_in_window`, or
    doesn't match another column's length for the SAME row -- this
    crashes with a numpy broadcast error deep inside the model code
  - missing required columns entirely
  - duplicate (post_evar3, event_date) rows (warning, not a hard error)

Does NOT flag individual `null` VALUES inside an otherwise correctly-sized
list, or a list that's entirely `null` (e.g. [null, null, null]) -- both
are explicitly VALID and expected (see FINDINGS.md / project discussion on
missing-value handling). Only checks structure (are lists present, are
lengths consistent), never the actual values inside them.

Run against the bundled default input:
    python check_data.py

Run against your own real data file:
    python check_data.py --input path/to/your_file.parquet

Exits with code 0 if clean, 1 if any error was found (safe to use as a
gate in a script: `python check_data.py && python -m jupyter ...`).
"""
import argparse
import sys

import pandas as pd

STRUCTURAL_COLUMNS = ["post_evar3", "event_date", "n_hits_in_window"]
BEHAVIOR_COLUMNS = [
    "channel",
    "device_type",
    "edition",
    "georegion",
    "page_hit_ref_type",
    "paywall",
    "content_source_hit",
    "content_type_hit",
    "section_hit",
    "subscriber_status",
    "page_id",
]
LIST_COLUMNS = [f"{col}_30_days" for col in BEHAVIOR_COLUMNS] + ["date_time_et_30_days"]
REQUIRED_COLUMNS = STRUCTURAL_COLUMNS + LIST_COLUMNS


def is_missing_cell(val) -> bool:
    """True if the cell itself is null (not a list at all) -- e.g. None,
    np.nan, pd.NA. A real (even empty) list/array returns False here."""
    if val is None:
        return True
    if not hasattr(val, "__len__"):
        # a bare scalar (e.g. float('nan')) has no __len__ -- treat as missing
        try:
            return bool(pd.isna(val))
        except (TypeError, ValueError):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/bucketized_30day_all_events.parquet")
    args = parser.parse_args()

    errors = []
    warnings = []

    print(f"Checking: {args.input}")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        print(f"FAIL: could not read file -- {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"  {len(df)} rows loaded")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"\nFAIL: missing required columns: {missing_cols}")
        print("Cannot continue further checks until these columns exist.")
        sys.exit(1)
    print(f"  all {len(REQUIRED_COLUMNS)} required columns present")

    dup_count = int(df.duplicated(subset=["post_evar3", "event_date"]).sum())
    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate (post_evar3, event_date) rows found")

    null_cell_rows = {col: [] for col in LIST_COLUMNS}
    length_mismatch_rows = []

    for idx, row in df.iterrows():
        n_hits = row["n_hits_in_window"]
        row_lengths = {}
        row_has_null_cell = False

        for col in LIST_COLUMNS:
            val = row[col]
            if is_missing_cell(val):
                null_cell_rows[col].append(idx)
                row_has_null_cell = True
                continue
            row_lengths[col] = len(val)

        if row_has_null_cell:
            continue  # already flagged as an error below; length check is meaningless here

        lengths_seen = set(row_lengths.values())
        if len(lengths_seen) > 1 or (lengths_seen and n_hits not in lengths_seen):
            length_mismatch_rows.append((idx, row["post_evar3"], row["event_date"], n_hits, dict(row_lengths)))

    for col, rows in null_cell_rows.items():
        if rows:
            errors.append(
                f"{col}: {len(rows)} row(s) have a NULL CELL (not even an empty list) -- "
                f"example row indices: {rows[:5]}"
            )

    if length_mismatch_rows:
        errors.append(
            f"{len(length_mismatch_rows)} row(s) have list lengths that don't match "
            f"n_hits_in_window or don't match each other"
        )
        print("\n  First 5 length-mismatch examples:")
        for idx, user, date, n_hits, lengths in length_mismatch_rows[:5]:
            print(f"    row {idx} (post_evar3={user}, event_date={date}): "
                  f"n_hits_in_window={n_hits}, lengths seen={lengths}")

    print()
    print("=" * 70)
    if warnings:
        print(f"{len(warnings)} WARNING(S) (not blocking):")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print(f"{len(errors)} ERROR(S) -- fix these before running the embedding notebooks:")
        for e in errors:
            print(f"  - {e}")
    if not errors and not warnings:
        print("All checks passed. Data is ready for the embedding notebooks.")
    elif not errors:
        print("No blocking errors -- only warnings above (informational).")
    print("=" * 70)

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
