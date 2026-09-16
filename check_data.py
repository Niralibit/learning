"""
Validates a bucketized 30-day-window input file BEFORE running it through
any of the embedding-generation notebooks (cpu/*.ipynb, gpu/*.ipynb).
Catches the exact failure modes discussed while designing this pipeline:

  - a `30d_*` cell that's a hard NULL (not a list at all, not even an
    empty one) -- this crashes every notebook immediately
  - a `30d_*` list whose length doesn't match `30d_DATE_TIME_ET`'s length
    for the SAME row -- this crashes with a numpy broadcast error deep
    inside the model code
  - missing required columns entirely
  - duplicate (USER_ID, EVENT_DT) rows (warning, not a hard error)

Does NOT flag individual `null` VALUES inside an otherwise correctly-sized
list, or a list that's entirely `null` (e.g. [null, null, null]) -- both
are explicitly VALID and expected (see FINDINGS.md / project discussion on
missing-value handling). Only checks structure (are lists present, are
lengths consistent), never the actual values inside them.

Schema checked (matches the real data schema, not the old synthetic one):
    USER_ID, EVENT_DT, 30d_CHANNEL, 30d_CONTENT_SOURCE_HIT,
    30d_CONTENT_TYPE_HIT, 30d_DATE_TIME_ET, 30d_DEVICE_TYPE, 30d_EDITION,
    30d_GEO_REGION, 30d_PAGE_HIT_REF_TYPE, 30d_PAGE_ID, 30d_PAYWALL,
    30d_SECTION_HIT, 30d_SUBSCRIBER_STATUS
Columns intentionally NOT checked (not read by the embedding notebooks):
    30d_POST_EVAR3 (redundant -- user id repeated per click), CONTACT_METHOD,
    ACTION_SUGGESTED, EVENT_SENT_PERIOD, OUTCOME (all event-level, joined
    back downstream using USER_ID + EVENT_DT, not behavioral history).
There is no separate row-count column in this schema -- each row's
expected length is `len(30d_DATE_TIME_ET)` for that row, and every other
`30d_*` column must match it.

USED TWO WAYS:
1. Standalone, as a quick pre-flight check before ever opening Jupyter:
       python check_data.py
       python check_data.py --input s3://your-bucket-name/input/your_file.parquet
   Exits 0 if clean, 1 if anything needs fixing. Reads directly from S3
   via pandas + s3fs (storage_options={}) -- see requirements.txt for the
   AWS-credentials note.
2. Imported by every notebook (code/*.ipynb), which call `validate(df)`
   right after loading the data and stop immediately with a clear message
   if it returns any errors -- so bad data is caught automatically on
   every run, not just when someone remembers to run this script by hand
   first. Single source of truth: the check is
   written once, here, not duplicated into all 8 notebooks.
"""
import argparse
import sys

import pandas as pd

USER_ID_COLUMN = "USER_ID"
EVENT_DATE_COLUMN = "EVENT_DT"
TIMESTAMP_COLUMN = "30d_DATE_TIME_ET"
BEHAVIOR_COLUMN_NAMES = [
    "30d_CHANNEL",
    "30d_DEVICE_TYPE",
    "30d_EDITION",
    "30d_GEO_REGION",
    "30d_PAGE_HIT_REF_TYPE",
    "30d_PAYWALL",
    "30d_CONTENT_SOURCE_HIT",
    "30d_CONTENT_TYPE_HIT",
    "30d_SECTION_HIT",
    "30d_SUBSCRIBER_STATUS",
    "30d_PAGE_ID",
]
LIST_COLUMNS = BEHAVIOR_COLUMN_NAMES + [TIMESTAMP_COLUMN]
REQUIRED_COLUMNS = [USER_ID_COLUMN, EVENT_DATE_COLUMN] + LIST_COLUMNS


def is_missing_cell(val) -> bool:
    """True if the cell itself is null (not a list at all) -- e.g. None,
    np.nan, pd.NA. A real (even empty) list/array returns False here."""
    if val is None:
        return True
    if not hasattr(val, "__len__"):
        try:
            return bool(pd.isna(val))
        except (TypeError, ValueError):
            return True
    return False


def validate(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Runs every check against an already-loaded DataFrame. Returns
    (errors, warnings) -- both plain lists of human-readable strings.
    Raises nothing; the caller decides what to do with the results (the
    CLI below exits non-zero, a notebook raises SystemExit with them)."""
    errors = []
    warnings = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")
        return errors, warnings  # nothing else can be checked without these

    dup_count = int(df.duplicated(subset=[USER_ID_COLUMN, EVENT_DATE_COLUMN]).sum())
    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate ({USER_ID_COLUMN}, {EVENT_DATE_COLUMN}) rows found")

    # dict-key access (not itertuples/getattr) -- real column names like
    # "30d_CHANNEL" start with a digit, not a valid Python identifier;
    # itertuples would silently rename such fields and break getattr.
    records = df.to_dict("records")

    null_cell_rows = {col: [] for col in LIST_COLUMNS}
    length_mismatch_rows = []

    for idx, r in enumerate(records):
        row_lengths = {}
        row_has_null_cell = False

        for col in LIST_COLUMNS:
            val = r[col]
            if is_missing_cell(val):
                null_cell_rows[col].append(idx)
                row_has_null_cell = True
                continue
            row_lengths[col] = len(val)

        if row_has_null_cell:
            continue  # already flagged as an error below; length check is meaningless here

        expected_len = row_lengths.get(TIMESTAMP_COLUMN)
        if len(set(row_lengths.values())) > 1:
            length_mismatch_rows.append((idx, r[USER_ID_COLUMN], r[EVENT_DATE_COLUMN], expected_len, dict(row_lengths)))

    for col, rows in null_cell_rows.items():
        if rows:
            errors.append(
                f"{col}: {len(rows)} row(s) have a NULL CELL (not even an empty list) -- "
                f"example row indices: {rows[:5]}"
            )

    if length_mismatch_rows:
        detail_lines = [
            f"    row {idx} ({USER_ID_COLUMN}={user}, {EVENT_DATE_COLUMN}={date}): "
            f"expected length (from {TIMESTAMP_COLUMN})={expected_len}, lengths seen={lengths}"
            for idx, user, date, expected_len, lengths in length_mismatch_rows[:5]
        ]
        errors.append(
            f"{len(length_mismatch_rows)} row(s) have 30d_* list lengths that don't all match "
            f"{TIMESTAMP_COLUMN}'s length for that row -- first 5 examples:\n" + "\n".join(detail_lines)
        )

    return errors, warnings


def print_report(errors: list, warnings: list) -> None:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="s3://your-bucket-name/input/bucketized_30day_all_events.parquet")
    args = parser.parse_args()

    print(f"Checking: {args.input}")
    try:
        df = pd.read_parquet(args.input, storage_options={})
    except Exception as e:
        print(f"FAIL: could not read file -- {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"  {len(df)} rows loaded")

    errors, warnings = validate(df)
    if not any("missing required columns" in e for e in errors):
        print(f"  all {len(REQUIRED_COLUMNS)} required columns present")
    print_report(errors, warnings)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
