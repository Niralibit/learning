"""
Joins one architecture's embedding output back to the event-level columns
that the embedding notebooks never read (CONTACT_METHOD, ACTION_SUGGESTED,
EVENT_SENT_PERIOD, OUTCOME), matched on (USER_ID, EVENT_DT).

This is the file meant to actually go to XGBoost: one row per event, with
the 256-d embedding vector sitting alongside the campaign details and (if
present) the real outcome label. The raw `30d_*` behavioral list-columns
are NOT carried over -- the embedding already summarizes them, and the
raw lists would just bloat the file with data XGBoost doesn't consume
directly.

Reads/writes directly to S3 via pandas + s3fs (storage_options={}) -- see
requirements.txt for the AWS-credentials note.

Run (defaults to the Hybrid output at the placeholder S3 paths below --
edit DEFAULT_EMBEDDINGS/DEFAULT_EVENT_DATA, or pass --embeddings/--event-data,
once you have the real bucket name):
    python join_embeddings.py

Or specify exactly which embedding file and event data to join:
    python join_embeddings.py --embeddings s3://your-bucket-name/output/hstu_embeddings.parquet --event-data s3://your-bucket-name/input/bucketized_30day_all_events.parquet --output s3://your-bucket-name/output/hstu_final.parquet
"""
import argparse
import os

import pandas as pd

DEFAULT_EMBEDDINGS = "s3://your-bucket-name/output/hybrid_embeddings.parquet"
DEFAULT_EVENT_DATA = "s3://your-bucket-name/input/bucketized_30day_all_events.parquet"
USER_ID_COLUMN = "USER_ID"
EVENT_DATE_COLUMN = "EVENT_DT"
EVENT_LEVEL_COLUMNS = ["CONTACT_METHOD", "ACTION_SUGGESTED", "EVENT_SENT_PERIOD", "OUTCOME"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS,
                         help="path to one <architecture>_embeddings.parquet file")
    parser.add_argument("--event-data", default=DEFAULT_EVENT_DATA,
                         help="path to the original event data file "
                              "(has CONTACT_METHOD/ACTION_SUGGESTED/EVENT_SENT_PERIOD/OUTCOME)")
    parser.add_argument("--output", default=None,
                         help="output path (default: <embeddings-file>_with_event.parquet, "
                              "next to the embeddings file)")
    args = parser.parse_args()

    print(f"Embeddings: {args.embeddings}")
    print(f"Event data: {args.event_data}")
    emb_df = pd.read_parquet(args.embeddings, storage_options={})
    event_df = pd.read_parquet(args.event_data, storage_options={})
    print(f"  {len(emb_df)} embedding rows, {len(event_df)} event rows")

    for col in (USER_ID_COLUMN, EVENT_DATE_COLUMN):
        if col not in emb_df.columns:
            raise SystemExit(f"FAIL: '{col}' not found in {args.embeddings} -- cannot join without it")
        if col not in event_df.columns:
            raise SystemExit(f"FAIL: '{col}' not found in {args.event_data} -- cannot join without it")

    present_event_cols = [c for c in EVENT_LEVEL_COLUMNS if c in event_df.columns]
    missing_event_cols = [c for c in EVENT_LEVEL_COLUMNS if c not in event_df.columns]
    if missing_event_cols:
        print(f"Note: these expected event-level columns are not in {args.event_data}, skipping: {missing_event_cols}")
    print(f"  carrying over: {present_event_cols}")

    event_subset = event_df[[USER_ID_COLUMN, EVENT_DATE_COLUMN] + present_event_cols].drop_duplicates(
        subset=[USER_ID_COLUMN, EVENT_DATE_COLUMN]
    )

    merged = emb_df.merge(event_subset, on=[USER_ID_COLUMN, EVENT_DATE_COLUMN], how="left")

    if present_event_cols:
        unmatched = int(merged[present_event_cols[0]].isna().sum())
        if unmatched > 0:
            print(f"WARNING: {unmatched} of {len(merged)} embedding rows had NO matching "
                  f"({USER_ID_COLUMN}, {EVENT_DATE_COLUMN}) in {args.event_data} -- "
                  f"those rows have null event-level columns in the output")
        else:
            print(f"  all {len(merged)} embedding rows matched an event row")

    out_path = args.output
    if out_path is None:
        base, ext = os.path.splitext(args.embeddings)
        out_path = f"{base}_with_event{ext}"

    merged.to_parquet(out_path, index=False, storage_options={})
    emb_cols = [c for c in merged.columns if c.startswith("emb_")]
    print(f"wrote {len(merged)} rows x {len(merged.columns)} cols -> {out_path}")
    print(f"  columns: {USER_ID_COLUMN}, {EVENT_DATE_COLUMN}, {len(emb_cols)} embedding dims, {present_event_cols}")


if __name__ == "__main__":
    main()
