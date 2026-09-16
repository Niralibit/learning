# GPU-only embedding notebooks (S3 input/output)

4 architectures, each an independent Jupyter notebook, GPU-only. Reads its
input directly from S3 and writes its output directly back to S3 -- no
local data files at all. **Nothing outside this folder is required** to
run it (aside from your AWS credentials being available in the
environment) -- zip this whole directory and it will run as-is on any
GPU-equipped AWS instance with the packages in `requirements.txt`
installed.

## Before you run anything: set the real bucket name

Every path below currently uses the placeholder `your-bucket-name`. Before
running in AWS, replace it with your real bucket in these 3 files:
- `code/hstu.ipynb`, `code/sasrec.ipynb`, `code/gru4rec.ipynb`,
  `code/hybrid.ipynb` -- each has `INPUT_PATH`/`OUTPUT_PATH` near the top
  (in the cell defining `COLUMN_NAME_MAP` etc.)
- `check_data.py` -- the `--input` default
- `join_embeddings.py` -- `DEFAULT_EMBEDDINGS` / `DEFAULT_EVENT_DATA`

Expected layout in S3:
```
s3://your-bucket-name/input/bucketized_30day_all_events.parquet   <- input, you provide this
s3://your-bucket-name/output/<architecture>_embeddings.parquet     <- written by each notebook
```

AWS credentials themselves are **not configured anywhere in this code** --
`s3fs` (via `boto3`/`botocore` underneath) automatically discovers
whatever is already available in the environment: an IAM role attached to
the instance, `~/.aws/credentials`, or `AWS_*` environment variables. As
long as that identity can read the input path and write the output path,
no code changes are needed for auth.

## What's inside

```
code_final/
├── requirements.txt          <- only external dependency list needed
├── check_data.py              <- validation logic; imported automatically by every
│                                  notebook, and also runnable standalone
├── join_embeddings.py         <- run after generating embeddings, to attach
│                                  CONTACT_METHOD/ACTION_SUGGESTED/EVENT_SENT_PERIOD/OUTCOME
├── src/
│   └── hstu.py                <- HSTULayer, used by hstu.ipynb and hybrid.ipynb
└── code/
    ├── hstu.ipynb
    ├── sasrec.ipynb
    ├── gru4rec.ipynb
    └── hybrid.ipynb
```

No `data/` folder -- the input comes from S3, not a bundled local file.

## Expected input schema (real data)

```
USER_ID, EVENT_DT,
30d_CHANNEL, 30d_CONTENT_SOURCE_HIT, 30d_CONTENT_TYPE_HIT, 30d_DATE_TIME_ET,
30d_DEVICE_TYPE, 30d_EDITION, 30d_GEO_REGION, 30d_PAGE_HIT_REF_TYPE,
30d_PAGE_ID, 30d_PAYWALL, 30d_SECTION_HIT, 30d_SUBSCRIBER_STATUS
```

Only the `30d_*` columns are read and embedded -- each one is a per-row
LIST of values, one entry per click in that user's 30-day window before
the event, all lists the same length for a given row (see
`check_data.py` below). `USER_ID` and `EVENT_DT` are carried through as
identifiers on the output.

**Columns intentionally NOT read by the notebooks** (present in the real
file, but ignored -- join them back afterward using `USER_ID` + `EVENT_DT`,
see `join_embeddings.py`):
- `30d_POST_EVAR3` -- redundant, just the user's own id repeated per click
- `CONTACT_METHOD`, `ACTION_SUGGESTED`, `EVENT_SENT_PERIOD` -- describe the
  campaign/event itself, not the user's prior behavior
- `OUTCOME` -- the real Converted/Ignored/Rejected label, if present.
  Deliberately not used here: the embedding must not depend on the
  outcome, since it needs to work identically at live-scoring time
  (before an outcome exists) as it does on historical training data.

**Technical note on real column names**: names like `30d_CHANNEL` start
with a digit, which is not a valid Python identifier. The notebooks use
`chunk.to_dict("records")` + dict-key lookups (not `itertuples()` +
`getattr()`) specifically because of this -- `itertuples()` silently
renames such columns internally and `getattr()` on the real name would
crash. If you ever add new `30d_*` columns, this pattern is safe for any
column name; you don't need to worry about identifier validity.

## Data validation is automatic -- built into every notebook

Every notebook imports `validate()` from `check_data.py` and calls it
right after loading the data from S3, printing the same report
`check_data.py` prints standalone -- if there are any errors, the
notebook stops immediately (`raise SystemExit(...)`) before touching any
model code. You don't have to remember to run `check_data.py` by hand
first; it happens automatically on every run. The check logic itself
lives in exactly one place (`check_data.py`) -- not duplicated into all 4
notebooks -- so a future fix or extension to the checks only needs to
happen once.

You can still run `python check_data.py` standalone as a quick pre-flight
check straight against the S3 file, before ever opening Jupyter:
```bash
python check_data.py --input s3://your-bucket-name/input/bucketized_30day_all_events.parquet
```
It catches the exact things that would otherwise crash a notebook deep
inside model code with a cryptic error: a `30d_*` column that's a hard
null (not even an empty list) for some row, a row where one `30d_*`
column's list length doesn't match `30d_DATE_TIME_ET`'s length for that
same row, a required column missing entirely, or duplicate
`(USER_ID, EVENT_DT)` rows (warning, not blocking). It does **not** flag a
`null` sitting inside an otherwise correctly-sized list, or a list that's
entirely `null` (e.g. `[null, null, null]`) -- both are valid, expected
ways to represent "this field wasn't captured for these clicks."

## What each notebook does

Builds that architecture's encoder with **random, untrained weights** and
runs the input data through it **exactly once** (`torch.no_grad()`) -- no
training loop, no loss, no labels needed. `torch.manual_seed(42)` is set
immediately before the model is built, so the "random" weights -- and
therefore the output embeddings -- are **reproducible**: rerunning a
notebook on the same input always produces bit-for-bit identical output,
it isn't a fresh random draw each time. Output is one 256-dimensional
vector per (`USER_ID`, `EVENT_DT`) row, uploaded to
`s3://your-bucket-name/output/<architecture>_embeddings.parquet`.

The 4 architectures (see the main project's `FINDINGS.md` for the full
background and the trained-vs-untrained discussion):
- **HSTU** -- pointwise SiLU attention with relative position + real
  elapsed-time bias.
- **SASRec** -- standard softmax multi-head attention + learned absolute
  positional embedding.
- **GRU4Rec** -- pure recurrence (GRU), no attention at all.
- **Hybrid** -- the 10 low-cardinality behavior columns go through a GRU
  (192-d); `page_id` (the sparse, ~900-value column) goes through real
  HSTU attention (64-d); concatenated back to 256-d.

Every notebook is independent -- no comparison or scoring between them,
just 4 separate outputs from the same input.

## After generating embeddings: `join_embeddings.py`

The notebooks output only `USER_ID` + `EVENT_DT` + the embedding columns
-- `CONTACT_METHOD`, `ACTION_SUGGESTED`, `EVENT_SENT_PERIOD`, and `OUTCOME`
are read from the input file but never carried into the notebook output.
This script joins them back (reading/writing S3 directly, same as the
notebooks), producing one file ready to hand to XGBoost -- the embedding
vector sitting alongside the campaign details and (if present) the real
outcome label:

```bash
python join_embeddings.py
# defaults to s3://your-bucket-name/output/hybrid_embeddings.parquet (the
# recommended architecture, see FINDINGS.md) joined against
# s3://your-bucket-name/input/bucketized_30day_all_events.parquet,
# writing s3://your-bucket-name/output/hybrid_embeddings_with_event.parquet

# or specify exactly which files:
python join_embeddings.py --embeddings s3://your-bucket-name/output/hstu_embeddings.parquet --event-data s3://your-bucket-name/input/bucketized_30day_all_events.parquet --output s3://your-bucket-name/output/hstu_final.parquet
```

Matches on `(USER_ID, EVENT_DT)`. Warns (doesn't fail) if any embedding
row has no matching event row -- those rows end up with null event-level
columns in the output, which is worth investigating rather than ignoring.
The raw `30d_*` behavioral list-columns are deliberately NOT carried
over -- the embedding already summarizes them.

## Setup

```bash
pip install -r requirements.txt
```

This code is GPU-only. `pip install -r requirements.txt` installs the
CPU-only build of torch by default -- replace it with a CUDA-matched
build (see the comment in `requirements.txt`) on a machine with an
NVIDIA GPU + driver.

## Running

Open any notebook in Jupyter/VS Code and run all cells top to bottom, or
run headless with `jupyter nbconvert --to notebook --execute <path>`.
Each one is fully independent -- run any subset in any order.

## Validated

All 4 notebooks' S3 read/validate/model/write path was tested end-to-end
against a real S3-API-compatible server (`moto` server mode -- a real
local HTTP server implementing S3's API, not just an in-process mock),
using the **actual, unmodified notebook code** (extracted straight from
the `.ipynb` files) with **no test-only code changes** -- AWS
credentials and the S3 endpoint were supplied purely via environment
variables (`AWS_ENDPOINT_URL_S3`, `AWS_ACCESS_KEY_ID`, etc.), the same
mechanism a real IAM role or `~/.aws/credentials` would use. Confirmed:
the notebook read real schema-shaped data from `s3://.../input/...`,
validated it, ran the model on the actual local GPU, and wrote a correct
256-d output back to `s3://.../output/...`, readable back afterward with
the right shape.

Also confirmed while building this: `pandas.read_parquet`/`to_parquet`
on an `s3://` path uses `pyarrow`'s *native* S3 handling by default (since
`pyarrow` is a required dependency here), which does NOT route through
`s3fs`/`boto3` -- meaning tools like `moto`'s `mock_aws` decorator (which
only intercepts `boto3`/`botocore` calls) silently don't apply to it.
Passing `storage_options={}` (even empty) forces pandas onto the
`s3fs`-based path instead, which does respect standard AWS credential
discovery and mocking -- this is why every S3 read/write in this codebase
explicitly passes `storage_options={}`. Confirmed harmless for local file
paths too, in case this code is ever pointed at a local file for
debugging.

Prior to the S3 rework, all 4 (GPU) notebooks were already validated
end-to-end against real-schema-shaped local data: correct 256-d output,
zero NaNs, zero duplicate `(USER_ID, EVENT_DT)` keys, genuinely different
embeddings per architecture (not a copy-paste bug), bit-for-bit
reproducible output across reruns (the fixed seed working as intended),
and correct handling of column names starting with a digit (e.g.
`30d_CHANNEL` -- `itertuples()` + `getattr()` silently breaks on these;
fixed by switching to `to_dict("records")` + dict-key access throughout).
The automatic data-validation integration was verified both ways: clean
data shows "All checks passed" and proceeds; a deliberately corrupted
file (a hard null in a `30d_*` cell) stops the notebook immediately at
the validation step with the exact error, never reaching the
model-building code.
