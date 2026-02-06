# openai-ft-cli

A lightweight command-line utility for **OpenAI-hosted fine-tuning** workflows (tested with `openai==2.17.0`).

## Why this exists

OpenAI’s web UI for fine-tuning is intentionally simple. In practice, it can feel limiting when you need to:

- quickly list *your* fine-tuned models
- enumerate fine-tuning jobs and correlate jobs → resulting models
- inspect job events (progress, warnings, status transitions)
- pull checkpoints and their metrics
- download and parse result files (often where train/eval loss live)
- estimate cost from billable training tokens

There isn’t an obvious, maintained “basic admin CLI” that covers these everyday tasks, so this script provides a pragmatic baseline.

---

## What it does

This single-file CLI (`openai_ft_cli.py`) provides:

- **Models**
  - list your fine-tuned models (filters `ft:` IDs)
  - delete a fine-tuned model by ID

- **Fine-tuning jobs**
  - list jobs (status, base model, resulting fine-tuned model)
  - list job events
  - list job checkpoints (including checkpoint metrics)

- **Training statistics export**
  - dump a job’s **all available** training artifacts to a single **LLM-friendly JSON blob**:
    - job metadata
    - checkpoints (+ metrics)
    - optionally events
    - result files (downloaded and CSV-parsed when applicable)

- **Cost estimate**
  - read `trained_tokens` from the job and estimate training USD using a configurable `$ / 1M tokens` rate table
  - defaults to `gpt-4.1-mini`

---

## Requirements

- Python 3.9+ recommended
- `openai==2.17.0` (tested)
  - other 2.x versions may work, but this script targets the 2.17.0 resource layout

Install dependency:

```bash
pip install --upgrade openai==2.17.0
```

---

## Authentication & environment variables

The script reads the API key from an environment variable.

Required:

- `OPENAI_API_KEY` — your OpenAI API key

Optional (if you use org/project scoping):

- `OPENAI_ORG_ID` — sent as `OpenAI-Organization` header
- `OPENAI_PROJECT` — sent as `OpenAI-Project` header

Examples:

```bash
export OPENAI_API_KEY="sk-..."
# optional:
export OPENAI_ORG_ID="org_..."
export OPENAI_PROJECT="proj_..."
```

---

## Download / run

Make the script executable (optional):

```bash
chmod +x openai_ft_cli.py
```

Run:

```bash
python openai_ft_cli.py --help
```

---

## Commands

### `models list`

Lists fine-tuned models visible to your key (IDs starting with `ft:`).

```bash
python openai_ft_cli.py models list
python openai_ft_cli.py models list --verbose
```

Options:
- `--verbose`  
  Outputs **all fields** returned by the Models API as **JSONL** (one JSON object per line), suitable for piping to `jq`.

---

### `models delete <model_id>`

Deletes a fine-tuned model by ID (must look like `ft:...`).

```bash
python openai_ft_cli.py models delete "ft:..."
```

Notes:
- Deleting models may require appropriate org permissions/role.

---

### `jobs list`

Lists fine-tuning jobs.

```bash
python openai_ft_cli.py jobs list
python openai_ft_cli.py jobs list --limit 50
python openai_ft_cli.py jobs list --after ftjob_...   # pagination cursor
python openai_ft_cli.py jobs list --verbose
```

Options:
- `--limit N` (default 20)
- `--after ID` (pagination cursor)
- `--verbose` outputs JSONL with all fields

---

### `events <job_id>`

Lists events for a fine-tuning job (progress/status messages).

```bash
python openai_ft_cli.py events ftjob-...
python openai_ft_cli.py events ftjob-... --limit 200
python openai_ft_cli.py events ftjob-... --verbose | jq .
```

Options:
- `--limit N` (default 50)
- `--after ID` (pagination cursor)
- `--verbose` outputs JSONL

---

### `checkpoints <job_id>`

Lists checkpoints for a fine-tuning job, including checkpoint metrics (when available).

```bash
python openai_ft_cli.py checkpoints ftjob-...
python openai_ft_cli.py checkpoints ftjob-... --verbose | jq .
```

Options:
- `--limit N` (default 20)
- `--after ID` (pagination cursor)
- `--verbose` outputs JSONL

---

### `stats <job_id>`

Dumps **everything available** for a job as **one JSON object** (ideal for analysis, regression tracking, or feeding into another LLM).

It pulls:
- job metadata
- checkpoints (+ metrics)
- optional events
- result files listed on the job (downloads them and parses CSV when applicable)

```bash
python openai_ft_cli.py stats ftjob-... --pretty > ftjob_stats.json
```

Options:
- `--pretty`  
  Pretty JSON output.
- `--no-events`  
  Skip events in output (reduces size).
- `--events-limit N` (default 200)  
  How many events to include if enabled.
- `--max-rows N` (default 5000; `-1` = unlimited)  
  Max CSV rows parsed per result file.
- `--include-raw-csv`  
  Include full raw result file text (can be large).

Typical usage patterns:

```bash
# full dump (pretty)
python openai_ft_cli.py stats ftjob-... --pretty > stats.json

# smaller dump (skip events)
python openai_ft_cli.py stats ftjob-... --no-events > stats_no_events.json

# include raw CSV (largest)
python openai_ft_cli.py stats ftjob-... --include-raw-csv > stats_with_raw.json
```

Output notes:
- Many fine-tune runs provide a results file (often CSV) containing step-level metrics.
- Column names vary; the script also emits:
  - `hints.result_file_loss_like_columns`
  - `hints.checkpoint_metric_keys`
  to help locate loss/accuracy fields quickly.

---

### `cost-estimate <job_id>`

Estimates training cost from `trained_tokens`:

```bash
python openai_ft_cli.py cost-estimate ftjob-... --pretty
```

Options:
- `--model MODEL` (default `gpt-4.1-mini`)  
  Select which model’s training rate is used.
  The script includes a small editable table `TRAINING_USD_PER_1M` near the top.
- `--pretty` pretty JSON output

Examples:

```bash
# default gpt-4.1-mini
python openai_ft_cli.py cost-estimate ftjob-... --pretty

# estimate using a different model pricing
python openai_ft_cli.py cost-estimate ftjob-... --model gpt-4.1 --pretty
```

Notes:
- `trained_tokens` is often `null` while the job is still running. Re-run after completion.
- This is an **estimate** based on the script’s local rate table. Update `TRAINING_USD_PER_1M` if pricing changes.

---

## Practical workflow examples

### Find the latest completed job and inspect metrics

```bash
python openai_ft_cli.py jobs list --limit 20
python openai_ft_cli.py checkpoints ftjob-... 
python openai_ft_cli.py stats ftjob-... --no-events --pretty > stats.json
```

### Compare models you’ve created

```bash
python openai_ft_cli.py models list
python openai_ft_cli.py models list --verbose | jq -r '.id'
```

### Clean up an old fine-tuned model

```bash
python openai_ft_cli.py models delete "ft:..."
```

---

## Troubleshooting

- **No models/jobs returned**
  - Confirm you set `OPENAI_API_KEY`
  - If you use Projects, set `OPENAI_PROJECT` so you’re querying the expected scope.

- **`trained_tokens` is null**
  - The job may still be running (or not finished producing accounting fields). Re-run after status is `succeeded`/`failed`.

- **Result files missing loss/eval loss**
  - Metrics exposure depends on model/run configuration and what the platform emits for your job.
  - Check both result files (CSV) and checkpoint metrics.

---

## License

Use/modify internally as you like. If you plan to publish, add a license header and choose a license (MIT/Apache-2.0/etc.).
