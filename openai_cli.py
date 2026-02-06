#!/usr/bin/env python3
"""
openai_ft_cli.py  (tested for openai==2.17.0)

A small CLI utility for OpenAI-hosted fine-tuning workflows:

Commands:
  models list [--verbose]
  models delete <model_id>

  jobs list [--limit N] [--after ID] [--verbose]
  events <job_id> [--limit N] [--after ID] [--verbose]
  checkpoints <job_id> [--limit N] [--after ID] [--verbose]

  stats <job_id> [--events-limit N] [--no-events]
                [--max-rows N] [--include-raw-csv]
                [--pretty]

  cost-estimate <job_id> [--model MODEL] [--pretty]

Environment:
  OPENAI_API_KEY   (required)
  OPENAI_ORG_ID    (optional)
  OPENAI_PROJECT   (optional)

Notes:
- "models list" filters only model IDs starting with "ft:".
- "stats" dumps everything it can find into one JSON blob:
    * job metadata
    * checkpoints (+ metrics)
    * optional events
    * result files (downloaded, CSV parsed if applicable)
- "cost-estimate" reads trained_tokens from the fine-tune job and multiplies
  by $/1M training tokens for the given model (default: gpt-4.1-mini).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency. Install with: pip install --upgrade openai", file=sys.stderr)
    raise

FINE_TUNE_PREFIXES = ("ft:",)

# Training $/1M tokens (Standard tier) — update if pricing changes.
# Default is gpt-4.1-mini as requested.
TRAINING_USD_PER_1M: Dict[str, float] = {
    "gpt-4.1-mini": 5.00,
    "gpt-4.1": 25.00,
    "gpt-4.1-nano": 1.50,
    "gpt-4o": 25.00,
    "gpt-4o-mini": 3.00,
    "gpt-3.5-turbo": 8.00,
    "davinci-002": 6.00,
    "babbage-002": 0.40,
}


# ----------------------------
# Helpers
# ----------------------------

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(f"Missing env var {name}. Example: export {name}='...'\n")
    return val


def is_finetuned_model_id(model_id: str) -> bool:
    return model_id.startswith(FINE_TUNE_PREFIXES)


def to_plain_dict(obj: Any) -> Dict[str, Any]:
    """Convert OpenAI SDK objects to a JSON-serializable dict."""
    if obj is None:
        return {}
    fn = getattr(obj, "model_dump", None)
    if callable(fn):
        return fn()
    fn = getattr(obj, "to_dict", None)
    if callable(fn):
        return fn()
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return d
    return {"value": str(obj)}


def print_verbose_jsonl(objs: Iterable[Any]) -> None:
    for o in objs:
        print(json.dumps(to_plain_dict(o), ensure_ascii=False, sort_keys=True))


def build_client() -> OpenAI:
    api_key = require_env("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID")
    project = os.getenv("OPENAI_PROJECT")

    default_headers = {}
    if org_id:
        default_headers["OpenAI-Organization"] = org_id
    if project:
        default_headers["OpenAI-Project"] = project

    if default_headers:
        return OpenAI(api_key=api_key, default_headers=default_headers)
    return OpenAI(api_key=api_key)


def normalize_model_name(model: str) -> str:
    """
    Map versioned names like 'gpt-4.1-mini-2025-04-14' to 'gpt-4.1-mini'
    when estimating cost.
    """
    if not model:
        return model
    parts = model.split("-")
    # Common date suffix: YYYY-MM-DD
    if len(parts) >= 4 and parts[-3].isdigit() and parts[-2].isdigit() and parts[-1].isdigit():
        return "-".join(parts[:-3])
    return model


def training_rate_for(model: str) -> float:
    key = normalize_model_name(model)
    if key in TRAINING_USD_PER_1M:
        return TRAINING_USD_PER_1M[key]
    raise SystemExit(
        f"Unknown model '{model}'. Use --model with one of: {', '.join(sorted(TRAINING_USD_PER_1M.keys()))}"
    )


# ----------------------------
# Models
# ----------------------------

def iter_models(client: OpenAI) -> Iterable[Any]:
    resp = client.models.list()
    return getattr(resp, "data", resp)


def cmd_models_list(client: OpenAI, verbose: bool) -> int:
    models = list(iter_models(client))
    ft_models = [m for m in models if is_finetuned_model_id(getattr(m, "id", ""))]
    ft_models = sorted(ft_models, key=lambda x: getattr(x, "id", ""))

    if not ft_models:
        print("No fine-tuned models found for this API key / org/project scope.")
        return 0

    if verbose:
        print_verbose_jsonl(ft_models)
        return 0

    for m in ft_models:
        mid = getattr(m, "id", "")
        owned_by = getattr(m, "owned_by", "")
        created = getattr(m, "created", None)
        print(f"{mid}\towned_by={owned_by}\tcreated={created}")

    return 0


def cmd_models_delete(client: OpenAI, model_id: str) -> int:
    if not is_finetuned_model_id(model_id):
        print(
            f"Refusing to delete '{model_id}' because it doesn't look like a fine-tuned model id "
            f"(expected prefix {FINE_TUNE_PREFIXES}).",
            file=sys.stderr,
        )
        return 2

    resp = client.models.delete(model_id)
    deleted = getattr(resp, "deleted", None)
    rid = getattr(resp, "id", None) or model_id
    print(f"deleted={deleted}\tmodel={rid}")
    return 0 if deleted else 1


# ----------------------------
# Fine-tuning jobs / events / checkpoints
# ----------------------------

def list_ft_jobs(client: OpenAI, limit: int, after: Optional[str]) -> Iterable[Any]:
    kwargs: Dict[str, Any] = {"limit": limit}
    if after:
        kwargs["after"] = after
    resp = client.fine_tuning.jobs.list(**kwargs)
    return getattr(resp, "data", resp)


def cmd_jobs_list(client: OpenAI, limit: int, after: Optional[str], verbose: bool) -> int:
    jobs = list(list_ft_jobs(client, limit=limit, after=after))
    if not jobs:
        print("No fine-tuning jobs found in this org/project scope.")
        return 0

    if verbose:
        print_verbose_jsonl(jobs)
        return 0

    for j in jobs:
        jid = getattr(j, "id", "")
        status = getattr(j, "status", "")
        base_model = getattr(j, "model", "")
        ft_model = getattr(j, "fine_tuned_model", None)
        created_at = getattr(j, "created_at", None)
        finished_at = getattr(j, "finished_at", None)
        print(f"{jid}\tstatus={status}\tbase={base_model}\tft={ft_model}\tcreated_at={created_at}\tfinished_at={finished_at}")
    return 0


def list_ft_events(client: OpenAI, job_id: str, limit: int, after: Optional[str]) -> Iterable[Any]:
    kwargs: Dict[str, Any] = {"fine_tuning_job_id": job_id, "limit": limit}
    if after:
        kwargs["after"] = after
    resp = client.fine_tuning.jobs.list_events(**kwargs)
    return getattr(resp, "data", resp)


def cmd_events_list(client: OpenAI, job_id: str, limit: int, after: Optional[str], verbose: bool) -> int:
    events = list(list_ft_events(client, job_id=job_id, limit=limit, after=after))
    if not events:
        print("No events returned (job id may be wrong, or job has no events yet).")
        return 0

    if verbose:
        print_verbose_jsonl(events)
        return 0

    for e in sorted(events, key=lambda x: getattr(x, "created_at", 0)):
        created_at = getattr(e, "created_at", None)
        level = getattr(e, "level", "")
        etype = getattr(e, "type", "")
        msg = getattr(e, "message", "")
        eid = getattr(e, "id", "")
        print(f"{created_at}\t{level}\t{etype}\t{eid}\t{msg}")
    return 0


def list_ft_checkpoints(client: OpenAI, job_id: str, limit: int, after: Optional[str]) -> Iterable[Any]:
    # openai==2.17.0: checkpoints are a sub-resource under fine_tuning.jobs
    kwargs: Dict[str, Any] = {"fine_tuning_job_id": job_id, "limit": limit}
    if after:
        kwargs["after"] = after
    resp = client.fine_tuning.jobs.checkpoints.list(**kwargs)
    return getattr(resp, "data", resp)


def cmd_checkpoints_list(client: OpenAI, job_id: str, limit: int, after: Optional[str], verbose: bool) -> int:
    cps = list(list_ft_checkpoints(client, job_id=job_id, limit=limit, after=after))
    if not cps:
        print("No checkpoints returned (job may not have produced checkpoints yet).")
        return 0

    if verbose:
        print_verbose_jsonl(cps)
        return 0

    for c in sorted(cps, key=lambda x: getattr(x, "created_at", 0)):
        created_at = getattr(c, "created_at", None)
        cid = getattr(c, "id", "")
        ckpt_model = getattr(c, "fine_tuned_model_checkpoint", "")
        metrics = getattr(c, "metrics", None)
        metrics_dict = to_plain_dict(metrics) if metrics is not None else {}
        print(f"{created_at}\t{cid}\t{ckpt_model}\tmetrics={json.dumps(metrics_dict, ensure_ascii=False, sort_keys=True)}")
    return 0


# ----------------------------
# Stats (job + checkpoints + result_files + optional events)
# ----------------------------

def download_file_text(client: OpenAI, file_id: str) -> str:
    """
    In openai 2.x SDK: client.files.content(file_id) returns a response-like
    object. We handle .text or bytes.
    """
    content = client.files.content(file_id)

    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text

    data = getattr(content, "content", None) or getattr(content, "data", None)
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", errors="replace")

    # Some SDK versions allow .read()
    read = getattr(content, "read", None)
    if callable(read):
        b = read()
        if isinstance(b, (bytes, bytearray)):
            return b.decode("utf-8", errors="replace")

    raise RuntimeError(f"Could not read file content for {file_id}")


def parse_csv_text(csv_text: str, max_rows: int) -> List[Dict[str, Any]]:
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(reader):
        if max_rows >= 0 and i >= max_rows:
            break
        # Keep strings (lossless); your downstream LLM/tooling can cast.
        rows.append(dict(row))
    return rows


def cmd_stats(
    client: OpenAI,
    job_id: str,
    include_events: bool,
    events_limit: int,
    max_rows: int,
    include_raw_csv: bool,
    pretty: bool,
) -> int:
    job = client.fine_tuning.jobs.retrieve(job_id)
    job_d = to_plain_dict(job)

    cps = list(list_ft_checkpoints(client, job_id=job_id, limit=100, after=None))
    cps_sorted = sorted(cps, key=lambda x: getattr(x, "created_at", 0))
    cps_d = [to_plain_dict(c) for c in cps_sorted]

    events_d: Optional[List[Dict[str, Any]]] = None
    if include_events:
        ev = list(list_ft_events(client, job_id=job_id, limit=events_limit, after=None))
        ev_sorted = sorted(ev, key=lambda x: getattr(x, "created_at", 0))
        events_d = [to_plain_dict(e) for e in ev_sorted]

    result_files = job_d.get("result_files") or []
    result_files_out: List[Dict[str, Any]] = []

    for fid in result_files:
        entry: Dict[str, Any] = {"file_id": fid}
        try:
            meta = to_plain_dict(client.files.retrieve(fid))
            entry["meta"] = meta

            text = download_file_text(client, fid)

            filename = (meta.get("filename") or "").lower()
            looks_csv = filename.endswith(".csv") or (text.splitlines() and "," in text.splitlines()[0])

            if looks_csv:
                entry["rows"] = parse_csv_text(text, max_rows=max_rows)
                if entry["rows"]:
                    entry["columns"] = list(entry["rows"][0].keys())
            else:
                entry["text_preview"] = text[:5000]

            if include_raw_csv:
                entry["raw_text"] = text

        except Exception as e:
            entry["download_error"] = str(e)

        result_files_out.append(entry)

    out = {
        "job": job_d,
        "checkpoints": cps_d,
        "events": events_d,
        "result_files": result_files_out,
        "hints": {
            "result_file_loss_like_columns": sorted(
                {
                    col
                    for rf in result_files_out
                    for col in (rf.get("columns") or [])
                    if "loss" in col.lower() or "accuracy" in col.lower()
                }
            ),
            "checkpoint_metric_keys": sorted(
                {
                    k
                    for cp in cps_d
                    for k in (cp.get("metrics") or {}).keys()
                }
            ),
        },
    }

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None, sort_keys=True))
    return 0


# ----------------------------
# Cost estimate (trained_tokens * rate)
# ----------------------------

def cmd_cost_estimate(client: OpenAI, job_id: str, model: str, pretty: bool) -> int:
    job = client.fine_tuning.jobs.retrieve(job_id)
    job_d = to_plain_dict(job)

    trained_tokens = job_d.get("trained_tokens")
    job_model = job_d.get("model")

    # Pricing model: CLI override (default provided) or job.model if CLI empty
    model_for_calc = model or job_model or "gpt-4.1-mini"
    rate = training_rate_for(model_for_calc)

    if trained_tokens is None:
        out = {
            "job_id": job_id,
            "status": job_d.get("status"),
            "trained_tokens": None,
            "note": "trained_tokens is null while job is running; rerun when finished.",
            "model_for_estimate": model_for_calc,
            "training_usd_per_1m_tokens": rate,
            "job_model": job_model,
            "fine_tuned_model": job_d.get("fine_tuned_model"),
        }
    else:
        estimated_usd = (float(trained_tokens) / 1_000_000.0) * float(rate)
        out = {
            "job_id": job_id,
            "status": job_d.get("status"),
            "trained_tokens": trained_tokens,
            "model_for_estimate": model_for_calc,
            "training_usd_per_1m_tokens": rate,
            "estimated_training_usd": estimated_usd,
            "job_model": job_model,
            "fine_tuned_model": job_d.get("fine_tuned_model"),
            "created_at": job_d.get("created_at"),
            "finished_at": job_d.get("finished_at"),
        }

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None, sort_keys=True))
    return 0


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="OpenAI fine-tune utility (models + jobs/events/checkpoints + stats + cost-estimate).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # models
    p_models = sub.add_parser("models", help="Manage fine-tuned models (via Models API)")
    sub_models = p_models.add_subparsers(dest="models_cmd", required=True)

    p_models_list = sub_models.add_parser("list", help="List fine-tuned models (ids starting with ft:)")
    p_models_list.add_argument("--verbose", action="store_true", help="Print all available fields as JSONL")

    p_models_del = sub_models.add_parser("delete", help="Delete a fine-tuned model by id")
    p_models_del.add_argument("model_id", help="Fine-tuned model id (e.g., ft:...)")

    # jobs
    p_jobs = sub.add_parser("jobs", help="Fine-tuning jobs")
    sub_jobs = p_jobs.add_subparsers(dest="jobs_cmd", required=True)

    p_jobs_list = sub_jobs.add_parser("list", help="List fine-tuning jobs")
    p_jobs_list.add_argument("--limit", type=int, default=20, help="Max jobs to return (default: 20)")
    p_jobs_list.add_argument("--after", type=str, default=None, help="Pagination cursor (job id)")
    p_jobs_list.add_argument("--verbose", action="store_true", help="Print all available fields as JSONL")

    # events
    p_events = sub.add_parser("events", help="List events for a fine-tuning job")
    p_events.add_argument("job_id", help="Fine-tuning job id (e.g., ftjob-...)")
    p_events.add_argument("--limit", type=int, default=50, help="Max events to return (default: 50)")
    p_events.add_argument("--after", type=str, default=None, help="Pagination cursor (event id)")
    p_events.add_argument("--verbose", action="store_true", help="Print all available fields as JSONL")

    # checkpoints
    p_cps = sub.add_parser("checkpoints", help="List checkpoints for a fine-tuning job")
    p_cps.add_argument("job_id", help="Fine-tuning job id (e.g., ftjob-...)")
    p_cps.add_argument("--limit", type=int, default=20, help="Max checkpoints to return (default: 20)")
    p_cps.add_argument("--after", type=str, default=None, help="Pagination cursor (checkpoint id)")
    p_cps.add_argument("--verbose", action="store_true", help="Print all available fields as JSONL")

    # stats
    p_stats = sub.add_parser("stats", help="Dump all available training statistics for a job as one JSON blob")
    p_stats.add_argument("job_id", help="Fine-tuning job id (e.g., ftjob-...)")
    p_stats.add_argument("--no-events", action="store_true", help="Do not include events timeline")
    p_stats.add_argument("--events-limit", type=int, default=200, help="Max events to include (default: 200)")
    p_stats.add_argument("--max-rows", type=int, default=5000, help="Max CSV rows per result file (default: 5000; -1 = unlimited)")
    p_stats.add_argument("--include-raw-csv", action="store_true", help="Include full raw results file text (can be large)")
    p_stats.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    # cost-estimate
    p_cost = sub.add_parser("cost-estimate", help="Estimate training cost from trained_tokens")
    p_cost.add_argument("job_id", help="Fine-tuning job id (e.g., ftjob-...)")
    p_cost.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model name used for pricing (default: gpt-4.1-mini). "
             f"Known: {', '.join(sorted(TRAINING_USD_PER_1M.keys()))}",
    )
    p_cost.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    args = p.parse_args()
    client = build_client()

    if args.cmd == "models":
        if args.models_cmd == "list":
            return cmd_models_list(client, verbose=args.verbose)
        if args.models_cmd == "delete":
            return cmd_models_delete(client, args.model_id)

    if args.cmd == "jobs":
        if args.jobs_cmd == "list":
            return cmd_jobs_list(client, limit=args.limit, after=args.after, verbose=args.verbose)

    if args.cmd == "events":
        return cmd_events_list(client, job_id=args.job_id, limit=args.limit, after=args.after, verbose=args.verbose)

    if args.cmd == "checkpoints":
        return cmd_checkpoints_list(client, job_id=args.job_id, limit=args.limit, after=args.after, verbose=args.verbose)

    if args.cmd == "stats":
        return cmd_stats(
            client,
            job_id=args.job_id,
            include_events=not args.no_events,
            events_limit=args.events_limit,
            max_rows=args.max_rows,
            include_raw_csv=args.include_raw_csv,
            pretty=args.pretty,
        )

    if args.cmd == "cost-estimate":
        return cmd_cost_estimate(client, job_id=args.job_id, model=args.model, pretty=args.pretty)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
