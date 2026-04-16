"""
check_trajectories.py

Sanity-check a collected trajectory JSONL file and print a summary report.

Reports:
  - Total trajectories
  - Score distribution: score == 1.0, score >= 0.5, score < 0.5
  - Schema errors: steps where reflect_on_search is called with original_question
    (this argument is not in reflect_on_search's schema — it causes an infinite loop)
  - One example trajectory printed in full (first row by default, or --example-id)

Usage (paths default under $AGENTIC_MEMORY_LOG_DIR, default /lc3T/AgenticMemory/logs):
    python sft/check_trajectories.py /lc3T/AgenticMemory/logs/trajectories_overfit100.jsonl
    python sft/check_trajectories.py /lc3T/AgenticMemory/logs/trajectories_overfit100.jsonl --example-id 3
    python sft/check_trajectories.py /lc3T/AgenticMemory/logs/trajectories_overfit100.jsonl --show-errors
"""

import argparse
import json
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema-error detection
# ---------------------------------------------------------------------------

def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def has_schema_error(row: dict) -> bool:
    """Return True if any step calls reflect_on_search with original_question."""
    for step in row.get("steps", []):
        for tc in step.get("tool_calls") or []:
            if tc.get("name") == "reflect_on_search":
                args = _parse_args(tc.get("arguments", {}))
                if "original_question" in args:
                    return True
    return False


# ---------------------------------------------------------------------------
# Pretty-print one trajectory
# ---------------------------------------------------------------------------

def print_example(row: dict, idx: int) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"EXAMPLE TRAJECTORY #{idx}")
    print(sep)
    print(f"  id            : {row.get('id', '—')}")
    print(f"  question_id   : {row.get('question_id', '—')}")
    print(f"  doc_name      : {row.get('doc_name', '—')}")
    print(f"  score         : {row.get('score', '—')}")
    print(f"  num_steps     : {row.get('num_steps', '—')}")
    print(f"  schema_error  : {has_schema_error(row)}")
    print()
    print("  QUESTION:")
    print(textwrap.indent(row.get("question", ""), "    "))
    print()
    print("  GROUND TRUTH :", row.get("ground_truth", "—"))
    print("  MODEL ANSWER :", row.get("model_answer", "—"))
    print()
    print("  STEPS:")
    for i, step in enumerate(row.get("steps", [])):
        tool_calls = step.get("tool_calls") or []
        tool_responses = step.get("tool_responses") or []
        if not tool_calls:
            continue
        print(f"    Step {i}:")
        for tc in tool_calls:
            args = _parse_args(tc.get("arguments", {}))
            args_preview = json.dumps(args, ensure_ascii=False)
            if len(args_preview) > 120:
                args_preview = args_preview[:117] + "..."
            print(f"      → {tc.get('name')}({args_preview})")
        for tr in tool_responses:
            content = (tr.get("content") or "").strip()
            if len(content) > 160:
                content = content[:157] + "..."
            n_imgs = len(tr.get("image_paths") or [])
            img_note = f" [{n_imgs} image(s)]" if n_imgs else ""
            print(f"      ← {content}{img_note}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sanity-check trajectory JSONL file.")
    parser.add_argument("input", help="Path to trajectory JSONL file")
    parser.add_argument(
        "--example-id",
        type=int,
        default=0,
        help="0-based index of the trajectory to display in full (default: 0)",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Also print all trajectories that have schema errors",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return

    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"WARNING: malformed JSON on line {lineno}, skipping.")

    total = len(rows)
    score_10 = sum(1 for r in rows if float(r.get("score", 0)) == 1.0)
    score_05 = sum(1 for r in rows if float(r.get("score", 0)) >= 0.5)
    score_lo = total - score_05
    schema_err = sum(1 for r in rows if has_schema_error(r))
    schema_ok = total - schema_err

    # Average steps among correct trajectories
    correct_rows = [r for r in rows if float(r.get("score", 0)) == 1.0]
    avg_steps = (
        sum(r.get("num_steps", 0) for r in correct_rows) / len(correct_rows)
        if correct_rows else 0
    )

    print()
    print("=" * 60)
    print(f"  TRAJECTORY SANITY CHECK: {path.name}")
    print("=" * 60)
    print(f"  Total trajectories      : {total}")
    print()
    print("  Score distribution:")
    print(f"    score == 1.0          : {score_10:4d}  ({100*score_10/total:.1f}%)")
    print(f"    score >= 0.5          : {score_05:4d}  ({100*score_05/total:.1f}%)")
    print(f"    score <  0.5          : {score_lo:4d}  ({100*score_lo/total:.1f}%)")
    print(f"    avg steps (score=1.0) : {avg_steps:.1f}")
    print()
    print("  Schema errors (reflect_on_search + original_question):")
    print(f"    has error             : {schema_err:4d}  ({100*schema_err/total:.1f}%)")
    print(f"    clean                 : {schema_ok:4d}  ({100*schema_ok/total:.1f}%)")
    print("=" * 60)

    # Print the requested example
    if 0 <= args.example_id < total:
        print_example(rows[args.example_id], args.example_id)
    else:
        print(f"\nWARNING: --example-id {args.example_id} out of range (0–{total-1}); skipping.")

    # Optionally print all error trajectories
    if args.show_errors:
        error_rows = [(i, r) for i, r in enumerate(rows) if has_schema_error(r)]
        if not error_rows:
            print("\nNo schema errors found.")
        else:
            print(f"\nTrajectories with schema errors ({len(error_rows)}):")
            for i, r in error_rows:
                print_example(r, i)


if __name__ == "__main__":
    main()
