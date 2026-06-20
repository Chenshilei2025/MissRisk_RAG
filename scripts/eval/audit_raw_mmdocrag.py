from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts.lib.common import find_json_inputs, iter_jsonl, normalize_space, write_json
from scripts.build.mmdocrag import parse_message_record, parse_structured_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw MMDocRAG files before MissRiskBench conversion.")
    parser.add_argument("--input-dir", type=Path, default=Path("data_missrisk/raw/mmdocrag"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/audits/mmdocrag_raw_audit.json"))
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = find_json_inputs(args.input_dir, names=["train.jsonl", "dev_15.jsonl", "dev_20.jsonl", "evaluation_15.jsonl", "evaluation_20.jsonl"])
    if not paths:
        raise SystemExit(f"No MMDocRAG JSONL files found under {args.input_dir}")

    file_counts: dict[str, int] = {}
    modality_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    quote_counts: Counter[str] = Counter()
    gold_count_hist: Counter[int] = Counter()
    problems: list[dict] = []
    total = 0

    for path in paths:
        file_total = 0
        for idx, record in enumerate(iter_jsonl(path)):
            if args.max_examples is not None and total >= args.max_examples:
                break
            question, quotes, _ = parse_structured_record(record) if record.get("question") else parse_message_record(record)
            if not normalize_space(question):
                problems.append({"file": path.name, "line": idx + 1, "problem": "missing_question"})
            if not quotes:
                problems.append({"file": path.name, "line": idx + 1, "problem": "no_quotes_parsed"})
            gold_count = sum(int(quote.get("label_answer_bearing") or 0) for quote in quotes)
            if gold_count == 0:
                problems.append({"file": path.name, "line": idx + 1, "problem": "no_gold_quote_inferred"})
            gold_count_hist[gold_count] += 1
            for modality in record.get("evidence_modality_type") or []:
                modality_counts[str(modality).lower()] += 1
            question_type_counts[str(record.get("question_type") or "unknown")] += 1
            for quote in quotes:
                quote_counts[str(quote.get("quote_type") or "unknown")] += 1
            total += 1
            file_total += 1
        file_counts[path.name] = file_total
        if args.max_examples is not None and total >= args.max_examples:
            break

    write_json(
        args.output_path,
        {
            "input_paths": [str(path) for path in paths],
            "total_examples_audited": total,
            "file_counts": file_counts,
            "quote_counts": dict(quote_counts),
            "gold_count_histogram": {str(key): value for key, value in sorted(gold_count_hist.items())},
            "evidence_modality_counts": dict(modality_counts.most_common()),
            "question_type_counts": dict(question_type_counts.most_common()),
            "problem_count": len(problems),
            "problem_examples": problems[:100],
        },
    )
    print(f"Wrote audit report to {args.output_path}")


if __name__ == "__main__":
    main()
