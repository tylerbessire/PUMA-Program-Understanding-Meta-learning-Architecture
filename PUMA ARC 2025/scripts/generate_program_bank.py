#!/usr/bin/env python3
"""Batch-generate ARC DSL programs with Gemini 1.5 Flash.

Implements Epic 1 actions from AGENTS.md:
- Load training tasks
- Prompt Gemini with RFT-oriented instructions
- Capture reasoning + DSL output for downstream verification
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - dependency checked at runtime
    raise SystemExit(
        "google-generativeai package not installed. Install with `pip install google-generativeai`."
    ) from exc

SYSTEM_INSTRUCTION = """You are the PUMA ARC synthesis agent. Analyse Abstraction and Reasoning Corpus
training tasks through an explicit Relational Frame Theory (RFT) lens. Decompose patterns in terms
of causality, spatial relations, object identity, comparison, and transformation rules. Then author a
DSL program compatible with arc_solver/dsl.py that reproduces the mapping from every training input
to its expected output. Always follow the required response template."""

PROGRAM_GUIDE = """Available tooling (non-executable, for conceptual grounding):
- object_reasoning.py: connected components, colour histograms, bounding boxes.
- object_tracker.py: persisting object identities across grids and time steps.
- heuristics_complete.py: reusable relational heuristics such as symmetry detection,
  counting, flood-filling, fill-largest-object, extend-lines, and rare colour expansion.
- dsl.py: canonical DSL primitives (paint, copy, translate, filter, rotate, reflect, compose,
  map_objects, conditional). Your emitted program must adhere strictly to the DSL signatures.
- rft_engine/rft/: Relational Frame Theory utilities (generate variations, promote rules).

Response requirements:
1. Begin with a section titled `## Reasoning` summarising the RFT analysis in no more than 6 bullet points. Each bullet must be under 20 words.
2. Follow with a section titled `## DSL Code` containing only the final program inside a Python code
   fence, e.g.
```
## DSL Code
```python
def solve(task):
    ...
```
```
3. Keep total response under 600 tokens—omit prompt restatements or redundant narration.
4. Use deterministic, fully-specified DSL constructs—do not place ellipses or pseudo-code.
5. Ensure the program when executed will produce the correct outputs for every training example.
"""

GENERATION_CONFIG: Dict[str, Any] = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 50,
    "candidate_count": 1,
    "max_output_tokens": 4096,
}

RATE_LIMIT_SECONDS = 10.0  # politeness delay between API calls


@dataclass
class TaskExample:
    """Representation of a single ARC task."""

    task_id: str
    train: List[Dict[str, Any]]
    test: List[Dict[str, Any]]


def load_training_tasks(challenges_path: Path, solutions_path: Optional[Path] = None) -> Dict[str, TaskExample]:
    """Load ARC training challenges and optional solutions."""
    with challenges_path.open("r", encoding="utf-8") as fh:
        challenges = json.load(fh)

    solutions: Dict[str, List[Any]] = {}
    if solutions_path:
        if solutions_path.exists():
            with solutions_path.open("r", encoding="utf-8") as fh:
                raw_solutions = json.load(fh)
            for task_id, records in raw_solutions.items():
                # Kaggle solutions store test outputs as list aligned with test inputs
                solutions[task_id] = records
        else:
            print(f"⚠ Solutions file not found: {solutions_path}. Proceeding without solutions.")

    examples: Dict[str, TaskExample] = {}
    for task_id, payload in challenges.items():
        train_entries = payload.get("train", [])
        test_entries = payload.get("test", [])
        # If solutions provided and equal length, attach outputs when missing
        if solutions_path and task_id in solutions:
            sol_records = solutions[task_id]
            for idx, entry in enumerate(test_entries):
                if "output" not in entry and idx < len(sol_records):
                    entry = dict(entry)
                    entry["output"] = sol_records[idx]
                    test_entries[idx] = entry
        examples[task_id] = TaskExample(task_id=task_id, train=train_entries, test=test_entries)
    return examples


def format_grid(grid: List[List[int]]) -> str:
    """Render a grid as space-separated rows."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def format_task_prompt(example: TaskExample) -> str:
    """Assemble the user-facing prompt for a single task."""
    lines: List[str] = []
    lines.append("You are analysing ARC task {task_id}.".format(task_id=example.task_id))
    lines.append("")
    lines.append(PROGRAM_GUIDE)
    lines.append("")
    lines.append("Training pairs:")
    for idx, pair in enumerate(example.train, start=1):
        lines.append(f"- Pair {idx} Input:\n{format_grid(pair['input'])}")
        lines.append(f"  Pair {idx} Output:\n{format_grid(pair['output'])}")
    if example.test:
        lines.append("")
        lines.append("Test inputs (predictions required in final program):")
        for idx, pair in enumerate(example.test, start=1):
            lines.append(f"- Test {idx} Input:\n{format_grid(pair['input'])}")
            if "output" in pair:
                lines.append(f"  (Reference output available for validation)\n{format_grid(pair['output'])}")
    lines.append("")
    lines.append("Deliver the required response structure with `## Reasoning` followed by `## DSL Code`.")
    return "\n".join(lines)


SECTION_PATTERN = re.compile(
    r"##\s*Reasoning(?P<reasoning>.*?)(?:##\s*DSL Code|\Z)",
    re.IGNORECASE | re.DOTALL,
)
CODE_PATTERN = re.compile(
    r"##\s*DSL Code(?P<code>.*)",
    re.IGNORECASE | re.DOTALL,
)
CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python)?\s*(?P<body>.*?)```",
    re.DOTALL,
)


def extract_text(response: Any) -> Tuple[Optional[str], List[Optional[str]]]:
    """Return concatenated candidate text and finish reasons."""
    if response is None:
        return None, []

    candidates = getattr(response, "candidates", None)
    collected: List[str] = []
    reasons: List[Optional[str]] = []

    if candidates:
        for candidate in candidates:
            reasons.append(getattr(candidate, "finish_reason", None))
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for part in parts:
                    text_value = getattr(part, "text", None)
                    if text_value:
                        collected.append(text_value)
        if collected:
            return "\n".join(collected).strip(), reasons

    text_attr = None
    try:
        text_attr = response.text
    except Exception:
        text_attr = None
    if text_attr:
        return text_attr, reasons

    try:
        return str(response), reasons
    except Exception:
        return None, reasons


def extract_sections(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract reasoning and DSL code blocks from the model output."""
    reasoning: Optional[str] = None
    code: Optional[str] = None

    reasoning_match = SECTION_PATTERN.search(text)
    if reasoning_match:
        reasoning = reasoning_match.group("reasoning").strip()

    code_match = CODE_PATTERN.search(text)
    if code_match:
        code_section = code_match.group("code").strip()
        block_match = CODE_BLOCK_PATTERN.search(code_section)
        if block_match:
            code = block_match.group("body").strip()
        else:
            code = code_section

    return reasoning, code


def load_completed_ids(output_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read existing log entries to support resuming."""
    completed: Dict[str, Dict[str, Any]] = {}
    if not output_path.exists():
        return completed
    text = output_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        if text[idx] in ("\n", "\r", " "):
            idx += 1
            continue
        try:
            record, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        task_id = record.get("task_id")
        if task_id:
            completed[task_id] = record
        newline = text.find("\n", end)
        if newline == -1:
            idx = length
        else:
            idx = newline + 1
    return completed


def run_generation(
    examples: Iterable[TaskExample],
    model_name: str,
    api_key: str,
    output_path: Path,
    delay: float,
    resume: bool,
    retry_incomplete: bool,
    max_tasks: Optional[int] = None,
) -> None:
    """Stream tasks through Gemini and record the responses."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_INSTRUCTION)

    completed = load_completed_ids(output_path) if resume else {}
    processed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as writer:
        for example in examples:
            if resume:
                prev = completed.get(example.task_id)
                if prev:
                    if prev.get("status") == "generated":
                        continue
                    if not retry_incomplete:
                        continue
            if max_tasks is not None and processed >= max_tasks:
                break

            prompt = format_task_prompt(example)
            status = "generated"
            reasoning_text: Optional[str] = None
            code_text: Optional[str] = None
            raw_text: Optional[str] = None
            error_message: Optional[str] = None
            finish_reasons: List[Optional[str]] = []
            attempts = 0

            for attempt in range(2):
                attempts = attempt + 1
                require_reasoning = attempt == 0
                instructions = ["You must follow the instructions exactly."]
                if attempt == 1:
                    instructions.append(
                        "Second attempt: respond ONLY with the `## DSL Code` section. No reasoning,"
                        " comments, or narration. Keep the code under 200 tokens."
                    )
                instructions.append(prompt)

                needs_retry = False
                throttle_retries = 0
                while True:
                    throttle_retries += 1
                    attempt_raw: Optional[str] = None
                    attempt_finish: List[Optional[str]] = []
                    attempt_reasoning: Optional[str] = None
                    attempt_code: Optional[str] = None

                    try:
                        response = model.generate_content(
                            instructions,
                            generation_config=GENERATION_CONFIG,
                        )
                        attempt_raw, attempt_finish = extract_text(response)
                        attempt_reasoning, attempt_code = extract_sections(attempt_raw or "")

                        has_code = bool(attempt_code and attempt_code.strip())
                        has_reasoning = bool(attempt_reasoning and attempt_reasoning.strip())

                        if has_code and (has_reasoning or not require_reasoning):
                            raw_text = attempt_raw
                            finish_reasons = attempt_finish
                            code_text = attempt_code
                            reasoning_text = attempt_reasoning or (
                                "Omitted per code-only fallback directive."
                                if not require_reasoning
                                else None
                            )
                            status = "generated"
                            break

                        status = "incomplete"
                        raw_text = attempt_raw
                        finish_reasons = attempt_finish
                        if not has_reasoning:
                            reasoning_text = attempt_raw
                        if not has_code:
                            code_text = ""

                        needs_retry = (
                            attempt == 0
                            and attempt_finish
                            and any(
                                reason == "MAX_TOKENS"
                                for reason in attempt_finish
                                if reason
                            )
                        )
                        if needs_retry:
                            wait_seconds = max(delay, RATE_LIMIT_SECONDS)
                            time.sleep(wait_seconds)
                        break

                    except Exception as exc:  # pragma: no cover - network errors only at runtime
                        error_message = f"{type(exc).__name__}: {exc}"
                        if (
                            any(token in error_message for token in ("ResourceExhausted", "429"))
                            and throttle_retries < 4
                        ):
                            wait_seconds = 15 * throttle_retries
                            print(
                                f"Rate limit hit; sleeping {wait_seconds:.1f}s "
                                f"(retry {throttle_retries})."
                            )
                            time.sleep(wait_seconds)
                            continue

                        status = "error"
                        finish_reasons = []
                        raw_text = attempt_raw
                        break

                if status == "generated":
                    break
                if status == "incomplete" and needs_retry and attempt == 0:
                    continue
                if status in ("incomplete", "error"):
                    break

            if code_text is None:
                code_text = ""
            if reasoning_text is None and raw_text is not None and status != "generated":
                reasoning_text = raw_text

            record: Dict[str, Any] = {
                "task_id": example.task_id,
                "status": status,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reasoning": reasoning_text,
                "dsl_program": code_text,
                "raw_response": raw_text,
            }
            record["attempts"] = attempts
            if error_message:
                record["error"] = error_message
            if finish_reasons:
                record["finish_reasons"] = finish_reasons

            writer.write(json.dumps(record) + "\n")
            writer.flush()
            processed += 1

            summary = status.upper()
            if status == "generated":
                summary += " ✅"
            elif status == "incomplete":
                summary += " ⚠"
            else:
                summary += " ✗"
            print(f"[{processed}] {example.task_id}: {summary}")
            time.sleep(delay)



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ARC DSL programs with Gemini")
    parser.add_argument(
        "--challenges",
        type=Path,
        default=Path("PUMA ARC 2025/PUMA/data/arc-agi_training_challenges.json"),
        help="Path to arc-agi_training_challenges.json",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=Path("PUMA ARC 2025/PUMA/data/arc-agi_training_solutions.json"),
        help="Optional path to arc-agi_training_solutions.json for supervised outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PUMA ARC 2025/artifacts/master_solution_log.jsonl"),
        help="Output JSONL file to append generation records",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gemini-2.0-flash",
        help="Gemini model name (e.g., models/gemini-2.0-flash, models/gemini-1.5-pro-latest)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("GEMINI_API_KEY"),
        help="Gemini API key (or set GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit the number of tasks processed",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks already present in the output file",
    )
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="When used with --resume, re-run tasks whose prior status isn't 'generated'",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=RATE_LIMIT_SECONDS,
        help="Delay between API calls in seconds",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not args.api_key:
        raise SystemExit("Gemini API key not provided. Use --api-key or set GEMINI_API_KEY.")

    tasks = load_training_tasks(args.challenges, args.solutions)
    ordered_examples = [tasks[key] for key in sorted(tasks.keys())]

    run_generation(
        examples=ordered_examples,
        model_name=args.model,
        api_key=args.api_key,
        output_path=args.output,
        delay=args.delay,
        resume=args.resume,
        retry_incomplete=args.retry_incomplete,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
