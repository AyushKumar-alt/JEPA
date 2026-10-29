"""
Downstream Decision Dataset Builder for the BEV Decision Mix.
Parses CHOICE, NOUL, and SCORE tasks into structured decision instances.
Enforces strict label isolation: inputs contain only legitimate decision context,
and targets are stored separately.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import random
import hashlib
import pyarrow.parquet as pq


@dataclass
class DownstreamExample:
    example_id: str
    state_id: str
    state_hash: str
    domain: str
    task_type: str  # 'choice', 'noul', 'score'
    input_text: str  # Legitimate context: state + question + candidate descriptions
    target_idx: int  # Ground truth integer target (never inside input_text)
    num_classes: int  # Number of valid options/classes
    option_labels: List[str]  # Human-readable option names


def format_downstream_prompt(
    state_text: str,
    instructions: str,
    task_type: str,
    criteria: Any,
) -> Tuple[str, List[str], Optional[Dict[str, int]]]:
    """
    Constructs the input context legitimately available to a decision model.
    CRITICAL: Does NOT include the ground truth label or any answer key indicator.
    """
    state_clean = state_text.strip()
    instr_clean = instructions.strip() if instructions else "Choose the correct decision."
    option_names: List[str] = []
    option_to_idx: Dict[str, int] = {}

    if task_type == "choice":
        # Criteria is typically a dict mapping key -> description
        if isinstance(criteria, dict):
            options_text_parts = []
            for idx, (opt_key, opt_desc) in enumerate(criteria.items()):
                option_names.append(opt_key)
                option_to_idx[opt_key] = idx
                options_text_parts.append(f"({chr(65 + idx)}) {opt_key}: {opt_desc}")
            options_str = " | ".join(options_text_parts)
        elif isinstance(criteria, list):
            options_text_parts = []
            for idx, opt_item in enumerate(criteria):
                opt_str = str(opt_item)
                option_names.append(opt_str)
                option_to_idx[opt_str] = idx
                options_text_parts.append(f"({chr(65 + idx)}) {opt_str}")
            options_str = " | ".join(options_text_parts)
        else:
            options_str = ""

        prompt = (
            f"State: {state_clean}\n"
            f"Question: {instr_clean}\n"
            f"Candidate Options: {options_str}"
        )

    elif task_type == "noul":
        # Binary boolean judgment (Yes / No)
        option_names = ["No (False)", "Yes (True)"]
        option_to_idx = {"false": 0, "true": 1, "0": 0, "1": 1}
        prompt = (
            f"State: {state_clean}\n"
            f"Proposition: {instr_clean}\n"
            f"Candidates: (A) No (False) | (B) Yes (True)"
        )

    elif task_type == "score":
        # Ordered score on scale 0 to K-1
        if isinstance(criteria, list):
            options_text_parts = []
            for idx, lvl_desc in enumerate(criteria):
                option_names.append(str(lvl_desc))
                option_to_idx[str(idx)] = idx
                options_text_parts.append(f"{idx}: {lvl_desc}")
            scale_str = " | ".join(options_text_parts)
        else:
            scale_str = "0 to 4 ordered rating"
            option_names = [str(i) for i in range(5)]
            option_to_idx = {str(i): i for i in range(5)}

        prompt = (
            f"State: {state_clean}\n"
            f"Evaluation Criterion: {instr_clean}\n"
            f"Score Scale: {scale_str}"
        )
    else:
        prompt = f"State: {state_clean}\nQuestion: {instr_clean}"

    return prompt, option_names, option_to_idx


def parse_downstream_dataset(
    parquet_path: Path,
    min_state_words: int = 4,
) -> List[DownstreamExample]:
    """
    Parses parquet file into a list of DownstreamExample objects.
    Extracts CHOICE, NOUL, and SCORE questions without leakage.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(str(parquet_path))
    states = table["state"].to_pylist()
    q_jsons = table["questions_json"].to_pylist()
    domains = table["domain"].to_pylist() if "domain" in table.column_names else ["unknown"] * len(states)

    examples: List[DownstreamExample] = []

    for row_idx, (raw_state, qj_str, domain) in enumerate(zip(states, q_jsons, domains)):
        if not raw_state or not qj_str:
            continue

        state_text = str(raw_state).strip()
        if len(state_text.split()) < min_state_words:
            continue

        state_hash = hashlib.md5(state_text.encode("utf-8")).hexdigest()
        state_id = f"row_{row_idx}_{state_hash[:8]}"

        try:
            q_dict = json.loads(qj_str)
        except Exception:
            continue

        if not isinstance(q_dict, dict):
            continue

        for q_key, q_val in q_dict.items():
            if not isinstance(q_val, dict):
                continue

            task_type = str(q_val.get("type", "")).lower()
            if task_type not in ["choice", "noul", "score"]:
                continue

            instructions = str(q_val.get("instructions", ""))
            criteria = q_val.get("criteria")
            raw_label = q_val.get("label")

            if raw_label is None:
                continue

            # Format input prompt (NO LABELS)
            prompt, opt_names, opt_to_idx = format_downstream_prompt(
                state_text=state_text,
                instructions=instructions,
                task_type=task_type,
                criteria=criteria,
            )

            # Map raw label to ground truth integer target
            target_idx = None
            if task_type == "choice":
                label_str = str(raw_label).strip()
                if opt_to_idx and label_str in opt_to_idx:
                    target_idx = opt_to_idx[label_str]
                elif label_str.isdigit():
                    target_idx = int(label_str)

            elif task_type == "noul":
                if isinstance(raw_label, bool):
                    target_idx = 1 if raw_label else 0
                elif str(raw_label).lower() in ["true", "yes", "1"]:
                    target_idx = 1
                elif str(raw_label).lower() in ["false", "no", "0"]:
                    target_idx = 0

            elif task_type == "score":
                try:
                    target_idx = int(raw_label)
                except (ValueError, TypeError):
                    pass

            if target_idx is None or target_idx < 0:
                continue

            num_classes = len(opt_names) if opt_names else (5 if task_type == "score" else 2)
            if target_idx >= num_classes and task_type != "score":
                continue

            example_id = f"{state_id}_{q_key}"
            examples.append(DownstreamExample(
                example_id=example_id,
                state_id=state_id,
                state_hash=state_hash,
                domain=domain,
                task_type=task_type,
                input_text=prompt,
                target_idx=target_idx,
                num_classes=num_classes,
                option_labels=opt_names,
            ))

    return examples


def create_label_efficiency_subsets(
    examples: List[DownstreamExample],
    fractions: List[float] = [0.10, 0.25, 0.50, 1.00],
    seed: int = 42,
) -> Dict[float, List[DownstreamExample]]:
    """
    Creates nested training subsets defined at the state-group level.
    Guarantees that multiple questions from the same state row remain together.
    """
    rng = random.Random(seed)

    # Group examples by unique state_hash
    state_to_examples: Dict[str, List[DownstreamExample]] = {}
    for ex in examples:
        if ex.state_hash not in state_to_examples:
            state_to_examples[ex.state_hash] = []
        state_to_examples[ex.state_hash].append(ex)

    unique_hashes = sorted(list(state_to_examples.keys()))
    rng.shuffle(unique_hashes)
    total_states = len(unique_hashes)

    subsets: Dict[float, List[DownstreamExample]] = {}
    sorted_fractions = sorted(fractions)

    for frac in sorted_fractions:
        num_states = max(1, int(round(total_states * frac)))
        selected_hashes = set(unique_hashes[:num_states])

        subset_examples: List[DownstreamExample] = []
        for h in selected_hashes:
            subset_examples.extend(state_to_examples[h])

        subsets[frac] = subset_examples

    return subsets
