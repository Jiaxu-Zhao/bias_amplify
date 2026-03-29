from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Dict, Iterable, List, Tuple


def _find_first(row: Dict, keys: Iterable[str]) -> str:
    for k in keys:
        if k in row and row[k]:
            return str(row[k])
    return ""




def _extract_stereoset_triplet(row: Dict) -> Tuple[str, str, str]:
    if "sentences" not in row or not isinstance(row["sentences"], list):
        return "", "", ""

    stereo = ""
    anti = ""
    for item in row["sentences"]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("gold_label", "")).lower()
        text = str(item.get("sentence", "")).strip()
        if not text:
            continue
        if "stereo" in label and "anti" not in label:
            stereo = text
        elif "anti" in label:
            anti = text

    if not anti or not stereo:
        return "", "", ""

    context = str(row.get("context", "")).strip()
    target = str(row.get("target", "")).strip()
    prompt = context or (f"Complete the sentence about {target}:" if target else "Complete the sentence:")
    return prompt, anti, stereo

def extract_triplet(row: Dict) -> Tuple[str, str, str]:
    sp, sc, sr = _extract_stereoset_triplet(row)
    if sp and sc and sr:
        return sp, sc, sr

    prompt = _find_first(row, ["prompt", "instruction", "question"])
    chosen = _find_first(row, ["chosen", "response_chosen", "preferred"])
    rejected = _find_first(row, ["rejected", "response_rejected", "dispreferred"])

    if not prompt and "messages" in row and isinstance(row["messages"], list) and row["messages"]:
        user_turns = [m.get("content", "") for m in row["messages"] if m.get("role") in {"user", "human"}]
        if user_turns:
            prompt = user_turns[-1]

    if not prompt and isinstance(row.get("chosen"), str):
        parts = row["chosen"].split("\n\nAssistant:")
        if parts:
            prompt = parts[0].strip()

    if not chosen and isinstance(row.get("chosen"), str):
        chosen = row["chosen"].strip()
    if not rejected and isinstance(row.get("rejected"), str):
        rejected = row["rejected"].strip()

    return prompt.strip(), chosen.strip(), rejected.strip()


def _sample(rows: List[Dict], max_samples: int, seed: int) -> List[Dict]:
    if len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    keep = set(idx[:max_samples])
    return [rows[i] for i in range(len(rows)) if i in keep]


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_one_experiment(exp: Dict, output_dir: Path, seed: int) -> Dict:
    from datasets import load_dataset

    name = exp["name"]
    source_dataset = exp["source_dataset"]
    source_split = exp["source_split"]
    max_samples = int(exp["max_samples"])

    raw = load_dataset(source_dataset, split=source_split)
    rows = _sample([dict(r) for r in raw], max_samples=max_samples, seed=seed)

    sft_rows: List[Dict] = []
    dpo_rows: List[Dict] = []
    rlhf_rows: List[Dict] = []
    self_distill_prompts: List[Dict] = []

    skipped = 0
    for row in rows:
        prompt, chosen, rejected = extract_triplet(row)
        if not prompt:
            skipped += 1
            continue

        self_distill_prompts.append({"prompt": prompt})

        if chosen:
            sft_rows.append({"prompt": prompt, "response": chosen})

        if chosen and rejected:
            pair = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            dpo_rows.append(pair)
            rlhf_rows.append(pair)

    exp_out = output_dir / name
    _write_jsonl(exp_out / "sft.jsonl", sft_rows)
    _write_jsonl(exp_out / "dpo.jsonl", dpo_rows)
    _write_jsonl(exp_out / "rlhf_pref.jsonl", rlhf_rows)
    _write_jsonl(exp_out / "self_distill_prompts.jsonl", self_distill_prompts)

    summary = {
        "experiment": name,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "rationale": exp.get("rationale", ""),
        "raw_rows": len(rows),
        "skipped_rows": skipped,
        "sft_rows": len(sft_rows),
        "dpo_rows": len(dpo_rows),
        "rlhf_rows": len(rlhf_rows),
        "self_distill_prompts": len(self_distill_prompts),
        "outputs": {
            "sft": str(exp_out / "sft.jsonl"),
            "dpo": str(exp_out / "dpo.jsonl"),
            "rlhf": str(exp_out / "rlhf_pref.jsonl"),
            "self_distill": str(exp_out / "self_distill_prompts.jsonl"),
        },
    }
    (exp_out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run(protocol_path: str, output_dir: str, seed: int = 42) -> List[Dict]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for exp in protocol["experiments"]:
        summaries.append(build_one_experiment(exp, out, seed=seed))

    (out / "all_experiments_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summaries
