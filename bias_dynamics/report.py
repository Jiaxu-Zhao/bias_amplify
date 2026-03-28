from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def build_markdown_report(output_dir: Path) -> Path:
    lines = ["# Bias Dynamics Report", ""]
    for fp in sorted(output_dir.glob("*_summary.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        lines.append(f"## {data['method']}")
        lines.append(f"- iterations: {data['iterations']}")
        lines.append(f"- final_bias_mean: {data['final_bias_mean']:.4f}")
        lines.append(f"- final_bias_variance: {data['final_bias_variance']:.6f}")
        lines.append("")

    inj = output_dir / "self_distill_bias_injection.json"
    if inj.exists():
        rows: List[Dict] = json.loads(inj.read_text(encoding="utf-8"))
        lines.append("## Bias Injection Sweep (Self-distill)")
        for row in rows:
            lines.append(f"- alpha={row['alpha']}: bias={row['bias_score']}, BAR={row['BAR']}")

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
