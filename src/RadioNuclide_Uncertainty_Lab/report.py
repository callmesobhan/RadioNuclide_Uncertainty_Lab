"""Human- and machine-readable reports with provenance."""

from __future__ import annotations

from typing import Any

from .budget import budget_markdown, matrix_markdown
from .workflow import Analysis

DISCLAIMER = "> Research tool. Not validated for clinical decision-making."


def to_dict(a: Analysis) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": a.model.name,
        "outputs": {
            o: {"unit": a.model.unit(o), "summary": a.summaries[o].as_dict()}
            for o in a.model.outputs
        },
        "inputs": a.model.inputs.describe(),
        "rank_sensitivity": {
            o: {
                "r2_linear": r.r2,
                "parameters": [
                    {
                        "name": n,
                        "spearman": float(r.spearman[i]),
                        "prcc": float(r.prcc[i]),
                        "src": float(r.src[i]),
                    }
                    for i, n in enumerate(r.names)
                ],
            }
            for o, r in a.ranks.items()
        },
        "provenance": a.provenance,
        "diagnostics": [
            {"level": d.level, "code": d.code, "message": d.message} for d in a.diagnostics
        ],
        "data_integrity": a.data_integrity,
    }
    if a.budgets is not None and a.matrix is not None:
        out["error_budget"] = {
            o: {
                "output_mean": b.output_mean,
                "output_sd": b.output_sd,
                "interaction_share": b.interaction_share,
                "confidence": b.conf,
                "factors": [
                    {
                        "factor": ln.factor,
                        "members": list(ln.members),
                        "s1": ln.s1,
                        "s1_ci": list(ln.s1_ci),
                        "st": ln.st,
                        "st_ci": list(ln.st_ci),
                        "sd_first": ln.sd_first,
                        "sd_total": ln.sd_total,
                        "rel_first": ln.rel_first,
                        "rel_total": ln.rel_total,
                    }
                    for ln in b.lines
                ],
            }
            for o, b in a.budgets.items()
        }
        out["budget_matrix_st"] = {
            "factors": list(a.matrix.factors),
            "outputs": list(a.matrix.outputs),
            "st": a.matrix.st.tolist(),
        }
    return out


def to_markdown(a: Analysis) -> str:
    m = a.model
    level = a.run.level
    parts = [f"# {m.name}", "", DISCLAIMER, "", "## Results", ""]
    parts += [
        f"| Output | Unit | Mean ± SD | CV | Median | {level:.0%} interval | Skew |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for o in m.outputs:
        s = a.summaries[o]
        parts.append(
            f"| {o} | {s.unit or '-'} | {s.mean:.4g} ± {s.sd:.3g} | {s.cv:.1%} | {s.median:.4g} | "
            f"[{s.ci_low:.4g}, {s.ci_high:.4g}] | {s.skewness:.2f} |"
        )
    worst_se = max(s.mean_se / abs(s.mean) for s in a.summaries.values())
    parts += [
        "",
        f"Monte Carlo standard error of each mean ≤ {worst_se:.2%}"
        " (iid formula; conservative for quasi-random sampling).",
        "",
    ]

    outputs = list(m.outputs)
    if len(outputs) > 1:
        parts += ["### Correlation between outputs", "", "Induced by shared inputs (Pearson).", ""]
        parts.append("| | " + " | ".join(outputs) + " |")
        parts.append("|---|" + "---:|" * len(outputs))
        for p in outputs:
            row = " | ".join(f"{a.result.correlation(p, q):+.2f}" for q in outputs)
            parts.append(f"| {p} | {row} |")
        parts.append("")

    parts += [
        "## Inputs",
        "",
        "| Parameter | Group | Unit | Median | Mean | SD | Source |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in m.inputs.describe():
        parts.append(
            f"| {r['name']} | {r['group']} | {r['unit'] or '-'} | {r['median']:.4g} | "
            f"{r['mean']:.4g} | {r['sd']:.3g} | {r['source'] or '-'} |"
        )

    if a.budgets is not None and a.matrix is not None:
        parts += ["", "## Error budget (variance-based)", ""]
        if len(outputs) > 1:
            parts += [
                "### Total-order share of variance, factors × outputs",
                "",
                matrix_markdown(a.matrix),
                "",
            ]
        for o in outputs:
            parts += [f"### {o}", "", budget_markdown(a.budgets[o], min_share=0.005), ""]

    parts += ["## Diagnostics", ""]
    if a.diagnostics:
        parts += [f"- **{d.level}** `{d.code}`: {d.message}" for d in a.diagnostics]
    else:
        parts += ["No issues detected by the automatic checks."]
    parts.append("")

    if a.data_integrity:
        di = a.data_integrity
        parts += [
            "## Data integrity",
            "",
            f"- Published claims relied on: {len(di['claims_used'])}"
            + (
                f" ({', '.join(di['claims_used'][:6])}"
                f"{', …' if len(di['claims_used']) > 6 else ''})"
                if di["claims_used"]
                else ""
            ),
            f"- Consistency checks: {di['checks_passed']} passed, "
            f"{di['checks_known_conflict']} known source discrepancies (their claims are excluded)",
            f"- Excluded claims (never used): {len(di['excluded_claims'])}",
        ]
        if di.get("s_value_dataset"):
            ds = di["s_value_dataset"]
            parts.append(
                f"- S-value dataset `{ds['dataset']}` (source fingerprint `{ds['fingerprint']}`); "
                f"cross-organ dose {'included' if ds['cross_dose'] else 'omitted'}"
            )
        parts.append("")

    parts += ["## Parameter screening (rank-based)", ""]
    for o in outputs:
        r = a.ranks[o]
        parts += [
            f"### {o}",
            "",
            f"Linear R² of output on inputs: {r.r2:.2f} "
            "(far below 1 means non-linearity or interactions; trust Sobol' over SRC). Top 6:",
            "",
            "| Parameter | Spearman | PRCC | SRC |",
            "|---|---:|---:|---:|",
        ]
        for name, _ in r.ranking("prcc")[:6]:
            i = r.names.index(name)
            parts.append(f"| {name} | {r.spearman[i]:+.3f} | {r.prcc[i]:+.3f} | {r.src[i]:+.3f} |")
        parts.append("")

    parts += ["## Provenance", ""]
    parts += [f"- {k}: `{v}`" for k, v in a.provenance.items()]
    return "\n".join(parts) + "\n"
