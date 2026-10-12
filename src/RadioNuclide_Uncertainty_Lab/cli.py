"""Command line: ``radionuclide-uncertainty-lab run config.yaml --out results/``.

The short alias ``rnulab`` runs the same entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import LoadedConfig, load_config
from .models import MODEL_BUILDERS
from .report import to_dict, to_markdown
from .workflow import analyze


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="radionuclide-uncertainty-lab", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"RadioNuclide_Uncertainty_Lab {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="propagate uncertainty and write a report")
    run.add_argument("config", type=Path)
    run.add_argument("--out", type=Path, default=Path("results"))
    run.add_argument("--n", type=int, help="override run.n (Monte Carlo samples)")
    run.add_argument("--sobol-n", type=int, help="override run.sobol_n (Sobol' base sample size)")
    run.add_argument("--seed", type=int, help="override run.seed")
    run.add_argument("--no-sobol", action="store_true", help="skip variance-based indices")
    run.add_argument("--no-plots", action="store_true")

    val = sub.add_parser("validate", help="check a config and print its inputs and outputs")
    val.add_argument("config", type=Path)

    sub.add_parser("models", help="list available model names")
    sub.add_parser(
        "check-data",
        help="run the consistency checks on published data and reference datasets",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            print("\n".join(sorted(MODEL_BUILDERS)))
            return 0
        if args.command == "check-data":
            from .datacheck import get_registry

            registry = get_registry()
            print(registry.format_report())
            return 0 if registry.ok else 1
        cfg = load_config(args.config)
        if args.command == "validate":
            model = cfg.model
            print(f"OK  {model.name}  (config sha256 {cfg.sha256[:12]})")
            print(f"outputs: {', '.join(model.outputs)}")
            for r in model.inputs.describe():
                print(
                    f"  {r['name']:<46} group={r['group']:<24} median={r['median']:.4g} "
                    f"sd={r['sd']:.3g} {r['unit']}"
                )
            return 0
        return _run(args, cfg)
    except (ValueError, KeyError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, cfg: LoadedConfig) -> int:
    run = replace(
        cfg.run,
        n=args.n or cfg.run.n,
        sobol_n=args.sobol_n or cfg.run.sobol_n,
        seed=args.seed if args.seed is not None else cfg.run.seed,
    )
    if run.seed is None:
        print("warning: no seed set; results will not be reproducible", file=sys.stderr)
    model = cfg.model
    analysis = analyze(
        model,
        run,
        sobol=not args.no_sobol,
        config_sha256=cfg.sha256,
        data_integrity=cfg.data_integrity(),
    )
    for d in analysis.diagnostics:
        if d.level == "warning":
            print(f"warning [{d.code}]: {d.message}", file=sys.stderr)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(to_dict(analysis), indent=2) + "\n")
    (out / "report.md").write_text(to_markdown(analysis))
    written = ["result.json", "report.md"]

    if not args.no_plots:
        try:
            from . import plots

            written += [str(p.relative_to(out)) for p in plots.write_figures(analysis, out)]
        except ImportError:
            print("note: matplotlib not installed; skipping plots", file=sys.stderr)

    print(model.name)
    for o in model.outputs:
        s = analysis.summaries[o]
        print(
            f"  {o:<22}{s.mean:>9.4g} ± {s.sd:<8.3g}{s.unit:<4} "
            f"({s.level:.0%} interval {s.ci_low:.4g}–{s.ci_high:.4g}; CV {s.cv:.1%})"
        )
    if analysis.budgets:
        for o in model.outputs:
            top = ", ".join(f"{ln.factor} {ln.st:.0%}" for ln in analysis.budgets[o].lines[:3])
            print(f"  largest sources for {o}: {top}")
    print(f"wrote {len(written)} files to {out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
