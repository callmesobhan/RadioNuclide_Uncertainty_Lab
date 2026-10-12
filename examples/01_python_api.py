"""The multi-organ reference scenario, driven from Python instead of the command line.

Run from the repository root:  python examples/01_python_api.py
"""

from RadioNuclide_Uncertainty_Lab import analyze, load_config, matrix_markdown

cfg = load_config("examples/configs/lu177_dotatate_multi_organ.yaml")
analysis = analyze(cfg.model, cfg.run, config_sha256=cfg.sha256)

print(cfg.model.name, "\n")
print(f"{'output':<20}{'mean ± SD':>18}{'CV':>8}   95% interval")
for name, s in analysis.summaries.items():
    print(
        f"{name:<20}{f'{s.mean:.3g} ± {s.sd:.2g} {s.unit}':>18}{s.cv:>8.1%}   "
        f"[{s.ci_low:.3g}, {s.ci_high:.3g}]"
    )

print("\nTotal-order variance share, factors x outputs:\n")
print(matrix_markdown(analysis.matrix))

# Shared systematic errors move organ doses together but cancel in their ratio.
r = analysis.result
print(f"\ncorr(kidneys, liver) = {r.correlation('kidneys', 'liver'):+.2f}  (shared errors)")
print(
    f"CV(kidneys) = {analysis.summaries['kidneys'].cv:.1%},  "
    f"CV(kidneys/liver) = {analysis.summaries['kidneys_to_liver'].cv:.1%}"
)
