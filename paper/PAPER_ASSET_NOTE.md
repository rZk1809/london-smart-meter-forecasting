# Paper Asset Note

This note documents the provenance of the assets used in `paper/ieee_small_sample_hybrid_paper.tex`.

## Result tables

- The main benchmark table was populated from `results/final/metrics_table.csv`.
- The hybrid comparison table was populated from:
  - `results/harp/harpv2_ablation_results.csv`
  - `results/harp/care_ablation_results.csv`
- The slice table and discussion values were populated from `results/harp/experiment_summary.json`.

## Figures reused from validated repository outputs

The paper uses copied high-resolution figures from `outputs/figures/harp`:

- `paper/figures/benchmark_leaderboard.png`
- `paper/figures/benchmark_heatmap.png`
- `paper/figures/hard_case_slice.png`
- `paper/figures/high_volatility_slice.png`
- `paper/figures/care_confidence_vs_gain.png`
- `paper/figures/retrieval_summary.png`
- `paper/figures/prototype_retrieval_case_1.png`
- `paper/figures/harp_residual_contribution.png`

The final paper source currently embeds:

- `benchmark_leaderboard.png`
- `benchmark_heatmap.png`
- `hard_case_slice.png`
- `high_volatility_slice.png`
- `care_confidence_vs_gain.png`
- `retrieval_summary.png`

The remaining copied figures stay in `paper/figures` as supplementary paper-ready assets and can be used in future revisions if an even longer version is needed.

## Regenerated artifacts

- `paper/ieee_small_sample_hybrid_paper.tex` was expanded into a longer, more detailed IEEE two-column paper with deeper explanations in each section.
- `paper/ieee_small_sample_hybrid_paper.pdf` was compiled locally from the LaTeX source.

## Reporting policy

- All reported metrics were copied from validated repository CSV/JSON artifacts.
- No metric values were estimated, interpolated, or manually adjusted for the paper.
- The paper is written with an honest negative-result stance: CatBoost is reported as the strongest overall model, while DynamicRetrievalResidual, HARP-v2, and CARE are reported as exploratory residual refinements that did not surpass the backbone on held-out RMSE.
