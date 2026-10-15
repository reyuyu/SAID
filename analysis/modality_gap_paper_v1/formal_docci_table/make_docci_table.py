from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'DOCCI_LONG_DCI_COMPARISON.csv'
SOURCE_COMMIT = '691f0bbe5bfb46eebc2fbab52b5282d4cf873bba'

def main():
    df = pd.read_csv(CSV)
    # Use only the requested formal DOCCI rows and strategies.
    wanted = {
        ('longclip', 'full', 'historical'): 'Long-CLIP',
        ('ours_native', 'full', 'historical'): 'Sema-CLIP (Global)',
        ('ours_masked', 'full', 'historical'): 'Sema-CLIP (Masked)',
        ('longclip', 'random_distributed_random50', 'random_distributed'): 'Long-CLIP',
        ('ours_native', 'random_distributed_random50', 'random_distributed'): 'Sema-CLIP (Global)',
        ('ours_masked', 'random_distributed_random50', 'random_distributed'): 'Sema-CLIP (Masked)',
        ('longclip', 'random_distributed_random10', 'random_distributed'): 'Long-CLIP',
        ('ours_native', 'random_distributed_random10', 'random_distributed'): 'Sema-CLIP (Global)',
        ('ours_masked', 'random_distributed_random10', 'random_distributed'): 'Sema-CLIP (Masked)',
    }
    rows = {}
    for (model, view, strategy), label in wanted.items():
        hit = df[(df.dataset == 'docci') & (df.model == model) & (df.view == view) & (df.strategy == strategy)]
        assert len(hit) == 1, (model, view, strategy, len(hit))
        r = hit.iloc[0]
        assert int(r.valid_count) == 1000
        if strategy == 'random_distributed':
            assert r.seed == 'mean5'
            assert r.centroid_gap_seed_std == r.centroid_gap_seed_std  # not NaN
        else:
            assert r.seed == 'base'
        rows[(model, view)] = {'label': label, 'gap': float(r.centroid_gap), 'seed': str(r.seed), 'strategy': str(r.strategy), 'valid_count': int(r.valid_count)}

    order = [('longclip', 'Long-CLIP'), ('ours_native', 'Sema-CLIP (Global)'), ('ours_masked', 'Sema-CLIP (Masked)')]
    vals = {}
    for model, label in order:
        vals[label] = {
            'full': rows[(model, 'full')]['gap'],
            'random50': rows[(model, 'random_distributed_random50')]['gap'],
            'random10': rows[(model, 'random_distributed_random10')]['gap'],
        }
    # Verify minima directly from the formal CSV-derived values.
    minima = {k: min(v[k] for v in vals.values()) for k in ('full', 'random50', 'random10')}
    source_rows = []
    for label, d in vals.items():
        source_rows.append({'model': label, **d, 'full_source': rows[({'Long-CLIP':'longclip','Sema-CLIP (Global)':'ours_native','Sema-CLIP (Masked)':'ours_masked'}[label], 'full')]['strategy'], 'random_seed_aggregation': 'mean5'})
    pd.DataFrame(source_rows).to_csv(ROOT / 'modality_gap_docci_values.csv', index=False)

    table = r'''% Formal DOCCI modality-gap table.
% Source: SAID commit SOURCE_COMMIT, analysis/random_caption_gap_v1/results/DOCCI_LONG_DCI_COMPARISON.csv
% Directly insertable with \input{tables/modality_gap_docci}.
\definecolor{GapNavy}{HTML}{213C5A}
\definecolor{GapAccent}{HTML}{194F78}
\definecolor{GapTint}{HTML}{EDF3F8}
\begin{table}[t]
\centering
\caption{\textbf{Modality gap under different caption completeness levels on DOCCI.} The gap is measured by the Euclidean distance between image and text centroids in the original 512-dimensional embedding space. Lower is better.}
\label{tab:modality_gap}
\vspace{2pt}
\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.19}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}l*{3}{c}@{}}
\toprule[1pt]
\textbf{Model} & \multicolumn{3}{c}{\textbf{Modality Gap $\downarrow$}} \\
\cmidrule(lr){2-4}
& \textbf{Full} & \textbf{Random 50\%} & \textbf{Random 10\%} \\
\midrule[0.55pt]
Long-CLIP & LONG_FULL & LONG_R50 & LONG_R10 \\
Sema-CLIP (Global) & NAT_FULL & NAT_R50 & NAT_R10 \\
\addlinespace[1pt]
\rowcolor{GapTint}
\textbf{Sema-CLIP (Masked)} & MASK_FULL & MASK_R50 & MASK_R10 \\
\bottomrule[1pt]
\end{tabular}}
\endgroup

\vspace{2pt}
\parbox{\columnwidth}{\scriptsize\emph{Note.} Random 50\% and Random 10\% use the mean over five random seeds under the \texttt{random\_distributed} strategy. Full uses the historical fixed-sample result.}
\end{table}
'''
    table = table.replace('SOURCE_COMMIT', SOURCE_COMMIT)
    repl = {
        'LONG_FULL': f"{vals['Long-CLIP']['full']:.4f}", 'LONG_R50': f"{vals['Long-CLIP']['random50']:.4f}", 'LONG_R10': f"{vals['Long-CLIP']['random10']:.4f}",
        'NAT_FULL': f"{vals['Sema-CLIP (Global)']['full']:.4f}", 'NAT_R50': f"{vals['Sema-CLIP (Global)']['random50']:.4f}", 'NAT_R10': f"{vals['Sema-CLIP (Global)']['random10']:.4f}",
        'MASK_FULL': f"\\textcolor{{GapAccent}}{{\\textbf{{{vals['Sema-CLIP (Masked)']['full']:.4f}}}}}", 'MASK_R50': f"\\textcolor{{GapAccent}}{{\\textbf{{{vals['Sema-CLIP (Masked)']['random50']:.4f}}}}}", 'MASK_R10': f"\\textcolor{{GapAccent}}{{\\textbf{{{vals['Sema-CLIP (Masked)']['random10']:.4f}}}}}",
    }
    for a, b in repl.items(): table = table.replace(a, b)
    (ROOT / 'modality_gap_docci.tex').write_text(table, encoding='utf-8')
    (ROOT / 'modality_gap_docci_standalone.tex').write_text(r'''\documentclass[10pt]{article}
\usepackage[margin=0.25in]{geometry}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{array}
\pagestyle{empty}
\begin{document}
\input{modality_gap_docci.tex}
\end{document}
''', encoding='utf-8')
    (ROOT / 'modality_gap_docci_verification.json').write_text(json.dumps({'source_commit': SOURCE_COMMIT, 'source_csv': str(CSV), 'rows_used': source_rows, 'column_minima': minima, 'all_random_rows': 'random_distributed with seed mean5', 'all_valid_counts': 1000}, indent=2), encoding='utf-8')
    print(pd.DataFrame(source_rows).to_string(index=False))

if __name__ == '__main__':
    main()
