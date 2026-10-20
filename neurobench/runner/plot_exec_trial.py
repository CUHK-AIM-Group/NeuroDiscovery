"""Standalone plot of independently verified executable development outcomes."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def render(delivery, stem='COMPARISON_r2'):
    result = json.loads((delivery / 'RESULTS.json').read_text(encoding='utf-8'))
    models = ['gpt-5.6-sol', 'deepseek-v4-pro']
    arms = ['A', 'B0', 'B1']
    names = {'A': 'NR OFF', 'B0': 'NR Current', 'B1': 'NR Candidate'}
    cases = [f'E{i:02d}' for i in range(1, 13)]
    colors = ['#dc6666', '#e5af45', '#c0c4ca', '#36a486', '#6378c1']
    values, letters, labels = [], [], []
    for model in models:
        for arm in arms:
            rows = [next(r for r in result['rows'] if (r['model'], r['arm'], r['case_id']) == (model, arm, cid)) for cid in cases]
            line, text = [], []
            for row in rows:
                if row['final_pass']:
                    line.append(4 if row['input_kind'] == 'must_block' else 3)
                    text.append('B' if row['input_kind'] == 'must_block' else 'P')
                elif row.get('final_oracle_verdict') == 'ORACLE_ERROR':
                    line.append(1); text.append('E')
                elif row['status'] == 'SUBMITTED':
                    line.append(0); text.append('F')
                elif row['status'] == 'NOT_DISPATCHED':
                    line.append(2); text.append('-')
                else:
                    line.append(1); text.append('U' if row['status'] == 'UNKNOWN' else 'T' if row['status'] == 'BUDGET_STOP' else 'E')
            values.append(line); letters.append(text)
            labels.append(('GPT-5.6-Sol' if model == models[0] else 'DeepSeek V4 Pro') + ' / ' + names[arm])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, (ax, timing) = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={'height_ratios': [3, 1.5]})
    ax.imshow(values, cmap=ListedColormap(colors), vmin=-.5, vmax=4.5, aspect='auto')
    ax.set_xticks(range(12), cases); ax.set_yticks(range(6), labels)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 12, 1), minor=True); ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2); ax.tick_params(which='minor', bottom=False, left=False)
    ax.axhline(2.5, color='#23364a', linewidth=2)
    ax.axvline(10.5, color='#23364a', linewidth=1.5)
    for y, row in enumerate(letters):
        for x, text in enumerate(row):
            ax.text(x, y, text, color='white' if values[y][x] in (0, 3, 4) else '#243342', ha='center', va='center', weight='bold')
    ax.set_title('12 development cases: artifact checks, all conditions request high', loc='left', fontsize=14, pad=16)
    ax.legend(handles=[Patch(color=colors[3], label='P: artifact pass'), Patch(color=colors[4], label='B: correct block'),
                       Patch(color=colors[0], label='F: submitted, fails oracle'), Patch(color=colors[1], label='T / U / E: budget / unknown / service')],
              loc='upper left', bbox_to_anchor=(-.005, -.11), ncol=2, frameon=False, fontsize=9)
    bar_colors = ['#91a3b3', '#4e8eb4', '#257a61']
    for i, model in enumerate(models):
        for j, arm in enumerate(arms):
            s = next(s for s in result['summary'] if (s['model'], s['arm']) == (model, arm))
            value = s['median_seconds_all_attempts']
            timing.barh(i * 4 + j, value or 0, color=bar_colors[j], height=.7)
            timing.text((value or 0) + 1, i * 4 + j, f'{value:.1f}s' if value is not None else 'unknown', va='center', fontsize=9)
    timing.set_yticks([0, 1, 2, 4, 5, 6], [names[a] for _ in models for a in arms])
    timing.invert_yaxis(); timing.set_xlim(0, 140)
    timing.set_xlabel('Median wall time per attempt (seconds; includes failures)')
    timing.axvline(120, color='#b96437', linestyle=':', linewidth=1)
    timing.text(121, -.7, '120s budget', color='#9a5430', fontsize=8)
    timing.spines[['top', 'right']].set_visible(False)
    timing.text(139, 1, 'GPT-5.6-Sol', ha='right', va='center', weight='bold')
    timing.text(139, 5, 'DeepSeek V4 Pro', ha='right', va='center', weight='bold')
    fig.text(.02, .015, '11 processable cases + 1 required block per condition. Exposed development inputs; descriptive engineering results, not a held-out test.', fontsize=9, color='#4e5964')
    fig.text(.02, .04, 'E06 uses the frozen strict oracle; its one-edge floating-point boundary sensitivity is documented in the report.', fontsize=9, color='#765636')
    fig.subplots_adjust(left=.23, right=.98, top=.92, bottom=.16, hspace=.65)
    for extension in ('png', 'svg', 'pdf'):
        target = delivery / (stem + '.' + extension)
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('delivery', type=Path)
    p.add_argument('--stem', default='COMPARISON_r2')
    args = p.parse_args()
    render(args.delivery, args.stem)
