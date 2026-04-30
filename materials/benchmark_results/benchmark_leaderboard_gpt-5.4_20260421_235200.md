# Benchmark Leaderboard

- Timestamp: 2026-04-21T23:41:11
- Scorer model: gpt-5.4
- Benchmark root: /mnt/d/Code/NeuroClaw/neuro_bench
- Output dir: /mnt/d/Code/NeuroClaw/output
- Scored case count: 100

## Ranking

| Rank | Model | Average Score (%) | Avg Skill Usage | Avg Tokens | Avg Time (s) |
|---:|---|---:|---:|---:|---:|
| 1 | claude-opus-4-6_withskills | 72.1 | 3.09 | 23958.77 | 88.511 |
| 2 | claude-sonnet-4-6_withskills | 70.39 | 5.52 | 26120.13 | 75.972 |
| 3 | claude-opus-4-6_noskills | 69.12 | 0.0 | 6834.43 | 72.687 |
| 4 | gpt-5.4_withskills | 67.69 | 0.83 | 11715.36 | 55.071 |
| 5 | claude-sonnet-4-6_noskills | 65.37 | 0.0 | 3864.14 | 48.534 |
| 6 | gpt-5.4_noskills | 64.57 | 0.0 | 4065.63 | 40.22 |
| 7 | qwen3.6-plus_withskills | 58.12 | 2.61 | 29762.44 | 68.955 |
| 8 | gemini-3.1-pro-preview_withskills | 56.65 | 1.3 | 15862.69 | 71.846 |
| 9 | gemini-3.1-pro-preview_noskills | 55.43 | 0.0 | 5334.76 | 46.227 |
| 10 | gemini-3-flash-preview_withskills | 54.1 | 2.16 | 30650.5 | 27.824 |
| 11 | gpt-5.4-mini_withskills | 50.61 | 1.78 | 21088.31 | 21.161 |
| 12 | qwen3.6-plus_noskills | 50.39 | 0.0 | 5264.91 | 56.97 |
| 13 | deepseek-v3.2_withskills | 49.63 | 2.69 | 42848.57 | 88.116 |
| 14 | gemini-3-flash-preview_noskills | 49.15 | 0.0 | 3942.31 | 15.868 |
| 15 | minimax-m2.7_withskills | 48.07 | 2.76 | 28501.27 | 76.397 |
| 16 | gpt-5.4-mini_noskills | 46.94 | 0.0 | 3113.58 | 8.388 |
| 17 | deepseek-v3.2_noskills | 45.49 | 0.0 | 3506.93 | 45.883 |
| 18 | grok-4-20-non-reasoning_withskills | 37.59 | 1.29 | 14950.43 | 17.039 |
| 19 | grok-4-20-non-reasoning_noskills | 35.97 | 0.0 | 2909.58 | 9.283 |
| 20 | minimax-m2.7_noskills | 35.1 | 0.0 | 4138.92 | 56.128 |

## Skill Gain (With Skills vs No Skills)

| Base Model | With Skills (%) | No Skills (%) | A_abs (%) | g | Interpretation |
|---|---:|---:|---:|---:|---|
| claude-opus-4-6 | 72.1 | 69.12 | 2.98 | 0.0965 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
| claude-sonnet-4-6 | 70.39 | 65.37 | 5.02 | 0.145 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
| deepseek-v3.2 | 49.63 | 45.49 | 4.14 | 0.0759 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
| gemini-3-flash-preview | 54.1 | 49.15 | 4.95 | 0.0973 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
| gemini-3.1-pro-preview | 56.65 | 55.43 | 1.22 | 0.0274 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
| gpt-5.4 | 67.69 | 64.57 | 3.12 | 0.0881 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
| gpt-5.4-mini | 50.61 | 46.94 | 3.67 | 0.0692 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
| grok-4-20-non-reasoning | 37.59 | 35.97 | 1.62 | 0.0253 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
| minimax-m2.7 | 48.07 | 35.1 | 12.97 | 0.1998 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
| qwen3.6-plus | 58.12 | 50.39 | 7.73 | 0.1558 | Positive normalized gain indicates proportional benefit from skills; consistency refers to similar proportion, not identical absolute improvement. |
