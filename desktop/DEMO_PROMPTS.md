# NeuroDiscovery demo prompts

Use **GPT-6-Astra**. Start a new conversation for each example and select the indicated **AutoResearch** mode in the model menu. The interface, prompts, replies, tool feedback, graph labels, tables, artifacts and automatic conversation titles use English.

## 1. Chat — Off

```text
Who are you, and what can you help me with?
```

## 2. Research ideas — Idea

```text
Help me develop research hypotheses about cerebellar–default mode network connectivity and attention symptoms in children with ADHD. Show your workflow.
```

The response previews eight priority candidates from a pool of 500. Open `hypotheses.csv` in the artifact panel to view, search or download the complete table. Scores requiring detailed review remain blank for the other 492 candidates.

## 3. Data preparation — Data

```text
Please prepare the datasets in /data/demo and show your workflow.
```

## 4. Experiments — Model

```text
Use the data in /data/demo/adhd200 to test the hypotheses in /ideas/adhd_network/IDEA.md. Show your workflow and results.
```

The experiment scope is named **Model** in the AutoResearch menu.

## 5. Complete research workflow — Full

```text
Explore and test hypotheses about cerebellar–default mode network connectivity and attention symptoms in children with ADHD, using /data/demo/adhd200. Show your workflow and results.
```

Each round presents the experimental comparison, development results, assessment and next step. The default sequence covers initial candidates, site variability and capacity matching, then motion controls. Unsuccessful H3 results remain in the record. The initial table retains 500 candidates; adding H6 produces a separate final table with 501 candidates.

Round tables are available as soon as their links appear. The final artifacts include `round_metrics.csv`, `loop_history.json` and `REPORT.md`. Development feedback and final validation are reported separately.

To show the branch that stops after 20 unsuccessful rounds, append this sentence to the Full prompt:

```text
If the evidence remains insufficient, show how the system continues and eventually stops.
```

## Running and recording

- The demo loads the production HTML, CSS, JavaScript and preload. No separate demo interface is used.
- Windows: run `dist-demo-native/NeuroDiscovery-Native-Demo-1.0.0-Portable-x64.exe`, or extract the ZIP and run `NeuroDiscovery.exe`.
- From source: run `npm run prepare:demo:showcase`, then `npm run dev:demo` in `desktop`.
- New profiles default to English. For an existing profile, select **English** in Settings. Start new conversations to record the English examples; previous message text is retained.
- Known automatic titles from older releases are updated to English once, after backing up the history. Custom titles remain unchanged.
- Tool waits last 0.5–1.6 seconds and replies stream in small chunks. Each individual wait is under five seconds; a complete workflow takes longer.
- `/data/demo` and `/ideas/...` are logical demo paths. The package includes the corresponding files under `resources/demo-workspace`.
- This keyless build uses scripted replies, scores and experimental results, with synthetic data arrays. It performs no real LLM calls, graph research, training or scientific validation. Artifact provenance preserves this distinction.
- Prompts, backward-compatible aliases, scenario content, scores and artifacts are defined in `demo/scenarios.js` and served by `demo/native-server.cjs`.
