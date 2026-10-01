# Hypothesis selection in the client

The **Hypothesis selection / 假设筛选偏好** control sets `novelty_mode` for the
current chat. It appears in the model menu only for the two AutoResearch modes
that produce hypotheses: **Idea** and **Full**. Data and Model runs cannot change
how hypotheses are selected, so the route is hidden there and a stored mode is
kept unchanged. The default is `balanced` for new chats and requests that omit
the mode. Existing saved chat selections are preserved. Selection is saved with
the chat and captured on each new or edited/resubmitted turn, including the
exported agent work log. Changing it does not modify an already frozen workflow.

- `novelty_first`: only evidence-backed novelty candidates; fewer or zero is valid.
- `balanced`: fill the budget from every scientifically valid candidate, ranked by
  novelty 40%, structure 20%, GNN 20%, scientific review 20%.

There are two modes. The earlier three were consolidated by a behaviour-preserving
rename: the previous `strict` became `novelty_first` and both the previous
`novelty_first` and `weighted` became `balanced`. The unambiguous old names
`strict` and `weighted` are still accepted at the request/CLI boundary and mapped
to their current equivalent; a live `novelty_first` request uses the new meaning. The client
only offers the two current modes. Stored chat selections written before the rename
are migrated once, using the recorded store schema version: an old stored
`novelty_first` meant the permissive composite mode and therefore migrates to
`balanced`, not to the current conservative `novelty_first`.

The within-tier quality score is 30% structure, 30% GNN, and 40% scientific
review. Novelty utility is 1 for supported substantive extension, 0.7
for a supported potential new relation, 0.2 for uncertainty, and 0 for known
relations. These are decision utilities, not probabilities or first-report claims.

HTTP and WebSocket chat handlers validate the mode and supply a scoped selection
instruction to the agent. The client does not itself perform literature searches
or certify novelty. A generated narrative alone is not a validated selection.
The deterministic selection boundary is `neurooracle.src.novelty_policy.select_reviewed`,
also exposed as `POST /api/hypotheses/select`. Both modes require the same
scientific validity and operationalization gates. A known-prior warning from any
expert vetoes novelty eligibility; unresolved evidence does not become novelty.

## Reviewed candidate contract

Each candidate needs `hypothesis_id`, `structural_score`, `gnn_path_score`,
`science: {verdict, critic_score}`, three independently obtained `expert_novelties`,
and an independent literature `adjudication`. Scores must be finite in [0, 1].
Each review contains:

- `classification`: `exact_prior`, `same_scientific_conclusion`,
  `substantive_extension`, `potential_new_relation`, or `uncertain`;
- `reference_ids`: an array of actual reviewed source identifiers;
- `scientific_delta`: documented substantive difference, empty when unknown;
- `executable_test_covers_delta`: boolean.

Adjudication additionally requires `search_evidence_sufficient` and
`registered_test_matches_hypothesis` booleans. Missing data must be disclosed,
not fabricated. The API recomputes gates from these reviews rather than trusting
client-supplied gate flags. Schema validation is not verification of paper contents.

```json
{"novelty_mode": "novelty_first", "limit": 5, "candidates": []}
```

The response retains every original candidate and literature label, appends
selection reasons, and returns ranked `selected_ids`. Known candidates selected
by a permissive mode remain replication; no mode grants a new-finding claim.

For local reviewed JSON lists, the agent can use the same policy without a server:

```bash
python -m neurooracle.src.novelty_policy --input reviewed.json --output selection.json --mode novelty_first --limit 5
```

The output must be a new file. Historical experiments under `.codex_tmp` are not
imported or modified. No model calls, statistics, KG edits, or experiment reruns
are performed by this selector.

## Interface reference

The visual refresh was informed by the locally installed DeepSeek Harness
`0.1.1-rc.2` frontend (`dsh-client-ui-layout/lib/client.js` and
`dsh-web-frontend/dist/assets/index-C6eRlFa6.css`): neutral surfaces, a compact
sidebar, unobtrusive controls, restrained borders, and clear menu hierarchy.
NeuroDiscovery uses its own CSS and retains its identity and existing workflows;
it does not bundle DSH code, logos, or runtime dependencies.
