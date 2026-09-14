# Model compatibility

Reviewed against provider documentation on 2026-09-14. These are **optional
application adapters**, not changes to any experiment protocol or default model.
Offline contract tests do not establish model entitlement, endpoint availability,
scientific performance, or compatibility with an unreleased model.

## Supported routes

| Family | Prepared route | Important controls / state |
| --- | --- | --- |
| GPT, including GPT-6 Astra | OpenAI Responses for official GPT-6; Chat Completions retained for older/custom routes | `reasoning.effort`, exact output cap, encrypted reasoning and function call IDs |
| Claude | Native Anthropic Messages | Adaptive thinking / effort on supported models, complete thinking signatures, `tool_use` / `tool_result` |
| Gemini | Google's OpenAI-compatible endpoint | `reasoning_effort`, unmodified thought signatures in tool metadata |
| Grok | xAI Chat Completions; Responses can be selected explicitly | Reviewed reasoning levels; no unsupported stop/penalty parameters |
| DeepSeek | OpenAI-compatible Chat Completions | Thinking in `extra_body`, reasoning effort, complete `reasoning_content` across tool rounds |
| Kimi | Moonshot OpenAI-compatible Chat Completions | Thinking toggle, provider-fixed sampling for K2.5/K2.6, preserved reasoning |
| GLM | Zhipu/Z.AI OpenAI-compatible Chat Completions | Thinking toggle, unchanged `clear_thinking` when explicitly configured, preserved reasoning |
| Qwen | DashScope OpenAI-compatible Chat Completions | `enable_thinking` in `extra_body`, reviewed Qwen3.8 effort levels, preserved reasoning |
| Ollama Cloud | Authenticated `https://ollama.com/v1` Chat Completions | `OLLAMA_API_KEY`, explicit reasoning effort, temperature, output cap and streaming usage; separate from local Ollama |

All nine routes reuse the same local tool executor and evidence-bounded
AutoResearch loop. Turning on AutoResearch does not by itself authorize new data,
provider fallback, changed scientific criteria, or increased experiment budgets.
Reasoning metadata is used for protocol continuity, not displayed as the answer.
This adaptation covers text research workflows and local function tools; it is
not a claim of support for every provider's image/video, live audio, server tools,
async tools, compaction, or managed-agent APIs.

## Desktop configuration

In **Settings → Models**, configure one active connection and select its models:

1. Enter the provider, API base URL and API key. Connection/generation controls and
   optional file-backed credentials are under **Advanced**.
2. Click **Fetch models**. This contacts the draft endpoint without saving or generating
   text. Search and check only the models you want; nothing is selected automatically.
3. Click **Add selected**, choose the default in **Your models**, then **Save settings**.
   Use **Restart now** only when it is shown for pending model/runtime changes.
   Only added models appear in the chat picker.

Removing a model removes it from the draft library, not from the provider or disk.
An explicitly empty library stays empty, including after backend normalization;
chat generation requires adding a model. Legacy desktop configs migrate only their
active model, not the complete provider catalog. Providers without model discovery
can use **Add ID** with an exact model identifier.

Provider changes apply a preset only after that explicit action, clear the previous
credentials/models and reset provider-specific controls. Editing the endpoint also
clears the previous credentials/models from the draft; re-enter credentials for the
new destination. Saving remains explicit. This is a single-connection library, not
a simultaneous multi-endpoint credential manager. Scientific defaults outside the
desktop-managed environment and historical interface names are unchanged.

Saving language, text size or other live preferences does not request a restart.
The desktop compares effective startup settings with this launch's baseline, not
merely the previous save. Unchanged/default-equivalent values do not create a
prompt; pending runtime changes remain pending across saves, and reverting them
removes the restart button. The in-memory comparison never reads or hashes a key file.

Ollama Cloud was added on 2026-09-15, following the [Cloud API](https://docs.ollama.com/cloud)
and [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) documentation.
Choose `ollama_cloud` for direct authenticated cloud access and `ollama` for the
unchanged local endpoint. Use the exact model ID returned by the cloud model list;
local `:cloud` aliases need not be the direct API ID. Availability and tool support
remain account/model-dependent.

For this client's compatible API route, use **`https://ollama.com/v1`** for cloud
or **`http://localhost:11434/v1`** for local Ollama (normally no key). Do not append
`/models` or `/chat/completions` to the base URL. Native Ollama/Anthropic examples
may use `https://ollama.com` instead; that is not this client's compatible base URL.

For desktop-only file-backed credentials, enter an absolute **Ollama key file**
path and leave the pasted key blank. The last two lines are the primary and reserve
keys. **Active Ollama key** is a manual selection, applied after Save and Restart;
there is no automatic rotation, quota bypass or fallback to an OpenAI key. The
launcher passes only the selected key to its backend as `OLLAMA_API_KEY`; the key
file is never copied into settings or the repository. This mode is restricted to
`https://ollama.com/v1`. Do not move the file while relying on it. The optional
desktop `environmentFile` path is passed as `NEUROCLAW_ENV_FILE`, allowing a source
launch to keep its environment separate from repository/experiment configuration.

The added controls are:

- **API protocol:** `auto`, `chat_completions`, `responses`, or `anthropic`.
  Auto uses Responses for `gpt-6*` at official OpenAI hosts and native Messages for
  Anthropic. A custom GPT-6 proxy must explicitly select Responses and support it;
  the app never silently redirects your key or request to OpenAI.
- **Reasoning effort / thinking mode:** leave at `default` unless intentionally
  changing model behavior. The two controls are not interchangeable; unsupported
  reviewed combinations fail before dispatch, without a fallback request.
- **Output token limit:** blank preserves the existing/provider default. Claude's
  required fallback remains 4,096 tokens unless configured. An explicit positive
  integer is passed as `max_output_tokens` for Responses, `max_completion_tokens`
  for OpenAI reasoning Chat, or `max_tokens` for Messages/other compatible Chat.
  Smaller call-specific budgets (for example 600 tokens for memory extraction)
  remain smaller; the configured ceiling is never exceeded. No cap increase or regeneration occurs on
  an incomplete result.
- **Temperature:** blank uses the provider default. GPT-6, recent Claude and Kimi's
  fixed-sampling models reject an explicitly configured override. Generic old probe
  parameters are omitted for these models; existing Ollama experiment settings are
  not translated into another provider's controls.

The server's configuration preflight performs no account/key lookup or model call.
The explicit **Fetch models** action contacts the draft endpoint;
it supports Anthropic's native authentication as well as compatible `/models` APIs.
Redirects are rejected to avoid forwarding credentials to another destination;
remote endpoints require HTTPS, with HTTP allowed for loopback development servers.
Discovery errors omit the upstream response body and credentials. There is no
automatic retry, key rotation, selection or catalog write. Startup and opening the
chat model menu make no discovery request. The legacy backend model-list route is
read-only and is no longer used by the desktop picker.
A listed model is not proof of tool support or successful generation. Custom model
IDs remain editable, so future IDs do not require a client release when they use
an already-supported protocol. New protocol features still require review/testing.

Optional background memory, semantic compression and reflection keep their
lightweight role: OpenAI retains `gpt-4o-mini`; for other endpoints explicitly set
`llm_backend.auxiliary_model` to an available small model. Without that setting,
their existing deterministic/disabled fallbacks apply, with no background model
request. They never switch to the selected flagship or a different provider.
Responses JSON formats are translated to `text.format`. Claude JSON schemas use
`output_config.format`; legacy schema-free `json_object` requests use a format
instruction plus local JSON validation, **not** a claim of schema-constrained
decoding. Malformed output is rejected without automatic regeneration.

Regional keys and endpoints must match. In particular, Moonshot China/global,
Zhipu China/Z.AI global, and DashScope regions/workspace endpoints are not assumed
interchangeable. Keep custom URLs explicit. API keys are never returned by the
provider preset/preflight/model-list responses.

## Continuity, cancellation and limits

Within an AutoResearch/tool turn, Responses output items, Claude content blocks,
Gemini signatures and compatible-provider reasoning are preserved unchanged.
The app does not fabricate missing historical reasoning. When DeepSeek receives
an older text-only conversation prefix (for example from HTTP chat history), that
prefix is explicitly attributed as imported visible conversation context; the
current native tool turn remains intact. This is not a claim that hidden state is
recoverable from an old text export.

Streaming preserves the user-visible response and normalizes actual usage.
Claude cached input is included in total input tokens. A truncated, filtered or
incomplete tool response never executes its partial tool calls or masquerades as
successful research delivery. Cancellation stops subsequent tool/model dispatch;
an already in-flight upstream read may drain before its iterator closes. Unknown
usage is not an account balance, and this layer does not estimate vendor billing.

The model-library implementation and its validation make no real provider calls or
scientific requests. Earlier, separately authorized Ollama smoke checks belong to
their own record; they are not evidence that all returned models can generate or
use tools. No commits, pushes or package rebuilds were made for this change. Existing v1.0.0 binaries
must be rebuilt to include these source changes. A release should add authorized
account-specific smoke tests: one text reply, a two-step local tool task, a small
AutoResearch deliverable, cancellation, and observed token usage for each endpoint.

## Offline verification

`core/llm/tests` exercises the provider wire routes with synthetic responses, actual
OpenAI/Anthropic SDKs over `httpx.MockTransport`, multi-tool metadata round trips,
AutoResearch continuation beyond eight rounds, cancellation, incomplete responses,
configuration safety, desktop JavaScript, and WebSocket streaming/usage. No real
model response is used as fixture evidence. These tests are included in CI.

## Official references

- [OpenAI GPT-6 migration and Responses](https://developers.openai.com/api/docs/guides/latest-model)
- [OpenAI Responses migration](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [Claude models](https://platform.claude.com/docs/en/models/overview), [effort](https://platform.claude.com/docs/en/build-with-claude/effort), [thinking and tools](https://platform.claude.com/docs/en/claude_api_primer)
- [Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- [Grok 4.6](https://docs.x.ai/developers/grok-4-6), [reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)
- [DeepSeek thinking and tool calls](https://api-docs.deepseek.com/guides/thinking_mode/)
- [Kimi K2.6 parameters and tools](https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart)
- [GLM thinking modes](https://docs.z.ai/guides/capabilities/thinking-mode), [GLM-5.1](https://docs.z.ai/guides/llm/glm-5.1)
- [Qwen Chat Completions parameters](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)
