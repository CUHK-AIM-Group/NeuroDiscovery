import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";


function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Invalid argument near ${key ?? "<end>"}`);
    }
    out[key.slice(2)] = value;
  }
  return out;
}


function candidatePayloadContent(value, allowedCandidateIds = null) {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    value.method !== "brainpilot_native"
  ) {
    return null;
  }
  let valid = false;
  if (Array.isArray(value.hypotheses) && value.hypotheses.length > 0) {
    valid = value.hypotheses.every((item) => {
      if (item === null || typeof item !== "object" || Array.isArray(item)) return false;
      const candidateId = String(item.candidate_id ?? "");
      const confidence = Number(item.confidence);
      return (
        Number.isInteger(Number(item.rank)) &&
        candidateId.trim().length > 0 &&
        (allowedCandidateIds === null || allowedCandidateIds.has(candidateId)) &&
        typeof item.rationale === "string" &&
        item.rationale.trim().length > 0 &&
        Number.isFinite(confidence) &&
        confidence >= 0 &&
        confidence <= 1
      );
    });
  } else if (
    allowedCandidateIds === null &&
    Array.isArray(value.rules) &&
    value.rules.length > 0
  ) {
    valid = value.rules.every((item) => {
      if (item === null || typeof item !== "object" || Array.isArray(item)) return false;
      const weight = Number(item.weight);
      return (
        Number.isInteger(Number(item.rank)) &&
        item.when !== null &&
        typeof item.when === "object" &&
        !Array.isArray(item.when) &&
        Object.keys(item.when).length > 0 &&
        typeof item.rationale === "string" &&
        item.rationale.trim().length > 0 &&
        Number.isFinite(weight) &&
        weight >= -1 &&
        weight <= 1
      );
    });
  }
  return valid ? JSON.stringify(value, null, 2) : null;
}


function balancedJsonObjects(text) {
  const objects = [];
  for (let start = 0; start < text.length; start += 1) {
    if (text[start] !== "{") continue;
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let index = start; index < text.length; index += 1) {
      const char = text[index];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (char === "\\") {
          escaped = true;
        } else if (char === '"') {
          inString = false;
        }
        continue;
      }
      if (char === '"') {
        inString = true;
      } else if (char === "{") {
        depth += 1;
      } else if (char === "}") {
        depth -= 1;
        if (depth === 0) {
          objects.push(text.slice(start, index + 1));
          start = index;
          break;
        }
      }
    }
  }
  return objects;
}


function extractCandidateContent(value, allowedCandidateIds = null, depth = 0) {
  if (depth > 8 || value === null || value === undefined) return null;
  if (typeof value === "object") {
    const direct = candidatePayloadContent(value, allowedCandidateIds);
    if (direct !== null) return direct;
    const nestedValues = Array.isArray(value) ? value : Object.values(value);
    for (const nested of nestedValues) {
      const content = extractCandidateContent(nested, allowedCandidateIds, depth + 1);
      if (content !== null) return content;
    }
    return null;
  }
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    const content = extractCandidateContent(parsed, allowedCandidateIds, depth + 1);
    if (content !== null) return content;
  } catch {
    // The candidate object may be embedded in prose passed to a native agent.
  }
  for (const candidate of balancedJsonObjects(value)) {
    try {
      const content = candidatePayloadContent(JSON.parse(candidate), allowedCandidateIds);
      if (content !== null) return content;
    } catch {
      // Continue to later balanced objects.
    }
  }
  return null;
}


function parseDeliveredContent(raw, allowedCandidateIds = null) {
  try {
    const payload = JSON.parse(raw);
    const implicitResult = payload?.to === "principal" && payload?.msg_type === undefined;
    if (
      (payload?.msg_type === "result_deliver" || implicitResult) &&
      typeof payload.content === "string"
    ) {
      return extractCandidateContent(payload.content, allowedCandidateIds);
    }
  } catch {
    // Tool arguments may arrive in multiple deltas.
  }
  return null;
}


function recoverCandidateFromEvents(events, allowedCandidateIds = null) {
  const chunks = new Map();
  const messageChunks = new Map();
  let recovered = null;
  for (const event of events) {
    if (
      event?.type === "TEXT_MESSAGE_CONTENT" &&
      event?.agent_name === "principal" &&
      typeof event.delta === "string"
    ) {
      const id = String(event.message_id ?? "principal-message");
      const raw = (messageChunks.get(id) ?? "") + event.delta;
      messageChunks.set(id, raw);
      const candidate = extractCandidateContent(raw, allowedCandidateIds);
      if (candidate !== null) {
        recovered = {
          content: candidate,
          reason: "validated_principal_text_artifact",
        };
      }
    }
    if (event?.type !== "TOOL_CALL_ARGS" || typeof event.delta !== "string") continue;
    const id = String(event.tool_call_id ?? "anonymous");
    const raw = (chunks.get(id) ?? "") + event.delta;
    chunks.set(id, raw);
    const delivered = parseDeliveredContent(raw, allowedCandidateIds);
    if (delivered !== null) return { content: delivered, reason: "result_deliver" };
    const candidate = extractCandidateContent(raw, allowedCandidateIds);
    if (candidate !== null) {
      recovered = {
        content: candidate,
        reason: "validated_native_candidate_artifact",
      };
    }
  }
  return recovered;
}


const args = parseArgs(process.argv.slice(2));
for (const required of ["prompt", "out", "client-dist"]) {
  if (!args[required]) throw new Error(`Missing --${required}`);
}

const outDir = path.resolve(args.out);
fs.mkdirSync(outDir, { recursive: true });
const prompt = fs.readFileSync(path.resolve(args.prompt), "utf8");
let allowedCandidateIds = null;
if (args.menu) {
  const menuLines = fs
    .readFileSync(path.resolve(args.menu), "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "");
  if (menuLines.length < 2 || menuLines[0].split(",")[0] !== "candidate_id") {
    throw new Error("BrainPilot menu must be a non-empty CSV with candidate_id first");
  }
  allowedCandidateIds = new Set(
    menuLines.slice(1).map((line) => line.split(",", 1)[0].replace(/^\"|\"$/g, "")),
  );
}
const eventsPath = path.join(outDir, "events.json");
if (fs.existsSync(eventsPath)) {
  const existingEvents = JSON.parse(fs.readFileSync(eventsPath, "utf8"));
  const recovered = recoverCandidateFromEvents(existingEvents, allowedCandidateIds);
  if (recovered !== null) {
    fs.writeFileSync(path.join(outDir, "final.txt"), recovered.content, "utf8");
    fs.writeFileSync(
      path.join(outDir, "client_meta.json"),
      JSON.stringify(
        {
          started_at: null,
          finished_at: new Date().toISOString(),
          session_id: null,
          end_reason: `${recovered.reason}_from_existing_events`,
          event_count: existingEvents.length,
        },
        null,
        2,
      ),
      "utf8",
    );
    console.log(
      JSON.stringify({
        session_id: null,
        reason: `${recovered.reason}_from_existing_events`,
        events: existingEvents.length,
      }),
    );
    process.exit(0);
  }
}
const clientModule = await import(pathToFileURL(path.resolve(args["client-dist"])).href);
const startedAt = new Date().toISOString();
const baseUrl = (args["base-url"] ?? "http://127.0.0.1:9460/api").replace(/\/$/, "");
const client = new clientModule.BrainPilotClient({
  baseUrl,
});
const reusedSession = typeof args["session-id"] === "string" && args["session-id"].trim() !== "";
const sessionId = reusedSession
  ? args["session-id"].trim()
  : await client.createSession();
const controller = new AbortController();
let iterator = client.streamEvents(sessionId, controller.signal)[Symbol.asyncIterator]();
const events = [];
const chunks = new Map();
const seenEventKeys = new Set();
let filteringReconnectReplay = false;
const maxEvents = Number(args["max-events"] ?? 1000);
const maxReplayEvents = Number(args["max-replay-events"] ?? 50000);
const maxStreamReconnects = Number(args["max-stream-reconnects"] ?? 5);
const statePollMs = Number(args["state-poll-ms"] ?? 10000);
const maxIdenticalDeliveries = Number(args["max-identical-deliveries"] ?? 0);
if (!Number.isInteger(maxIdenticalDeliveries) || maxIdenticalDeliveries < 0) {
  throw new Error("--max-identical-deliveries must be one non-negative integer");
}
const maxDeliveryInterruptAttempts = Number(
  args["max-delivery-interrupt-attempts"] ?? 3,
);
if (
  !Number.isInteger(maxDeliveryInterruptAttempts) ||
  maxDeliveryInterruptAttempts < 1
) {
  throw new Error("--max-delivery-interrupt-attempts must be one positive integer");
}
let content = null;
let sessionActive = true;
let replayedEventsDrained = 0;
let idleBoundaryObserved = false;
let postSendEventsSkipped = 0;
let streamReconnects = 0;
let principalRunId = null;
const principalRunIds = new Set();
let sendResult = null;
let streamEnded = false;
let pendingNext = null;
const deliveryCounts = new Map();
const countedDeliveryToolCallIds = new Set();
let maxObservedIdenticalDeliveries = 0;
let deliveryLoopInterruptRequested = false;
let deliveryLoopInterruptAttempts = 0;

async function recordDeliveredContent(toolCallId, delivered) {
  content = delivered;
  if (countedDeliveryToolCallIds.has(toolCallId)) return;
  countedDeliveryToolCallIds.add(toolCallId);
  const count = (deliveryCounts.get(delivered) ?? 0) + 1;
  deliveryCounts.set(delivered, count);
  maxObservedIdenticalDeliveries = Math.max(maxObservedIdenticalDeliveries, count);
  if (
    maxIdenticalDeliveries > 0 &&
    count >= maxIdenticalDeliveries &&
    deliveryLoopInterruptAttempts < maxDeliveryInterruptAttempts
  ) {
    const response = await fetch(
      `${baseUrl}/sessions/${encodeURIComponent(sessionId)}/interrupt`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
        signal: controller.signal,
      },
    );
    if (!response.ok) {
      throw new Error(`BrainPilot repeated-delivery interrupt failed: HTTP ${response.status}`);
    }
    deliveryLoopInterruptRequested = true;
    deliveryLoopInterruptAttempts += 1;
  }
}

function isReconnectReplay(event) {
  const eventKey = JSON.stringify(event);
  if (filteringReconnectReplay && seenEventKeys.has(eventKey)) return true;
  filteringReconnectReplay = false;
  seenEventKeys.add(eventKey);
  return false;
}
async function polledInactiveEvent() {
  const response = await fetch(
    `${baseUrl}/sessions/${encodeURIComponent(sessionId)}/state`,
    { signal: controller.signal },
  );
  if (!response.ok) return null;
  const state = await response.json();
  if (state?.runState?.active !== false) return null;
  return {
    type: "CUSTOM",
    name: "session_state",
    _ts: new Date().toISOString(),
    _brainpilot_state_poll: true,
    value: { runState: state.runState },
  };
}
async function waitForEventOrPoll(eventPromise) {
  let timeoutId = null;
  try {
    return await Promise.race([
      eventPromise.then((item) => ({ kind: "event", item })),
      new Promise((resolve) => {
        timeoutId = setTimeout(() => resolve({ kind: "poll" }), statePollMs);
      }),
    ]);
  } finally {
    if (timeoutId !== null) clearTimeout(timeoutId);
  }
}
async function nextEvent() {
  while (true) {
    try {
      pendingNext ??= iterator.next();
      const outcome =
        statePollMs > 0
          ? await waitForEventOrPoll(pendingNext)
          : { kind: "event", item: await pendingNext };
      if (outcome.kind === "poll") {
        const inactive = await polledInactiveEvent();
        if (inactive !== null) return { value: inactive, done: false };
        continue;
      }
      const item = outcome.item;
      pendingNext = null;
      if (!item.done) return item;
      streamEnded = true;
      if (streamReconnects >= maxStreamReconnects) return item;
    } catch (error) {
      if (streamReconnects >= maxStreamReconnects) throw error;
    }
    streamReconnects += 1;
    filteringReconnectReplay = true;
    iterator = client.streamEvents(sessionId, controller.signal)[Symbol.asyncIterator]();
    pendingNext = null;
  }
}
try {
  if (reusedSession) {
    while (replayedEventsDrained < maxReplayEvents) {
      const { value: replayedEvent, done } = await nextEvent();
      if (done) {
        break;
      }
      if (isReconnectReplay(replayedEvent)) continue;
      replayedEventsDrained += 1;
      if (
        replayedEvent?.type === "CUSTOM" &&
        replayedEvent?.name === "session_state" &&
        replayedEvent?.value?.runState?.active === false
      ) {
        idleBoundaryObserved = true;
        break;
      }
    }
    if (!idleBoundaryObserved) {
      throw new Error("BrainPilot event stream did not reach an idle boundary");
    }
  }

  sendResult = await client.sendMessage(sessionId, prompt);
  if (sendResult?.accepted !== true) {
    throw new Error("BrainPilot rejected the round message");
  }
  if (sendResult?.queued === true) {
    throw new Error("BrainPilot session was still active; refusing to mix two rounds");
  }

  while (events.length < maxEvents) {
    const { value: event, done } = await nextEvent();
    if (done) {
      break;
    }
    if (isReconnectReplay(event)) continue;
    if (event?._brainpilot_state_poll === true) {
      events.push(event);
      sessionActive = false;
      break;
    }
    const eventRunId =
      typeof event?.run_id === "string" && event.run_id.trim() !== ""
        ? event.run_id.trim()
        : null;
    if (principalRunId === null) {
      if (
        event?.type === "RUN_STARTED" &&
        event?.agent_name === "principal" &&
        eventRunId !== null
      ) {
        principalRunId = eventRunId;
        principalRunIds.add(eventRunId);
      } else {
        postSendEventsSkipped += 1;
        continue;
      }
    } else if (
      event?.type === "RUN_STARTED" &&
      event?.agent_name === "principal" &&
      eventRunId !== null
    ) {
      principalRunIds.add(eventRunId);
    }
    events.push(event);
    if (event?.type === "TOOL_CALL_ARGS" && typeof event.delta === "string") {
      const id = String(event.tool_call_id ?? "anonymous");
      // BrainPilot may reuse a tool_call_id in a later principal run.  Keep
      // each streamed argument payload isolated by both run and call IDs.
      const deliveryEventKey = `${eventRunId ?? "unknown"}\u0000${id}`;
      const raw = (chunks.get(deliveryEventKey) ?? "") + event.delta;
      chunks.set(deliveryEventKey, raw);
      const delivered = parseDeliveredContent(raw, allowedCandidateIds);
      if (delivered !== null) {
        await recordDeliveredContent(deliveryEventKey, delivered);
      }
    }
    if (event?.type === "CUSTOM" && event?.name === "session_state") {
      sessionActive = event?.value?.runState?.active !== false;
      if (!sessionActive) break;
    }
  }
} finally {
  controller.abort();
}
fs.writeFileSync(path.join(outDir, "events.json"), JSON.stringify(events, null, 2), "utf8");
if (sessionActive) {
  const boundary = events.length >= maxEvents ? "event limit" : streamEnded ? "stream end" : "unknown boundary";
  throw new Error(`BrainPilot session remained active at ${boundary}; refusing partial output`);
}
let endReason = deliveryLoopInterruptRequested
  ? "result_deliver_repeated_delivery_guard"
  : "result_deliver";
if (content === null) {
  const recovered = recoverCandidateFromEvents(events, allowedCandidateIds);
  if (recovered !== null) {
    content = recovered.content;
    endReason = recovered.reason;
  }
}
if (content === null) {
  throw new Error("BrainPilot emitted no parseable native candidate artifact");
}
fs.writeFileSync(path.join(outDir, "final.txt"), content, "utf8");
fs.writeFileSync(
  path.join(outDir, "client_meta.json"),
  JSON.stringify(
    {
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      session_id: sessionId,
      session_reused: reusedSession,
      transport_run_id: sendResult?.runId ?? null,
      principal_run_id: principalRunId,
      principal_run_ids: [...principalRunIds],
      idle_boundary_observed: idleBoundaryObserved,
      replayed_events_drained: replayedEventsDrained,
      stream_reconnects: streamReconnects,
      post_send_events_skipped: postSendEventsSkipped,
      end_reason: endReason,
      event_count: events.length,
      max_identical_deliveries: maxIdenticalDeliveries,
      max_delivery_interrupt_attempts: maxDeliveryInterruptAttempts,
      max_observed_identical_deliveries: maxObservedIdenticalDeliveries,
      delivery_loop_interrupt_requested: deliveryLoopInterruptRequested,
      delivery_loop_interrupt_attempts: deliveryLoopInterruptAttempts,
    },
    null,
    2,
  ),
  "utf8",
);
console.log(
  JSON.stringify({
    session_id: sessionId,
    session_reused: reusedSession,
    transport_run_id: sendResult?.runId ?? null,
    principal_run_id: principalRunId,
    principal_run_ids: [...principalRunIds],
    replayed_events_drained: replayedEventsDrained,
    stream_reconnects: streamReconnects,
    post_send_events_skipped: postSendEventsSkipped,
    reason: endReason,
    events: events.length,
    max_identical_deliveries: maxIdenticalDeliveries,
    max_delivery_interrupt_attempts: maxDeliveryInterruptAttempts,
    max_observed_identical_deliveries: maxObservedIdenticalDeliveries,
    delivery_loop_interrupt_requested: deliveryLoopInterruptRequested,
    delivery_loop_interrupt_attempts: deliveryLoopInterruptAttempts,
  }),
);

// Last Updated At: 2026-08-01 10:20 HKT
