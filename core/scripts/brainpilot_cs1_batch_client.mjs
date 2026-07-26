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


function parseDeliveredContent(raw) {
  try {
    const payload = JSON.parse(raw);
    const implicitResult = payload?.to === "principal" && payload?.msg_type === undefined;
    if (
      (payload?.msg_type === "result_deliver" || implicitResult) &&
      typeof payload.content === "string"
    ) {
      const content = payload.content.trim();
      if (content.startsWith("{") && content.includes('"hypotheses"')) return content;
    }
  } catch {
    // Tool arguments may arrive in multiple deltas.
  }
  return null;
}


const args = parseArgs(process.argv.slice(2));
for (const required of ["prompt", "out", "client-dist"]) {
  if (!args[required]) throw new Error(`Missing --${required}`);
}

const outDir = path.resolve(args.out);
fs.mkdirSync(outDir, { recursive: true });
const prompt = fs.readFileSync(path.resolve(args.prompt), "utf8");
const clientModule = await import(pathToFileURL(path.resolve(args["client-dist"])).href);
const startedAt = new Date().toISOString();
const client = new clientModule.BrainPilotClient({
  baseUrl: args["base-url"] ?? "http://127.0.0.1:9460/api",
});
const sessionId = await client.createSession();
const controller = new AbortController();
const stream = client.streamEvents(sessionId, controller.signal);
await client.sendMessage(sessionId, prompt);
const events = [];
const chunks = new Map();
const maxEvents = Number(args["max-events"] ?? 1000);
let content = null;
try {
  for await (const event of stream) {
    events.push(event);
    if (event?.type === "TOOL_CALL_ARGS" && typeof event.delta === "string") {
      const id = String(event.tool_call_id ?? "anonymous");
      const raw = (chunks.get(id) ?? "") + event.delta;
      chunks.set(id, raw);
      content = parseDeliveredContent(raw);
      if (content !== null) break;
    }
    if (
      event?.type === "CUSTOM" &&
      event?.name === "session_state" &&
      event?.value?.runState?.active === false
    ) {
      break;
    }
    if (events.length >= maxEvents) break;
  }
} finally {
  controller.abort();
}
fs.writeFileSync(path.join(outDir, "events.json"), JSON.stringify(events, null, 2), "utf8");
if (content === null) throw new Error("BrainPilot emitted no parseable result_deliver content");
fs.writeFileSync(path.join(outDir, "final.txt"), content, "utf8");
fs.writeFileSync(
  path.join(outDir, "client_meta.json"),
  JSON.stringify(
    {
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      session_id: sessionId,
      end_reason: "result_deliver",
      event_count: events.length,
    },
    null,
    2,
  ),
  "utf8",
);
console.log(JSON.stringify({ session_id: sessionId, reason: "result_deliver", events: events.length }));
