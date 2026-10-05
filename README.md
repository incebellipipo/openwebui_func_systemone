# System One for Open WebUI

Use [System One](https://docs.typesafe.ai/concepts/system-one) models from [Open WebUI](https://github.com/open-webui/open-webui). System One models don't generate text. You give them a `state` and a set of typed questions, and they return typed answers with probabilities and a confidence value. This repo connects Open WebUI to any server that implements the System One request format, including a local [Ollama](https://docs.ollama.com/api/systemone) server and TypeSafe's hosted API.

There are two independent pieces. Install one or both.

| File | Type in Open WebUI | Use it when |
|---|---|---|
| [`system_one_pipe.py`](system_one_pipe.py) | Function (pipe) | You want to pick a System One model from the model dropdown, paste a request body, and see the typed answer. Nothing else is needed. |
| [`system_one.py`](system_one.py) | Tool | You want a normal chat model to decide on its own when to call System One during a conversation (classify, score, yes/no). |

For sending exact requests yourself, such as benchmarks and testing, the pipe is the right choice. The tool adds its descriptions to every request of any chat where it is enabled, which matters if your chat model has a small context window.

## Question types

System One answers three kinds of question:

- **noul**: a yes/no question. Returns the probability of yes.
- **choice**: pick one option from a set you define. Returns the chosen option, a probability for every option, and a confidence.
- **score**: rate against an ordered scale you define. Returns a probability-weighted score, a probability for every level, and a confidence.

Example request body:

```json
{
  "model": "tev1",
  "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
  "questions": {
    "status": {
      "type": "choice",
      "instructions": "What is the invoice status?",
      "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}
    },
    "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}
  }
}
```

See the [TypeSafe API reference](https://docs.typesafe.ai/api) for the full format.

## Requirements

- Open WebUI with Functions and Tools enabled.
- A server that implements `POST /v1/systemone`:
  - **Ollama** v0.35.0 or later, with a local System One model (a GGUF model with a scoring-capable runner; cloud and MLX models are not supported). See the [Ollama System One docs](https://docs.ollama.com/api/systemone).
  - **TypeSafe** hosted API (`https://api.typesafe.ai`), with an API key.
  - Other providers that implement the same format (see [Other providers](#other-providers)).

## Install: model dropdown (pipe)

1. In Open WebUI, go to **Admin Panel → Functions → +**, paste the contents of `system_one_pipe.py`, and save.
2. Switch the function **on** with its toggle. New functions are off by default.
3. Click the gear icon and check the valves (below).
4. Reload the page. Entries named **System One: ...** appear in the model dropdown.

### Using it

Pick a System One model in the dropdown, then paste a request body as your message. The body is sent as written to the server. A code fence around it is fine, and so are Python-style dicts (single quotes, `True`/`False`, trailing commas).

- A model picked in the dropdown **overrides** any `"model"` field in the pasted JSON. The footer under the answer shows which model answered.
- The **System One: (model from JSON)** entry does the opposite: it takes the model from the JSON's `"model"` field.

The reply shows each answer, with probabilities sorted from most to least likely, followed by the raw JSON response:

```
status → overdue · confidence 0.992

- overdue — 99.9%
- draft — 0.1%
- paid — <0.1%

large → yes · P(yes) 0.993

tev1 · 398 tokens in, 3 out
```

### Pipe valves

| Valve | Default | Meaning |
|---|---|---|
| `BASE_URL` | `http://ollama:11434` | Server base URL. Use `http://localhost:11434` if Open WebUI runs outside Docker. |
| `ENDPOINT_PATH` | `/v1/systemone` | Endpoint path. |
| `MODEL_NAMES` | `tev1,clef-flash` | Models to list in the dropdown, comma-separated. `*` lists every model the Ollama server has (models that are not System One models are refused by the server). |
| `ADD_JSON_MODEL_ENTRY` | on | Adds the "(model from JSON)" entry. |
| `API_KEY` | empty | Leave empty for a local Ollama server. When set, it is sent in the auth header. |
| `AUTH_HEADER` / `AUTH_SCHEME` | `Authorization` / `Bearer` | How the key is sent. Clear the scheme to send the bare key. |
| `REQUEST_TIMEOUT` | 120 | Seconds. |
| `SHOW_RAW_JSON` | on | Append the raw response JSON in a code block. |

## Install: chat model calls System One (tool)

1. Go to **Workspace → Tools → +**, paste the contents of `system_one.py`, and save.
2. Click the gear icon and set the valves.
3. Pick a **normal chat model that supports tool calling** (not a System One model) and enable the tool for the chat with the **+** button in the message box. To attach it permanently, edit the model under Workspace → Models. Setting Function Calling to **Native** in the model's Advanced Params usually makes calls more reliable.
4. Give the chat model enough context. The tool descriptions use several thousand tokens, and 8k or more is a sensible minimum.

Ask in plain language and name the tool when it matters:

```
Use system_one_classify. State: own vessel sees the target on its starboard bow ...
Question: which is the most seamanlike manoeuvre? Options:
01_slow_9: reduce speed to 3 kn ...
02_starboard_45: alter 45° to starboard ...
```

The tool exposes `system_one_yes_no`, `system_one_classify`, `system_one_score`, `system_one_ask` (several typed questions in one request) and `system_one_classify_batch` (one question over a list of items, run in parallel).

Because the chat model writes the request, it may paraphrase your wording. Use the pipe when a request must be sent exactly.

### Tool valves

| Valve | Default | Meaning |
|---|---|---|
| `PROVIDER_NAME` | `System One` | Label used in status messages and footers. |
| `BASE_URL` | `https://api.typesafe.ai` | Server base URL. |
| `ENDPOINT_PATH` | `/v1/systemone` | Endpoint path. |
| `MODEL` | `jev-latest` | Model name sent in the request. Use your local model name for Ollama. |
| `API_KEY` | empty | Optional. No auth header is sent when empty. |
| `AUTH_HEADER` / `AUTH_SCHEME` | `Authorization` / `Bearer` | How the key is sent. |
| `EXTRA_HEADERS` / `EXTRA_BODY` | empty | JSON objects merged into the headers or body. For Ollama, `{"keep_alive": "30m"}` in `EXTRA_BODY` keeps the model loaded. |
| `MAX_CHOICE_OPTIONS` | 26 | Ollama allows 26 options, TypeSafe 255. |
| `MAX_SCORE_LEVELS` | 10 | TypeSafe allows 10 levels, Ollama 26. |
| `MAX_QUESTIONS` | 64 | Questions per request. |
| `MAX_REQUEST_BYTES` | 65536 | Local check on the request size (Ollama's limit without images). `0` disables it. |
| `MAX_STATE_CHARS` | 24000 | Longer state is truncated. |
| `CONTEXT_MESSAGES` | 12 | Trailing chat messages used as state when none is given. |
| `PARSE_JSON_STATE` | on | Send a state that is a JSON string as structured data. |
| `REQUEST_TIMEOUT`, `MAX_RETRIES`, `BACKOFF_FACTOR` | 30, 3, 0.75 | Timeout and retry settings (429, 529, 5xx, timeouts). |
| `BATCH_CONCURRENCY`, `BATCH_MAX_ITEMS` | 8, 200 | Batch classification limits. |
| `RAW_JSON_OUTPUT` | off | Return the raw JSON instead of the formatted summary. |

## Other providers

The code only assumes the System One request and response format, so other providers need only valve changes.

| Provider | `BASE_URL` | `ENDPOINT_PATH` | `MODEL` |
|---|---|---|---|
| Ollama | `http://ollama:11434` | `/v1/systemone` | your local model, e.g. `tev1` |
| TypeSafe | `https://api.typesafe.ai` | `/v1/systemone` | `jev-latest` |
| OpenRouter (untested) | `https://openrouter.ai` | `/api/alpha/decisions` | `typesafe/jev-1.13` |

The OpenRouter values come from the original script this tool is based on and have not been tested here. To use two providers at once, import the tool twice under different names and configure each one separately.

## Troubleshooting

**"System One: ..." doesn't appear in the dropdown.** Make sure `system_one_pipe.py` was pasted under **Functions** (not Tools), that its toggle is on, and reload the page.

**HTTP 400 mentioning context.** The model's context window on the server is too small for the request. Raise the context length for the model in Ollama (for example `PARAMETER num_ctx` in its Modelfile, or the `OLLAMA_CONTEXT_LENGTH` environment variable) and check the [Ollama docs](https://docs.ollama.com/api/systemone) for how System One uses it. Pasting a request into a normal chat with a System One model selected from Open WebUI's own model list does not work either, because Open WebUI sends it to the model as a chat message instead of a System One request. Use the pipe entry instead.

**HTTP 404.** The model name doesn't exist on the server, or `BASE_URL` / `ENDPOINT_PATH` is wrong. If Open WebUI runs in Docker, `localhost` refers to the container. Use the Ollama container's hostname or `host.docker.internal`.

**Request refused for size.** The body is over `MAX_REQUEST_BYTES` (tool) or the server's limit. Shorten the state.

**Unreadable request (pipe).** The message wasn't a valid JSON object. The pipe replies with an example.

## Status

The pipe has been used against Ollama with local System One models (`tev1`, `clef-flash`). The tool has so far only been checked for syntax, not run end to end, so treat it as less proven.

## Credits and licence

`system_one.py` is adapted from [`ask_jev.py`](https://github.com/systemgroupnet/openwebui-ask-jev/blob/master/ask_jev.py) by Aryan Ebrahimpour (MIT), generalized from the TypeSafe/Jev API to any System One provider. System One and Jev are TypeSafe's; Ollama's System One API is documented at [docs.ollama.com/api/systemone](https://docs.ollama.com/api/systemone). This project is not affiliated with either.

MIT licence. Add a `LICENSE` file with your name and the year, and keep the original author's credit above.
