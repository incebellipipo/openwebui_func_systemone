"""
title: System One Models (raw JSON)
author: Emir Cem Gezer
description: Adds System One models (e.g. local Ollama models) to the Open WebUI model dropdown, so you can pick one yourself. Paste a System One request body as your message and it is sent as-is to the System One endpoint; the typed answers come back formatted. The pasted JSON may also be a Python-style dict (single quotes, trailing commas).
version: 0.2.0
licence: MIT
"""

import ast
import json
import re
from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, Field

NOUL_THRESHOLD = 0.5
JSON_MODEL_ID = "from_json"

HELP = (
    "Paste a System One request body as your message, for example:\n\n"
    "```json\n"
    '{"state": "Checkout has returned 500 errors since 9am.",\n'
    ' "questions": {"label": {"type": "choice",\n'
    '   "instructions": "Which label fits?",\n'
    '   "criteria": {"billing": "Payments", "bug": "Software errors"}}}}\n'
    "```\n\n"
    "A model picked in the dropdown overrides any `model` in the JSON. With the \"(model from JSON)\" entry, the JSON's `model` is used."
)


class Pipe:
    class Valves(BaseModel):
        BASE_URL: str = Field(
            default="http://ollama:11434",
            description="Server base URL. Ollama: http://ollama:11434 (or http://localhost:11434).",
        )
        ENDPOINT_PATH: str = Field(
            default="/v1/systemone",
            description="System One endpoint path.",
        )
        MODEL_NAMES: str = Field(
            default="tev1,clef-flash",
            description="Comma-separated model names to list in the dropdown, e.g. 'tev1,clef-flash,nimble'. Use '*' to list every model the Ollama server has (non System One models will be rejected by the server if you pick them).",
        )
        ADD_JSON_MODEL_ENTRY: bool = Field(
            default=False,
            description="Also add a 'System One: (model from JSON)' dropdown entry. With it, the model comes from the \"model\" field of the pasted JSON; the other entries ignore that field and use the model you picked.",
        )
        API_KEY: str = Field(
            default="",
            description="Optional API key. Leave empty for a local Ollama server (no auth header is sent).",
        )
        AUTH_HEADER: str = Field(default="Authorization", description="Header carrying the API key.")
        AUTH_SCHEME: str = Field(default="Bearer", description="Prefix before the key; empty sends the bare key.")
        REQUEST_TIMEOUT: int = Field(default=120, description="Per-request timeout in seconds.")
        SHOW_RAW_JSON: bool = Field(
            default=True,
            description="Append the raw response JSON (in a code block) under the formatted answer.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def pipes(self) -> list:
        names = [n.strip() for n in self.valves.MODEL_NAMES.split(",") if n.strip()]
        if "*" in names:
            names = [n for n in names if n != "*"]
            for found in await self._discover():
                if found not in names:
                    names.append(found)
        entries = []
        if self.valves.ADD_JSON_MODEL_ENTRY:
            entries.append({"id": JSON_MODEL_ID, "name": "System One: (model from JSON)"})
        entries.extend({"id": n, "name": f"System One: {n}"} for n in names)
        return entries or [{"id": JSON_MODEL_ID, "name": "System One: (model from JSON)"}]

    async def _discover(self) -> list:
        """Names of all models on the server, via Ollama's /api/tags. Empty on any failure."""
        url = f"{self.valves.BASE_URL.strip().rstrip('/')}/api/tags"
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self._headers()) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
            return [m["name"] for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    async def pipe(self, body: dict, __event_emitter__=None) -> str:
        message = self._last_user_text(body)
        if not message.strip():
            return HELP

        try:
            request = self._parse_request(message)
        except ValueError:
            return "I could not read that message as a System One request.\n\n" + HELP

        for field in ("state", "questions"):
            if field not in request:
                return f"The request is missing the required field `{field}`.\n\n" + HELP

        # body["model"] looks like "<function id>.<model name>". A model picked
        # in the dropdown always wins over any "model" in the pasted JSON; only
        # the "(model from JSON)" entry takes the model from the JSON.
        selected = str(body.get("model", ""))
        selected = selected.split(".", 1)[1] if "." in selected else selected
        if selected and selected != JSON_MODEL_ID:
            request["model"] = selected
        elif not request.get("model"):
            return (
                'This entry needs a `"model"` field in the JSON (for example '
                '`"model": "tev1"`), or pick a specific System One model from the dropdown.'
            )

        await self._emit(__event_emitter__, f"Sending to {request['model']}...", False)
        try:
            data = await self._post(request)
        except Exception as exc:  # network, HTTP, JSON errors
            await self._emit(__event_emitter__, "Request failed", True)
            return f"**Request failed:** {exc}"
        await self._emit(__event_emitter__, "Done", True)

        out = self._render(data)
        if self.valves.SHOW_RAW_JSON:
            raw = json.dumps(data, indent=2, ensure_ascii=False)
            out += f"\n\n**Raw response**\n\n```json\n{raw}\n```"
        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _last_user_text(body: dict) -> str:
        for message in reversed(body.get("messages") or []):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
        return ""

    @staticmethod
    def _parse_request(text: str) -> dict:
        text = text.strip()
        fenced = re.search(r"```(?:json|python)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        # Keep only the outermost {...} in case of surrounding prose.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no object found")
        text = text[start : end + 1]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Trailing commas are the usual problem.
            cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                # Python-style dict (single quotes, True/False/None).
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError("not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("not an object")
        return parsed

    def _url(self) -> str:
        base = self.valves.BASE_URL.strip().rstrip("/")
        path = self.valves.ENDPOINT_PATH.strip().lstrip("/")
        return f"{base}/{path}"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        key = self.valves.API_KEY.strip()
        if key:
            scheme = self.valves.AUTH_SCHEME.strip()
            headers[self.valves.AUTH_HEADER.strip() or "Authorization"] = (
                f"{scheme} {key}" if scheme else key
            )
        return headers

    async def _post(self, request: dict) -> dict:
        timeout = aiohttp.ClientTimeout(total=max(1, self.valves.REQUEST_TIMEOUT))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self._url(), headers=self._headers(), json=request
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    short = re.sub(r"\s+", " ", text).strip()[:600] or "(empty)"
                    raise RuntimeError(f"HTTP {resp.status}: {short}")
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"non-JSON response: {text[:300]}") from exc

    async def _emit(self, emitter, description: str, done: bool) -> None:
        if not emitter:
            return
        try:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done, "hidden": done},
                }
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _num(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _render(self, data: dict) -> str:
        answers = data.get("answers") or {}
        if not answers:
            return "No answers in the response."
        blocks = [self._render_answer(key, answer) for key, answer in answers.items()]
        usage = data.get("usage") or {}
        footer = f"_{data.get('model', '?')}"
        if usage:
            footer += (
                f" · {usage.get('input_tokens', '?')} tokens in,"
                f" {usage.get('output_tokens', '?')} out"
            )
        blocks.append(footer + "_")
        return "\n\n".join(blocks)

    def _render_answer(self, key: str, answer: Any) -> str:
        if not isinstance(answer, dict):
            return f"**{key}** → {answer}"
        kind = answer.get("type")
        conf = self._num(answer.get("confidence"))
        conf_text = f" · confidence {conf:.3f}" if conf is not None else ""

        if kind == "noul":
            p = self._num(answer.get("noul"))
            verdict = "yes" if p is not None and p >= NOUL_THRESHOLD else "no"
            shown = f"{p:.3f}" if p is not None else "n/a"
            return f"**{key}** → **{verdict}** · P(yes) {shown}"

        if kind == "choice":
            head = f"**{key}** → **{answer.get('choice', 'n/a')}**{conf_text}"
            return head + self._distribution(answer.get("probabilities"))

        if kind == "score":
            score = self._num(answer.get("score"))
            legend = answer.get("legend") if isinstance(answer.get("legend"), dict) else {}
            level = legend.get(str(round(score))) if score is not None else None
            top = f"/{len(legend) - 1}" if legend else ""
            shown = f"{score:.2f}" if score is not None else "n/a"
            level_text = f" ({level})" if level else ""
            head = f"**{key}** → **{shown}{top}**{level_text}{conf_text}"
            return head + self._distribution(answer.get("probabilities"), legend)

        return f"**{key}** → `{json.dumps(answer, ensure_ascii=False)}`"

    @staticmethod
    def _percent(p: float) -> str:
        pct = p * 100
        if pct >= 99.95:
            return "100%"
        if pct < 0.05:
            return "<0.1%"
        return f"{pct:.1f}%"

    def _distribution(self, probabilities: Any, legend: Any = None) -> str:
        """One line per option, most likely first, as a markdown list."""
        if not isinstance(probabilities, dict) or not probabilities:
            return ""
        rows = []
        for name, value in sorted(
            probabilities.items(), key=lambda kv: -(self._num(kv[1]) or 0)
        ):
            p = self._num(value)
            if p is None:
                continue
            label = f"{name}: {legend[name]}" if isinstance(legend, dict) and name in legend else name
            rows.append(f"- {label} — {self._percent(p)}")
        return "\n\n" + "\n".join(rows) if rows else ""
