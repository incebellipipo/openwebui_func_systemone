"""
title: System One Client
author: adapted from Aryan Ebrahimpour's "Ask Jev"
description: Ask any System One provider typed questions - yes/no (noul), choice, or score - about text, the current conversation, or attached files. Works with TypeSafe's native API (Jev), a local Ollama server (POST /v1/systemone, v0.35.0+), and any other provider that implements the System One request/response format (OpenRouter's Decisions API, a proxy, a gateway). Everything provider-specific lives in the valves.
version: 0.3.0
licence: MIT
required_open_webui_version: 0.4.0
"""

import asyncio
import json
import random
import re
from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, Field

NOUL_THRESHOLD = 0.5
MIN_SCORE_LEVELS = 2
VALID_TYPES = ("noul", "choice", "score")

# System One models are typed decision models: instead of generating text they
# return a calibrated JSON answer in one shot. The wire format is:
#
#   POST <base_url><endpoint_path>
#   {"state": <str|object|array>, "model": "...", "questions": {
#       "<id>": {"type": "noul|choice|score", "instructions": ..., "criteria": ...}}}
#
#   -> {"model": "...", "answers": {"<id>": {"type": "noul", "noul": 0.95}
#                                         | {"type": "choice", "choice": ..., "probabilities": {...}, "confidence": ...}
#                                         | {"type": "score", "score": ..., "legend": {...}, "probabilities": {...}, "confidence": ...}},
#       "usage": {"input_tokens": n, "output_tokens": n}}
#
# Providers that implement it differ only in base URL, path, model name and
# auth header, which is exactly what the valves below configure.


class SystemOneError(Exception):
    """Raised when the System One provider is unreachable or rejects a request."""


class Tools:
    class Valves(BaseModel):
        PROVIDER_NAME: str = Field(
            default="System One",
            description="Display name used in status messages and result footers, e.g. 'TypeSafe', 'OpenRouter'.",
        )
        API_KEY: str = Field(
            default="",
            description="API key for the provider. Sent in the auth header on every request. Leave empty for providers that need no auth, such as a local Ollama server; no auth header is sent then.",
        )
        BASE_URL: str = Field(
            default="https://api.typesafe.ai",
            description="Provider base URL. TypeSafe native: https://api.typesafe.ai. Ollama: http://ollama:11434 (or http://localhost:11434). OpenRouter: https://openrouter.ai. Or any proxy / gateway.",
        )
        ENDPOINT_PATH: str = Field(
            default="/v1/systemone",
            description="Path appended to BASE_URL. TypeSafe native and Ollama: /v1/systemone. OpenRouter Decisions API: /api/alpha/decisions.",
        )
        MODEL: str = Field(
            default="jev-latest",
            description="Model identifier sent in the 'model' field. TypeSafe: 'jev-latest' or a pinned id like 'jev-1.13.0'. Ollama: the local model name, e.g. 'tev1'. OpenRouter: 'typesafe/jev-1.13'.",
        )
        AUTH_HEADER: str = Field(
            default="Authorization",
            description="Name of the header that carries the API key.",
        )
        AUTH_SCHEME: str = Field(
            default="Bearer",
            description="Prefix placed before the key in the auth header ('Bearer ' style). Leave empty to send the bare key, e.g. for providers that use an 'x-api-key' header.",
        )
        EXTRA_HEADERS: str = Field(
            default="",
            description="Optional JSON object of additional headers, e.g. {\"HTTP-Referer\": \"https://my.site\", \"X-Title\": \"Open WebUI\"}.",
        )
        EXTRA_BODY: str = Field(
            default="",
            description="Optional JSON object merged into the top level of every request body, for provider-specific fields. Ollama example: {\"keep_alive\": \"30m\"} keeps the model loaded between calls.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=30,
            description="Per-request timeout in seconds.",
        )
        MAX_RETRIES: int = Field(
            default=3,
            description="Retries after a 429 (rate limit), 529 (overload), 5xx, timeout, or network error.",
        )
        BACKOFF_FACTOR: float = Field(
            default=0.75,
            description="Base seconds for exponential backoff: factor * 2^attempt, plus jitter. A Retry-After header wins when present.",
        )
        MAX_STATE_CHARS: int = Field(
            default=24000,
            description="Hard cap on the characters sent as 'state'. Longer content is truncated. TypeSafe's Jev allows about 32k tokens for state plus the longest question; raise or lower this for other models.",
        )
        CONTEXT_MESSAGES: int = Field(
            default=12,
            description="How many trailing chat messages to use when no state is supplied.",
        )
        PARSE_JSON_STATE: bool = Field(
            default=True,
            description="If 'state' is a string holding a JSON object or array, send it as structured data instead of plain text (System One accepts both).",
        )
        MAX_CHOICE_OPTIONS: int = Field(
            default=26,
            description="Maximum options in a choice question. Ollama allows 26; TypeSafe allows 255. The default is the stricter limit so requests are never rejected for size.",
        )
        MAX_SCORE_LEVELS: int = Field(
            default=10,
            description="Maximum levels in a score question. TypeSafe allows 10; Ollama allows 26.",
        )
        MAX_QUESTIONS: int = Field(
            default=64,
            description="Maximum questions in one request (Ollama allows 64).",
        )
        MAX_REQUEST_BYTES: int = Field(
            default=65536,
            description="Refuse to send a request body larger than this many bytes (Ollama's limit without images is 64 KiB). Set 0 to disable the check.",
        )
        BATCH_CONCURRENCY: int = Field(
            default=8,
            description="Parallel requests during batch classification.",
        )
        BATCH_MAX_ITEMS: int = Field(
            default=200,
            description="Maximum items accepted by a single batch call.",
        )
        RAW_JSON_OUTPUT: bool = Field(
            default=False,
            description="Return the raw provider JSON instead of a formatted summary.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # Tools exposed to the model
    # ------------------------------------------------------------------

    async def system_one_yes_no(
        self,
        question: str,
        state: str = "",
        yes_means: str = "",
        no_means: str = "",
        __event_emitter__=None,
        __messages__: Optional[list] = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Ask a System One model a calibrated yes/no question about some content and get
        back the probability that the statement is true (0 = definitely no, 1 = definitely yes).
        Use it instead of judging yourself whenever a defensible, calibrated probability
        or a fast structured yes/no is needed.

        :param question: The yes/no question to evaluate, e.g. "Does this message convey urgency?".
        :param state: The content to evaluate. Leave empty to use attached files, or failing that the recent conversation.
        :param yes_means: Optional description of what a yes means, to sharpen a borderline question.
        :param no_means: Optional description of what a no means.
        :return: The probability and verdict.
        """
        question_spec: dict[str, Any] = {"type": "noul", "instructions": question}
        criteria = {}
        if yes_means.strip():
            criteria["true"] = yes_means.strip()
        if no_means.strip():
            criteria["false"] = no_means.strip()
        if criteria:
            question_spec["criteria"] = criteria
        return await self._ask_one(
            question, question_spec, state, __event_emitter__, __messages__, __files__
        )

    async def system_one_classify(
        self,
        question: str,
        options: dict,
        state: str = "",
        __event_emitter__=None,
        __messages__: Optional[list] = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Ask a System One model to pick exactly one option from a closed set, with a
        probability for every option and an overall confidence. Good for routing,
        triage, intent and category labelling.

        :param question: What is being decided, e.g. "Which team should handle this ticket?".
        :param options: Mapping of option name to a short description, e.g. {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations"}. 2 to 26 options (the limit is set in the tool's valves).
        :param state: The content to evaluate. Leave empty to use attached files, or failing that the recent conversation.
        :return: The chosen option, its confidence, and the full probability distribution.
        """
        try:
            criteria = self._coerce_options(options)
        except SystemOneError as exc:
            return f"{self.valves.PROVIDER_NAME} error: {exc}"
        question_spec = {
            "type": "choice",
            "instructions": question,
            "criteria": criteria,
        }
        return await self._ask_one(
            question, question_spec, state, __event_emitter__, __messages__, __files__
        )

    async def system_one_score(
        self,
        question: str,
        levels: list,
        state: str = "",
        __event_emitter__=None,
        __messages__: Optional[list] = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Ask a System One model to place content on an ordered scale you define,
        returning a probability-weighted score between 0 and len(levels)-1.
        Good for severity, sentiment intensity, risk and rubric grading.

        :param question: What is being rated, e.g. "How frustrated is the customer?".
        :param levels: Ordered list of level labels from lowest to highest, e.g. ["Calm", "Frustrated", "Very angry"]. Between 2 and 10 levels.
        :param state: The content to evaluate. Leave empty to use attached files, or failing that the recent conversation.
        :return: The weighted score, the closest level, confidence, and the distribution.
        """
        try:
            criteria = self._coerce_levels(levels)
        except SystemOneError as exc:
            return f"{self.valves.PROVIDER_NAME} error: {exc}"
        question_spec = {
            "type": "score",
            "instructions": question,
            "criteria": criteria,
        }
        return await self._ask_one(
            question, question_spec, state, __event_emitter__, __messages__, __files__
        )

    async def system_one_ask(
        self,
        questions: dict,
        state: str = "",
        __event_emitter__=None,
        __messages__: Optional[list] = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Ask a System One model several typed questions about the same content in a
        single request. The questions are evaluated in parallel with barely any added
        latency, so use this whenever more than one question is asked about the same content.

        Each entry in `questions` is keyed by a name you choose, and its value is an
        object with a "type" of "noul", "choice" or "score", an "instructions" string,
        and for choice a "criteria" object of option to description, or for score a
        "criteria" array of 2 to 10 ordered level labels.

        Example: {"urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {"type": "choice", "instructions": "Who should handle it?",
        "criteria": {"billing": "Payments and refunds", "tech": "Bugs and outages"}}}

        :param questions: Map of question name to a typed question object.
        :param state: The content to evaluate. Leave empty to use attached files, or failing that the recent conversation.
        :return: One formatted answer per question.
        """
        name = self.valves.PROVIDER_NAME
        try:
            resolved = self._resolve_state(state, __messages__, __files__)
            spec = self._coerce_questions(questions)
            payload = self._payload(resolved, spec)
            await self._emit(
                __event_emitter__, f"Asking {name} {len(spec)} question(s)...", False
            )
            data = await self._request(payload, __event_emitter__)
        except SystemOneError as exc:
            await self._emit(__event_emitter__, f"{name} request failed", True)
            return f"{name} error: {exc}"
        await self._emit(__event_emitter__, f"{name} answered", True)
        labels = {k: self._label(v.get("instructions"), k) for k, v in spec.items()}
        return self._render(data, labels)

    async def system_one_classify_batch(
        self,
        question: str,
        options: dict,
        items: list,
        __event_emitter__=None,
    ) -> str:
        """
        Classify many items with the same question, running the calls in parallel.
        Use this for lists, pasted rows, or any "label each of these" request.

        :param question: What is being decided for every item, e.g. "Which team should handle this?".
        :param options: Mapping of option name to a short description. 2 to 26 options.
        :param items: The list of text items to classify, one per row.
        :return: A table of per-item results plus a tally of each label.
        """
        name = self.valves.PROVIDER_NAME
        try:
            criteria = self._coerce_options(options)
            texts = self._coerce_items(items)
        except SystemOneError as exc:
            return f"{name} error: {exc}"

        question_spec = {
            "answer": {"type": "choice", "instructions": question, "criteria": criteria}
        }
        semaphore = asyncio.Semaphore(max(1, self.valves.BATCH_CONCURRENCY))
        done = 0
        total = len(texts)
        await self._emit(__event_emitter__, f"Classifying {total} items...", False)

        async def classify(session, index: int, text: str):
            nonlocal done
            async with semaphore:
                try:
                    payload = self._payload(self._truncate(text), question_spec)
                    data = await self._request(payload, None, session=session)
                    result: dict[str, Any] = {"index": index, "text": text, "data": data}
                except SystemOneError as exc:
                    result = {"index": index, "text": text, "error": str(exc)}
            done += 1
            if done % 10 == 0 or done == total:
                await self._emit(
                    __event_emitter__, f"Classified {done}/{total}...", False
                )
            return result

        timeout = aiohttp.ClientTimeout(total=max(1, self.valves.REQUEST_TIMEOUT))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results = await asyncio.gather(
                *(classify(session, i, t) for i, t in enumerate(texts, start=1))
            )
        await self._emit(__event_emitter__, f"Classified {total} items", True)
        return self._render_batch(question, results, criteria)

    # ------------------------------------------------------------------
    # Shared single-question flow
    # ------------------------------------------------------------------

    async def _ask_one(
        self,
        label: str,
        question_spec: dict,
        state: str,
        emitter,
        messages: Optional[list],
        files: Optional[list],
    ) -> str:
        name = self.valves.PROVIDER_NAME
        try:
            resolved = self._resolve_state(state, messages, files)
            payload = self._payload(resolved, {"answer": question_spec})
            await self._emit(emitter, f"Asking {name}...", False)
            data = await self._request(payload, emitter)
        except SystemOneError as exc:
            await self._emit(emitter, f"{name} request failed", True)
            return f"{name} error: {exc}"
        await self._emit(emitter, f"{name} answered", True)
        return self._render(data, {"answer": label})

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _payload(self, state: Any, questions: dict) -> dict:
        payload: dict[str, Any] = {
            "state": state,
            "model": self.valves.MODEL.strip(),
            "questions": questions,
        }
        extra = self._json_valve(self.valves.EXTRA_BODY, "EXTRA_BODY")
        for key, value in extra.items():
            payload.setdefault(key, value)
        return payload

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        key = self.valves.API_KEY.strip()
        if key:
            header = (self.valves.AUTH_HEADER or "Authorization").strip()
            scheme = (self.valves.AUTH_SCHEME or "").strip()
            headers[header] = f"{scheme} {key}" if scheme else key
        for k, v in self._json_valve(self.valves.EXTRA_HEADERS, "EXTRA_HEADERS").items():
            headers[str(k)] = str(v)
        return headers

    @staticmethod
    def _json_valve(raw: str, name: str) -> dict:
        raw = (raw or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemOneError(f"valve {name} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemOneError(f"valve {name} must be a JSON object.")
        return parsed

    def _url(self) -> str:
        base = self.valves.BASE_URL.strip().rstrip("/")
        path = (self.valves.ENDPOINT_PATH or "").strip().lstrip("/")
        if not base:
            raise SystemOneError("no BASE_URL configured in this tool's valves.")
        return f"{base}/{path}" if path else base

    async def _request(
        self,
        payload: dict,
        emitter,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> dict:
        if not self.valves.MODEL.strip():
            raise SystemOneError("no MODEL configured in this tool's valves.")
        limit = self.valves.MAX_REQUEST_BYTES
        if limit > 0:
            size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            if size > limit:
                raise SystemOneError(
                    f"request body is {size} bytes, over the {limit}-byte limit. Shorten the state or raise MAX_REQUEST_BYTES."
                )
        if session is not None:
            return await self._send(session, payload, emitter)
        timeout = aiohttp.ClientTimeout(total=max(1, self.valves.REQUEST_TIMEOUT))
        async with aiohttp.ClientSession(timeout=timeout) as owned:
            return await self._send(owned, payload, emitter)

    async def _send(
        self, session: aiohttp.ClientSession, payload: dict, emitter
    ) -> dict:
        url = self._url()
        headers = self._headers()
        name = self.valves.PROVIDER_NAME
        attempts = max(0, self.valves.MAX_RETRIES) + 1
        last = "unknown error"

        for attempt in range(attempts):
            retry_after: Optional[float] = None
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        try:
                            return json.loads(body)
                        except json.JSONDecodeError as exc:
                            raise SystemOneError(
                                f"{name} returned a non-JSON body: {self._short(body)}"
                            ) from exc
                    if resp.status in (401, 403):
                        raise SystemOneError(
                            f"HTTP {resp.status} - the provider rejected the credentials. Check API_KEY, AUTH_HEADER and AUTH_SCHEME in the tool valves."
                        )
                    if resp.status == 404:
                        raise SystemOneError(
                            f"HTTP 404 - not found at {url}. Check BASE_URL and ENDPOINT_PATH, and that the model name in MODEL exists on the server."
                        )
                    if resp.status == 413:
                        raise SystemOneError(
                            f"HTTP 413 - request body too large: {self._short(body)}"
                        )
                    if resp.status == 400:
                        raise SystemOneError(
                            f"HTTP 400 - the request was rejected: {self._short(body)}. If this mentions context, increase the model's context length on the server."
                        )
                    if resp.status == 422:
                        raise SystemOneError(
                            f"HTTP 422 - the request body was rejected: {self._short(body)}"
                        )
                    if resp.status in (429, 529) or resp.status >= 500:
                        last = f"HTTP {resp.status}: {self._short(body)}"
                        retry_after = self._retry_after(resp.headers.get("Retry-After"))
                    else:
                        raise SystemOneError(f"HTTP {resp.status}: {self._short(body)}")
            except asyncio.TimeoutError:
                last = f"timed out after {self.valves.REQUEST_TIMEOUT}s"
            except aiohttp.ClientError as exc:
                last = f"network error: {exc}"

            if attempt == attempts - 1:
                break
            delay = retry_after
            if delay is None:
                delay = self.valves.BACKOFF_FACTOR * (2**attempt)
            delay = max(0.1, delay * (1 + random.random() * 0.2))
            await self._emit(emitter, f"{name} unavailable ({last}), retrying...", False)
            await asyncio.sleep(delay)

        raise SystemOneError(f"request failed after {attempts} attempt(s) - {last}")

    @staticmethod
    def _retry_after(header: Optional[str]) -> Optional[float]:
        if not header:
            return None
        try:
            return min(60.0, max(0.0, float(header.strip())))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _short(text: str, limit: int = 300) -> str:
        collapsed = re.sub(r"\s+", " ", (text or "")).strip()
        if not collapsed:
            return "(empty)"
        return collapsed[:limit] + ("..." if len(collapsed) > limit else "")

    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------

    def _resolve_state(
        self, state: Any, messages: Optional[list], files: Optional[list]
    ) -> Any:
        if isinstance(state, (dict, list)) and state:
            return self._truncate_structured(state)

        explicit = (state or "").strip() if isinstance(state, str) else ""
        if explicit:
            if self.valves.PARSE_JSON_STATE and explicit[0] in "{[":
                try:
                    parsed = json.loads(explicit)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, (dict, list)) and parsed:
                    return self._truncate_structured(parsed)
            return self._truncate(explicit)

        chunks = []
        for entry in files or []:
            fname, content = self._file_text(entry)
            if content:
                chunks.append(f"# {fname}\n{content}")
        if chunks:
            return self._truncate("\n\n".join(chunks))

        window = max(1, self.valves.CONTEXT_MESSAGES)
        lines = []
        for message in (messages or [])[-window:]:
            if not isinstance(message, dict):
                continue
            text = self._message_text(message.get("content"))
            if text:
                lines.append(f"{message.get('role', 'user')}: {text}")
        if lines:
            return self._truncate("\n".join(lines), keep_tail=True)

        raise SystemOneError(
            "nothing to evaluate. Pass the text as 'state', attach a file, or ask about a conversation that has messages."
        )

    def _truncate_structured(self, value: Any) -> Any:
        # Structured state can't be cut mid-string safely, so if it is over the
        # cap it is sent as truncated JSON text rather than silently exceeding it.
        serialized = json.dumps(value, ensure_ascii=False)
        if len(serialized) <= max(200, self.valves.MAX_STATE_CHARS):
            return value
        return self._truncate(serialized)

    @staticmethod
    def _file_text(entry: Any) -> tuple:
        if not isinstance(entry, dict):
            return ("file", "")
        inner = entry.get("file") if isinstance(entry.get("file"), dict) else entry
        fname = str(
            inner.get("filename") or inner.get("name") or entry.get("name") or "file"
        )
        data = inner.get("data") if isinstance(inner.get("data"), dict) else {}
        content = data.get("content") or inner.get("content") or ""
        return (fname, content if isinstance(content, str) else "")

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(p for p in parts if p).strip()
        return ""

    def _truncate(self, text: str, keep_tail: bool = False) -> str:
        limit = max(200, self.valves.MAX_STATE_CHARS)
        if len(text) <= limit:
            return text
        if keep_tail:
            return "[...truncated...]\n" + text[-limit:]
        return text[:limit] + "\n[...truncated...]"

    @staticmethod
    def _as_obj(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _label(instructions: Any, fallback: str) -> str:
        # Instructions may be a string, or a structured object/array whose
        # "question" field holds the actual question.
        if isinstance(instructions, str) and instructions.strip():
            return instructions.strip()
        if isinstance(instructions, dict) and isinstance(
            instructions.get("question"), str
        ):
            return instructions["question"].strip()
        return fallback

    def _coerce_options(self, options: Any) -> dict:
        parsed = self._as_obj(options)
        if isinstance(parsed, list):
            parsed = {str(item): str(item) for item in parsed}
        if not isinstance(parsed, dict) or not parsed:
            raise SystemOneError(
                "'options' must be an object mapping each option name to a short description."
            )
        # Descriptions may be a string, a structured value, or null (no extra detail).
        criteria = {
            str(k): (v if isinstance(v, (dict, list)) or v is None else str(v))
            for k, v in parsed.items()
        }
        cap = max(2, self.valves.MAX_CHOICE_OPTIONS)
        if not 2 <= len(criteria) <= cap:
            raise SystemOneError(
                f"a choice question needs between 2 and {cap} options, got {len(criteria)}."
            )
        return criteria

    def _coerce_levels(self, levels: Any) -> list:
        parsed = self._as_obj(levels)
        if isinstance(parsed, dict):
            parsed = list(parsed.values())
        if not isinstance(parsed, list):
            raise SystemOneError(
                "'levels' must be an ordered list of level labels, lowest first."
            )
        labels = [
            item if isinstance(item, (dict, list)) else str(item)
            for item in parsed
            if str(item).strip()
        ]
        cap = max(MIN_SCORE_LEVELS, self.valves.MAX_SCORE_LEVELS)
        if not MIN_SCORE_LEVELS <= len(labels) <= cap:
            raise SystemOneError(
                f"a score question needs between {MIN_SCORE_LEVELS} and {cap} levels, got {len(labels)}."
            )
        return labels

    def _coerce_questions(self, questions: Any) -> dict:
        parsed = self._as_obj(questions)
        if not isinstance(parsed, dict) or not parsed:
            raise SystemOneError(
                "'questions' must be an object mapping a question name to a typed question."
            )
        if len(parsed) > max(1, self.valves.MAX_QUESTIONS):
            raise SystemOneError(
                f"{len(parsed)} questions exceeds the limit of {self.valves.MAX_QUESTIONS} per request."
            )
        spec = {}
        for key, raw in parsed.items():
            question = self._as_obj(raw)
            if not isinstance(question, dict):
                raise SystemOneError(f"question '{key}' must be an object.")
            qtype = str(question.get("type", "")).lower()
            if qtype not in VALID_TYPES:
                raise SystemOneError(
                    f"question '{key}' has type '{qtype or 'missing'}', expected one of {', '.join(VALID_TYPES)}."
                )
            instructions = question.get("instructions")
            if not instructions:
                raise SystemOneError(f"question '{key}' is missing 'instructions'.")
            built: dict[str, Any] = {"type": qtype, "instructions": instructions}
            criteria = question.get("criteria")
            if qtype == "choice":
                built["criteria"] = self._coerce_options(criteria)
            elif qtype == "score":
                built["criteria"] = self._coerce_levels(criteria)
            elif criteria is not None:
                built["criteria"] = self._as_obj(criteria)
            spec[str(key)] = built
        return spec

    def _coerce_items(self, items: Any) -> list:
        parsed = self._as_obj(items)
        if isinstance(parsed, str):
            parsed = [line for line in parsed.splitlines() if line.strip()]
        if not isinstance(parsed, list) or not parsed:
            raise SystemOneError("'items' must be a non-empty list of strings to classify.")
        texts = [str(item).strip() for item in parsed if str(item).strip()]
        if not texts:
            raise SystemOneError("'items' contained no non-empty entries.")
        cap = max(1, self.valves.BATCH_MAX_ITEMS)
        if len(texts) > cap:
            raise SystemOneError(
                f"{len(texts)} items exceeds the batch limit of {cap}. Split the list or raise BATCH_MAX_ITEMS."
            )
        return texts

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

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

    def _render(self, data: dict, labels: dict) -> str:
        if self.valves.RAW_JSON_OUTPUT:
            return json.dumps(data, indent=2, ensure_ascii=False)
        answers = data.get("answers") or {}
        if not answers:
            return (
                f"{self.valves.PROVIDER_NAME} returned no answers. Raw response: "
                f"{self._short(json.dumps(data))}"
            )
        lines = []
        for key, answer in answers.items():
            lines.append(self._render_answer(key, labels.get(key, key), answer))
        lines.append("")
        lines.append(self._footer(data))
        return "\n".join(lines)

    def _render_answer(self, key: str, label: str, answer: Any) -> str:
        if not isinstance(answer, dict):
            return f"**{label}** -> {answer}"
        atype = answer.get("type")

        if atype == "noul":
            p = self._num(answer.get("noul"))
            verdict = "yes" if p is not None and p >= NOUL_THRESHOLD else "no"
            shown = f"{p:.2f}" if p is not None else "n/a"
            return f"**{label}** -> **{verdict}** (probability {shown})"

        if atype == "choice":
            choice = answer.get("choice", "n/a")
            out = f"**{label}** -> **{choice}**{self._confidence(answer)}"
            dist = self._distribution(answer.get("probabilities"))
            return out + (f"\n  {dist}" if dist else "")

        if atype == "score":
            score = self._num(answer.get("score"))
            legend = answer.get("legend") if isinstance(answer.get("legend"), dict) else {}
            level = legend.get(str(round(score))) if score is not None else None
            maximum = len(legend) - 1 if legend else None
            shown = f"{score:.2f}" if score is not None else "n/a"
            scale = f"/{maximum}" if maximum is not None else ""
            level_text = f' - closest level: "{level}"' if level else ""
            out = (
                f"**{label}** -> **{shown}{scale}**{level_text}"
                f"{self._confidence(answer)}"
            )
            dist = self._distribution(answer.get("probabilities"), legend)
            return out + (f"\n  {dist}" if dist else "")

        return f"**{label}** ({key}) -> {json.dumps(answer, ensure_ascii=False)}"

    def _render_batch(self, question: str, results: list, criteria: dict) -> str:
        if self.valves.RAW_JSON_OUTPUT:
            return json.dumps(results, indent=2, ensure_ascii=False)
        rows = ["| # | Item | Answer | Confidence |", "|---|---|---|---|"]
        tally = {name: 0 for name in criteria}
        failures = []
        input_tokens = 0
        output_tokens = 0
        total_cost = 0.0
        for result in sorted(results, key=lambda r: r["index"]):
            preview = self._cell(result["text"])
            if "error" in result:
                failures.append(f"{result['index']}: {result['error']}")
                rows.append(f"| {result['index']} | {preview} | error | - |")
                continue
            data = result["data"]
            usage = data.get("usage") or {}
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            total_cost += self._num(usage.get("cost")) or 0.0
            answer = (data.get("answers") or {}).get("answer") or {}
            choice = str(answer.get("choice", "n/a"))
            tally[choice] = tally.get(choice, 0) + 1
            conf = self._num(answer.get("confidence"))
            conf_text = f"{conf:.2f}" if conf is not None else "-"
            rows.append(f"| {result['index']} | {preview} | {choice} | {conf_text} |")

        summary = ", ".join(f"{name}: {count}" for name, count in tally.items() if count)
        out = [
            f"**{question}** - {len(results)} item(s) classified by {self.valves.PROVIDER_NAME}",
            "",
        ]
        out.extend(rows)
        out.append("")
        out.append(f"Tally: {summary or 'none'}")
        if failures:
            out.append(f"Failed: {len(failures)} - " + "; ".join(failures[:5]))
        footer = (
            f"Model: {self.valves.MODEL} | "
            f"tokens in/out: {input_tokens}/{output_tokens}"
        )
        if total_cost:
            footer += f" | cost: ${total_cost:.6f}"
        out.append(footer)
        return "\n".join(out)

    @staticmethod
    def _cell(text: str, limit: int = 70) -> str:
        flat = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
        return flat[:limit] + ("..." if len(flat) > limit else "")

    @staticmethod
    def _num(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _confidence(self, answer: dict) -> str:
        conf = self._num(answer.get("confidence"))
        return f" (confidence {conf:.2f})" if conf is not None else ""

    def _distribution(self, probabilities: Any, legend: Any = None) -> str:
        if not isinstance(probabilities, dict) or not probabilities:
            return ""
        parts = []
        for key, value in sorted(
            probabilities.items(), key=lambda kv: -(self._num(kv[1]) or 0)
        ):
            p = self._num(value)
            if p is None or p < 0.005:
                continue
            label = (
                f"{key} {legend[key]}"
                if isinstance(legend, dict) and key in legend
                else key
            )
            parts.append(f"{label} {p * 100:.0f}%")
        return " | ".join(parts)

    def _footer(self, data: dict) -> str:
        usage = data.get("usage") or {}
        parts = [f"{self.valves.PROVIDER_NAME} {data.get('model', self.valves.MODEL)}"]
        if usage:
            parts.append(
                "tokens in/out: "
                f"{usage.get('input_tokens', '?')}/{usage.get('output_tokens', '?')}"
            )
            cost = self._num(usage.get("cost"))
            if cost is not None:
                parts.append(f"cost: ${cost:.6f}")
        return "_" + " | ".join(parts) + "_"
