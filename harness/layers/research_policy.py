"""Evidence-led research controls for real-model runs.

The practice MockModel follows a fixed plan, so this policy deliberately leaves
that path untouched. Real models receive a short, one-turn research nudge and a
premature FINAL can be deferred into a concrete search/fetch action while tool
budget remains.
"""

from __future__ import annotations

import json
import re

from arena.model import MockModel, ModelResponse, is_degraded, parse_output, render_action

from harness.middleware import Middleware


LEDGER_KEY = "research_policy.ledger"
DEFAULT_MIN_FULL_DOCS = 2
DEFAULT_MAX_FINAL_DEFERRALS = 4

_STOPWORDS = {
    "bao", "cac", "cho", "cua", "duoc", "gi", "hay", "khi", "khong",
    "la", "mot", "nhung", "the", "thi", "trong", "va", "voi",
}
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_FACET_RE = re.compile(r"[?;,]|\b(?:và|hoặc)\b", re.IGNORECASE)


def _normal_words(text: str) -> list[str]:
    return [word.casefold() for word in _WORD_RE.findall(text) if len(word) >= 3]


def is_fixed_mock_model(model) -> bool:
    """Recognise MockModel through the frozen runner's bounded wrappers."""
    seen = set()
    for _ in range(4):
        if isinstance(model, MockModel):
            return True
        identity = id(model)
        if identity in seen:
            break
        seen.add(identity)
        model = getattr(model, "inner", None)
        if model is None:
            break
    return False


class ResearchPolicy(Middleware):
    """Track evidence and defer unsupported early answers on real models."""

    name = "research_policy"

    def __init__(
        self,
        min_full_docs: int = DEFAULT_MIN_FULL_DOCS,
        max_final_deferrals: int = DEFAULT_MAX_FINAL_DEFERRALS,
    ) -> None:
        self.min_full_docs = max(1, int(min_full_docs))
        self.max_final_deferrals = max(0, int(max_final_deferrals))

    @staticmethod
    def _is_fixed_mock(ctx) -> bool:
        return is_fixed_mock_model(ctx.model)

    @staticmethod
    def _facets(question: str) -> list[str]:
        facets = [part.strip() for part in _FACET_RE.split(question) if part.strip()]
        return facets or ([question.strip()] if question.strip() else [])

    def _ledger(self, ctx) -> dict:
        ledger = ctx.state.get(LEDGER_KEY)
        if isinstance(ledger, dict):
            return ledger
        ledger = {
            "search_queries": [],
            "candidate_doc_ids": [],
            "fetched_doc_ids": [],
            "full_doc_ids": [],
            "question_facets": self._facets(ctx.question),
            "final_deferrals": 0,
            "tool_attempts": 0,
        }
        ctx.state[LEDGER_KEY] = ledger
        return ledger

    def before_agent(self, ctx) -> None:
        self._ledger(ctx)

    @staticmethod
    def _append_unique(items: list, value: str) -> None:
        if value and value not in items:
            items.append(value)

    def _sync_full_docs(self, ctx, ledger: dict) -> None:
        if ctx.corpus is None:
            return
        observed = ctx.observed_text
        for doc in ctx.corpus.docs:
            if doc.body and doc.body in observed:
                self._append_unique(ledger["full_doc_ids"], doc.doc_id)

    @staticmethod
    def _facet_is_covered(facet: str, observed: str) -> bool:
        words = [word for word in _normal_words(facet) if word not in _STOPWORDS]
        if not words:
            return True
        observed_words = set(_normal_words(observed))
        required = min(2, len(set(words)))
        return len(set(words) & observed_words) >= required

    def _uncovered_facets(self, ctx, ledger: dict) -> list[str]:
        return [
            facet
            for facet in ledger["question_facets"]
            if not self._facet_is_covered(facet, ctx.observed_text)
        ]

    @staticmethod
    def _can_call_tool(ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is None or ctx.tools.calls < limit - 1

    def _diversified_query(self, ctx, report: dict, ledger: dict) -> str:
        uncovered = self._uncovered_facets(ctx, ledger)
        focus = uncovered[0] if uncovered else ctx.question
        answer = report.get("answer") if isinstance(report, dict) else ""
        answer = answer if isinstance(answer, str) else ""
        prior_words = {
            word
            for query in ledger["search_queries"]
            for word in _normal_words(query)
        }
        novel = [
            word
            for word in _normal_words(answer)
            if word not in prior_words and word not in _STOPWORDS
        ]
        suffix = " ".join(dict.fromkeys(novel[:6]))
        query = f"{focus} {suffix} nguồn chính thức cập nhật ngoại lệ".strip()
        if query in ledger["search_queries"]:
            query += f" đối chiếu {len(ledger['search_queries']) + 1}"
        return query[:500]

    def before_model(self, ctx, messages):
        if self._is_fixed_mock(ctx):
            return messages
        ledger = self._ledger(ctx)
        self._sync_full_docs(ctx, ledger)
        uncovered = self._uncovered_facets(ctx, ledger)
        nudge = (
            "Research status: "
            f"{len(ledger['search_queries'])} distinct searches; "
            f"{len(ledger['full_doc_ids'])} full documents read. "
            "Before FINAL, cover every part of the question, fetch full documents, "
            "quote claims verbatim from one line, and use each claim's real doc_id."
        )
        if uncovered:
            nudge += " Evidence is still thin for: " + " | ".join(uncovered[:3])
        if ledger["search_queries"]:
            nudge += " Do not repeat these queries: " + " | ".join(
                ledger["search_queries"][-3:]
            )
        # The status contains user/model-derived text, so keep user-level
        # authority instead of promoting it into a system instruction.
        return messages + [{"role": "user", "content": nudge}]

    def wrap_tool_call(self, ctx, call, name, args):
        ledger = self._ledger(ctx)
        ledger["tool_attempts"] += 1
        result = call(name, args)
        if not result.ok or is_degraded(result.content):
            return result

        if name == "search":
            query = args.get("query")
            if isinstance(query, str):
                self._append_unique(ledger["search_queries"], query)
            try:
                payload = json.loads(result.content)
            except (TypeError, ValueError):
                payload = []
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and isinstance(item.get("doc_id"), str):
                        self._append_unique(
                            ledger["candidate_doc_ids"], item["doc_id"]
                        )
        elif name == "fetch_doc":
            doc_id = args.get("doc_id")
            if isinstance(doc_id, str):
                self._append_unique(ledger["fetched_doc_ids"], doc_id)
        return result

    def _next_candidate(self, ledger: dict) -> str | None:
        fetched = set(ledger["fetched_doc_ids"])
        return next(
            (doc_id for doc_id in ledger["candidate_doc_ids"] if doc_id not in fetched),
            None,
        )

    def after_model(self, ctx, response):
        if self._is_fixed_mock(ctx) or not self._can_call_tool(ctx):
            return response
        # Import lazily to avoid a module cycle while BudgetPolicy is imported.
        from harness.agent import _canonicalise

        parsed = parse_output(_canonicalise(response.text))
        if parsed.kind != "final":
            return response

        ledger = self._ledger(ctx)
        self._sync_full_docs(ctx, ledger)
        if ledger["final_deferrals"] >= self.max_final_deferrals:
            return response

        tool = None
        args = None
        if not ledger["search_queries"]:
            tool, args = "search", {"query": ctx.question, "k": 5}
        elif not ledger["full_doc_ids"]:
            candidate = self._next_candidate(ledger)
            if candidate:
                tool, args = "fetch_doc", {"doc_id": candidate}
            else:
                tool, args = "search", {
                    "query": self._diversified_query(ctx, parsed.final, ledger),
                    "k": 5,
                }
        elif (
            len(ledger["search_queries"]) < 2
            and len(ledger["full_doc_ids"]) < self.min_full_docs
        ):
            tool, args = "search", {
                "query": self._diversified_query(ctx, parsed.final, ledger),
                "k": 5,
            }
        elif len(ledger["full_doc_ids"]) < self.min_full_docs:
            candidate = self._next_candidate(ledger)
            if candidate:
                tool, args = "fetch_doc", {"doc_id": candidate}

        if tool is None:
            return response
        ledger["final_deferrals"] += 1
        return ModelResponse(
            text=render_action(
                "Tôi cần thu thập thêm bằng chứng trước khi kết luận.", tool, args
            ),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
