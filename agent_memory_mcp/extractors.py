"""Text -> (entities, relations) extractors (DESIGN.md §6, §8.1).

`RulesExtractor` is the deterministic, offline default. `LlmExtractor` (optional,
Anthropic) is filled in at build step 12 and import-guarded.
"""

from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable

from .models import Entity, Relation, Statement
from .vocab import EXTRACT_TRIGGERS, infer_type, node_id

# Tokens stripped from the edges of an entity noun phrase.
_EDGE_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "its",
    "his",
    "her",
    "their",
    "our",
    "my",
    "this",
    "that",
    "these",
    "those",
    "of",
}

# Triggers sorted so the most specific (longest) phrase matches first.
_TRIGGERS_SORTED = sorted(EXTRACT_TRIGGERS, key=lambda t: len(t[0]), reverse=True)


@runtime_checkable
class Extractor(Protocol):
    def extract(self, text: str) -> list[Statement]: ...


def _split_sentences(text: str) -> list[str]:
    # split on line breaks and sentence terminators
    rough = re.split(r"[\n]+|(?<=[.!?])\s+", text)
    return [s.strip() for s in rough if s.strip()]


def _split_clauses(sentence: str) -> list[tuple[str, Optional[str]]]:
    """Split a sentence into clauses.

    Returns (clause_text, carried_subject). For relative clauses introduced by
    which/who/that the carried subject is filled in by the caller.
    """
    # keep the delimiter so we know whether it was a relative pronoun (coref)
    parts = re.split(r"(,?\s+and\s+|,\s+which\s+|,\s+who\s+|,\s+that\s+)", sentence)
    clauses: list[tuple[str, bool]] = []
    coref_next = False
    for i, part in enumerate(parts):
        if i % 2 == 1:  # delimiter
            coref_next = bool(re.search(r"which|who|that", part))
            continue
        p = part.strip().strip(",").strip()
        if p:
            clauses.append((p, coref_next))
            coref_next = False
    # second pass: assign carried subjects for coref clauses
    out: list[tuple[str, Optional[str]]] = []
    prev_object: Optional[str] = None
    for text, is_coref in clauses:
        carried = prev_object if is_coref else None
        out.append((text, carried))
        prev_object = None  # updated by extractor after it parses the object
    return out


def _clean_name(phrase: str) -> str:
    phrase = phrase.strip().strip(".,;:!?").strip()
    # drop possessive markers
    phrase = re.sub(r"'s\b", "", phrase)
    tokens = phrase.split()
    while tokens and tokens[0].lower() in _EDGE_STOP:
        tokens.pop(0)
    while tokens and tokens[-1].lower() in _EDGE_STOP:
        tokens.pop()
    return " ".join(tokens).strip()


def _find_trigger(clause: str) -> Optional[tuple[str, str, bool, int, int]]:
    low = clause.lower()
    for phrase, rel, flip in _TRIGGERS_SORTED:
        m = re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", low)
        if m:
            return phrase, rel, flip, m.start(), m.end()
    return None


class RulesExtractor:
    """Deterministic pattern/verb extractor over the §8.1 relation vocabulary."""

    def extract(self, text: str) -> list[Statement]:
        statements: list[Statement] = []
        for sentence in _split_sentences(text):
            entities: list[Entity] = []
            relations: list[Relation] = []
            prev_object_name: Optional[str] = None

            clause_specs = _split_clauses(sentence)
            for idx, (clause, carried) in enumerate(clause_specs):
                # carried subject from a preceding "which/who/that"
                if carried is None and idx > 0:
                    carried = None
                subject_override = (
                    carried if carried else prev_object_name if _looks_headless(clause) else None
                )

                parsed = self._parse_clause(clause, subject_override)
                if parsed is None:
                    prev_object_name = None
                    continue
                src_ent, dst_ent, rel = parsed
                entities.append(src_ent)
                entities.append(dst_ent)
                relations.append(rel)
                prev_object_name = dst_ent.name

            # de-duplicate entities by id, preserving order
            seen: set[str] = set()
            uniq_entities = []
            for e in entities:
                if e.id not in seen:
                    seen.add(e.id)
                    uniq_entities.append(e)

            statements.append(Statement(text=sentence, entities=uniq_entities, relations=relations))
        return statements

    def _parse_clause(
        self, clause: str, subject_override: Optional[str]
    ) -> Optional[tuple[Entity, Entity, Relation]]:
        found = _find_trigger(clause)
        if not found:
            return None
        _phrase, rel, flip, start, end = found
        left = clause[:start]
        right = clause[end:]

        subj_name = _clean_name(left)
        obj_name = _clean_name(right)
        if subject_override and not subj_name:
            subj_name = subject_override
        if not subj_name or not obj_name:
            return None

        if flip:
            subj_name, obj_name = obj_name, subj_name

        src_type = infer_type(subj_name, rel, "src")
        dst_type = infer_type(obj_name, rel, "dst")
        src = Entity(id=node_id(src_type, subj_name), type=src_type, name=subj_name)
        dst = Entity(id=node_id(dst_type, obj_name), type=dst_type, name=obj_name)
        relation = Relation(src=src.id, rel=rel, dst=dst.id)
        return src, dst, relation


def _looks_headless(clause: str) -> bool:
    """Heuristic: does this clause begin directly with a trigger verb (no subject)?"""
    found = _find_trigger(clause)
    if not found:
        return False
    _phrase, _rel, _flip, start, _end = found
    return _clean_name(clause[:start]) == ""


class LlmExtractor:
    """Optional Anthropic-backed structured extractor (see build step 12)."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        from .config import load_config

        cfg = load_config()
        self.model = model or cfg.model
        self._api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import os

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'llm' extractor requires the anthropic package. "
                "Install it with: pip install 'agent-memory-mcp[llm]'"
            ) from exc
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "The 'llm' extractor requires ANTHROPIC_API_KEY to be set in the environment."
            )
        self._client = anthropic.Anthropic(api_key=key)

    def extract(self, text: str) -> list[Statement]:
        import json

        self._ensure_client()
        prompt = _LLM_PROMPT.format(
            vocab=", ".join(sorted(set(r for _, r, _ in EXTRACT_TRIGGERS))), text=text
        )
        for _attempt in range(2):  # single bounded retry
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = _first_text_block(resp)
                data = json.loads(_extract_json(raw))
                return self._to_statements(text, data)
            except Exception:  # noqa: BLE001 - bounded retry then fall back
                continue
        # On repeated failure, degrade gracefully to raw storage (never throw).
        return [Statement(text=s) for s in _split_sentences(text)] or [Statement(text=text)]

    def _to_statements(self, text: str, data: dict) -> list[Statement]:
        entities: list[Entity] = []
        relations: list[Relation] = []
        by_name: dict[str, Entity] = {}
        for rel_obj in data.get("relations", []):
            src_name = str(rel_obj.get("src", "")).strip()
            dst_name = str(rel_obj.get("dst", "")).strip()
            rel = str(rel_obj.get("rel", "RELATED_TO")).strip().upper()
            if not src_name or not dst_name:
                continue
            src_type = infer_type(src_name, rel, "src")
            dst_type = infer_type(dst_name, rel, "dst")
            src = Entity(id=node_id(src_type, src_name), type=src_type, name=src_name)
            dst = Entity(id=node_id(dst_type, dst_name), type=dst_type, name=dst_name)
            by_name[src.id] = src
            by_name[dst.id] = dst
            relations.append(Relation(src=src.id, rel=rel, dst=dst.id))
        entities = list(by_name.values())
        return [Statement(text=text, entities=entities, relations=relations)]


_LLM_PROMPT = """Extract entities and typed relations from the text below.
Use ONLY these relation types: {vocab}.
Return STRICT JSON: {{"relations": [{{"src": "Name", "rel": "REL", "dst": "Name"}}]}}.
Use display names (e.g. "Dana", "Search Team") for src/dst, not ids.
If nothing can be structured, return {{"relations": []}}.

TEXT:
{text}
"""


def _first_text_block(resp) -> str:
    """Return the first text block's text from an Anthropic Messages response."""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", "text") == "text" and getattr(block, "text", None):
            return block.text
    # fall back to the first block's text attribute
    content = getattr(resp, "content", None)
    if content:
        return getattr(content[0], "text", "") or ""
    return ""


def _extract_json(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return "{}"
    return raw[start : end + 1]


def get_extractor(name: str = "rules", model: Optional[str] = None) -> Extractor:
    """Factory: ``rules`` (default, offline) or ``llm`` (optional, Anthropic)."""
    name = (name or "rules").lower()
    if name == "rules":
        return RulesExtractor()
    if name == "llm":
        return LlmExtractor(model=model)
    raise ValueError(f"Unknown extractor: {name!r} (expected 'rules' or 'llm')")
