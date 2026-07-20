"""The shared relation vocabulary (DESIGN.md §8.1).

Extending the tables here is the documented way to teach the rules extractor and
the query relation-inference map new relations.
"""

from __future__ import annotations

import re

# --- Relation trigger phrases for EXTRACTION -------------------------------
# Each entry: (surface phrase, relation, flip) where flip=True means
# "src and dst are swapped" (e.g. "A manages B" => B is MANAGED_BY A).
EXTRACT_TRIGGERS: list[tuple[str, str, bool]] = [
    # MANAGED_BY
    ("reports to", "MANAGED_BY", False),
    ("is managed by", "MANAGED_BY", False),
    ("managed by", "MANAGED_BY", False),
    ("manages", "MANAGED_BY", True),
    # MEMBER_OF
    ("is a member of", "MEMBER_OF", False),
    ("member of", "MEMBER_OF", False),
    ("is on the", "MEMBER_OF", False),
    ("on the team", "MEMBER_OF", False),
    ("belongs to", "MEMBER_OF", False),
    ("part of", "MEMBER_OF", False),
    # OWNS
    ("is responsible for", "OWNS", False),
    ("responsible for", "OWNS", False),
    ("is the owner of", "OWNS", False),
    ("owner of", "OWNS", False),
    ("maintains", "OWNS", False),
    ("owns", "OWNS", False),
    # USES
    ("depends on", "USES", False),
    ("is built on", "USES", False),
    ("built on", "USES", False),
    ("relies on", "USES", False),
    ("uses", "USES", False),
    # LOCATED_IN
    ("is located in", "LOCATED_IN", False),
    ("located in", "LOCATED_IN", False),
    ("is based in", "LOCATED_IN", False),
    ("based in", "LOCATED_IN", False),
    # IS_A
    ("works as", "IS_A", False),
    ("role is", "IS_A", False),
    ("is an", "IS_A", False),
    ("is a", "IS_A", False),
    # RELATED_TO
    ("is related to", "RELATED_TO", False),
    ("related to", "RELATED_TO", False),
]

# Fact-sentence templates (DESIGN.md §8.1).
FACT_TEMPLATES: dict[str, str] = {
    "MANAGED_BY": "{src} is managed by {dst}.",
    "MEMBER_OF": "{src} is a member of {dst}.",
    "OWNS": "{src} owns {dst}.",
    "USES": "{src} uses {dst}.",
    "IS_A": "{src} is a {dst}.",
    "LOCATED_IN": "{src} is located in {dst}.",
    "RELATED_TO": "{src} is related to {dst}.",
}

# --- Keyword -> relation map for QUERY relation inference -------------------
# Ordered longest-first at match time. Includes question phrasings.
QUERY_TRIGGERS: list[tuple[str, str]] = [
    ("report to", "MANAGED_BY"),
    ("reports to", "MANAGED_BY"),
    ("managed by", "MANAGED_BY"),
    ("manager", "MANAGED_BY"),
    ("manage", "MANAGED_BY"),
    ("member of", "MEMBER_OF"),
    ("belong to", "MEMBER_OF"),
    ("part of", "MEMBER_OF"),
    ("team", "MEMBER_OF"),
    ("owned by", "OWNS"),
    ("owned", "OWNS"),
    ("owner", "OWNS"),
    ("owns", "OWNS"),
    ("own", "OWNS"),
    ("responsible for", "OWNS"),
    ("maintains", "OWNS"),
    ("maintain", "OWNS"),
    ("depends on", "USES"),
    ("depend on", "USES"),
    ("built on", "USES"),
    ("relies on", "USES"),
    ("used by", "USES"),
    ("used", "USES"),
    ("using", "USES"),
    ("uses", "USES"),
    ("use", "USES"),
    ("located in", "LOCATED_IN"),
    ("based in", "LOCATED_IN"),
    ("location", "LOCATED_IN"),
    ("where", "LOCATED_IN"),
    ("role", "IS_A"),
    ("works as", "IS_A"),
    ("job", "IS_A"),
    ("is a", "IS_A"),
]

# Relation-position default node types (keyword override in TYPE_KEYWORDS wins).
POSITION_TYPES: dict[str, tuple[str, str]] = {
    "MANAGED_BY": ("Person", "Person"),
    "MEMBER_OF": ("Person", "Team"),
    "OWNS": ("Person", "Thing"),
    "USES": ("Thing", "Thing"),
    "IS_A": ("Person", "Thing"),
    "LOCATED_IN": ("Thing", "Thing"),
    "RELATED_TO": ("Thing", "Thing"),
}

# Substring -> node type overrides (checked before position defaults).
TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("team", "Team"),
    ("service", "Service"),
    ("project", "Project"),
    ("database", "Service"),
    ("system", "Service"),
    ("platform", "Service"),
]


def slug(name: str) -> str:
    """Lowercase + hyphenate a display name into an id slug."""
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    return s.strip("-")


def node_id(type_: str, name: str) -> str:
    return f"{type_.lower()}:{slug(name)}"


def infer_type(name: str, rel: str, position: str) -> str:
    """Infer a node type from name keywords, else the relation-position default."""
    low = name.lower()
    for kw, typ in TYPE_KEYWORDS:
        if kw in low:
            return typ
    default = POSITION_TYPES.get(rel, ("Thing", "Thing"))
    return default[0] if position == "src" else default[1]


def fact_sentence(rel: str, src_name: str, dst_name: str) -> str:
    tmpl = FACT_TEMPLATES.get(rel, "{src} is related to {dst}.")
    return tmpl.format(src=src_name, dst=dst_name)
