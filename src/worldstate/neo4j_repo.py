"""In-memory world-state graph backed by networkx.

Replaces the former Neo4j driver wrapper. Every node carries provenance
(one or more tool_run_ids). Engagement isolation: every node is tagged
with engagement_id; queries scoped. Secondary indices provide O(1)
lookups by kind and by (kind, field, value).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime
from typing import Any
from uuid import UUID

import networkx as nx
import structlog

# ``CREDENTIAL_SECRET_FIELDS`` is re-exported from worldstate.schema (its
# single source of truth: the Credential fields marked ``secret=True``).
# These are Fernet-encrypted in ``to_json`` and decrypted in ``from_json``
# so a serialized graph holds only ciphertext while the in-memory graph
# stays usable by tools.
from worldstate.schema import CREDENTIAL_SECRET_FIELDS, Edge, WorldStateNode

log = structlog.get_logger(__name__)


class ProvenanceRequiredError(ValueError):
    """A node or edge without provenance was offered for write -- refused."""


class EngagementMismatchError(ValueError):
    """A node/edge was offered with an engagement_id that doesn't match the parent -- refused."""


ProvenanceValidator = Callable[[UUID, list[UUID]], Awaitable[bool]]
"""A function ``(engagement_id, tool_run_ids) -> bool`` that returns True iff all
tool_run_ids exist for the given engagement."""


# Fields that are indexed in ``_by_kind_field`` for O(1) lookups.
# Each entry is ``(node_kind, field_name)``.
INDEXED_FIELDS: list[tuple[str, str]] = [
    ("Host", "ip"),
    ("Service", "port"),
    ("Service", "host_id"),
    ("Credential", "username"),
    ("Credential", "domain"),
    ("WebApp", "base_url"),
    ("WebApp", "service_id"),
    ("Finding", "severity"),
    ("Session", "host_id"),
    ("Endpoint", "webapp_id"),
]


class InMemoryGraphRepo:
    """networkx-backed world-state graph with secondary indices.

    Drop-in replacement for the former ``Neo4jRepo``. The public async
    API surface is preserved so callers (agents, queries, API endpoints)
    work without changes.
    """

    def __init__(
        self,
        graph: nx.DiGraph | None = None,
        provenance_validator: ProvenanceValidator | None = None,
    ) -> None:
        self.G = graph or nx.DiGraph()
        self._provenance_validator = provenance_validator
        # Secondary indices
        self._by_kind: dict[str, set[str]] = defaultdict(set)
        self._by_kind_field: dict[tuple[str, str, Any], set[str]] = defaultdict(set)

    # ------------------------------------------------------------------ #
    # Serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_json(self) -> str:
        """Serialize the graph to a JSON string (node-link format).

        Credential secret fields are Fernet-encrypted in the serialized copy
        so the on-disk checkpoint never holds plaintext. The live in-memory
        node attrs are NOT mutated -- we operate on the node_link_data dict.
        """
        data = nx.node_link_data(self.G)
        _encrypt_credential_secrets(data)
        return json.dumps(data)

    @classmethod
    def from_json(
        cls,
        data: str,
        provenance_validator: ProvenanceValidator | None = None,
    ) -> InMemoryGraphRepo:
        """Deserialize from a JSON string and rebuild indices.

        Credential secret fields are decrypted back to plaintext so the
        rehydrated in-memory graph is usable by tools (lossless round-trip).
        """
        parsed = json.loads(data)
        _decrypt_credential_secrets(parsed)
        G = nx.node_link_graph(parsed)
        repo = cls(graph=G, provenance_validator=provenance_validator)
        repo._rebuild_indices()
        return repo

    def load_json(self, data: str) -> None:
        """Rehydrate THIS repo in place from a serialized graph.

        Replaces ``self.G`` with the deserialized graph and rebuilds the
        secondary indices. Credential secret fields are decrypted back to
        plaintext (reusing ``_decrypt_credential_secrets``) so the live graph
        stays usable by tools.

        Unlike ``from_json`` (which constructs a NEW repo), this mutates the
        existing object so callers holding a reference to this repo (the
        bundle's agents, which all share it) see the rehydrated data without
        rewiring. Used on engagement resume to restore the world-state
        snapshot into the freshly-built bundle's repo.
        """
        parsed = json.loads(data)
        _decrypt_credential_secrets(parsed)
        self.G = nx.node_link_graph(parsed)
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild ``_by_kind`` and ``_by_kind_field`` from the graph."""
        self._by_kind.clear()
        self._by_kind_field.clear()
        for nid, props in self.G.nodes(data=True):
            kind = props.get("kind", "")
            if kind:
                self._by_kind[kind].add(str(nid))
                self._update_field_indices(str(nid), kind, props)

    def _update_field_indices(
        self, node_id: str, kind: str, props: dict[str, Any]
    ) -> None:
        """Add entries to ``_by_kind_field`` for all indexed fields of a node."""
        for idx_kind, field in INDEXED_FIELDS:
            if idx_kind != kind:
                continue
            val = props.get(field)
            if val is not None:
                self._by_kind_field[(kind, field, _index_key(val))].add(node_id)

    def _remove_field_indices(
        self, node_id: str, kind: str, props: dict[str, Any]
    ) -> None:
        """Remove entries from ``_by_kind_field`` for all indexed fields."""
        for idx_kind, field in INDEXED_FIELDS:
            if idx_kind != kind:
                continue
            val = props.get(field)
            if val is not None:
                key = (kind, field, _index_key(val))
                self._by_kind_field[key].discard(node_id)

    # ------------------------------------------------------------------ #
    # Upsert                                                             #
    # ------------------------------------------------------------------ #

    async def upsert_node(self, node: WorldStateNode) -> None:
        if not node.provenance:
            raise ProvenanceRequiredError(
                f"{node.kind} {node.id} offered without provenance -- refused"
            )
        # Cross-check provenance against sibling store if validator wired.
        if self._provenance_validator is not None:
            ok = await self._provenance_validator(
                node.engagement_id, list(node.provenance)
            )
            if not ok:
                raise ProvenanceRequiredError(
                    f"{node.kind} {node.id} cites unknown tool_run_ids: {node.provenance}"
                )
        await self._upsert(node)

    async def upsert_nodes(self, nodes: Iterable[WorldStateNode]) -> None:
        for n in nodes:
            await self.upsert_node(n)

    async def _upsert(self, node: WorldStateNode) -> None:
        node_id = str(node.id)
        kind = node.kind
        props = _serialize_node(node)

        existing = self.G.nodes.get(node_id)
        if existing and existing.get("kind"):
            # Remove old field-index entries before updating.
            self._remove_field_indices(node_id, kind, dict(existing))
            # Additive / monotonic merge: a later low-detail re-parse that
            # lands on the same deterministic id must NOT wipe fields an
            # earlier, richer parse populated, nor demote confidence. See
            # ws-01. ``_merge_props`` mutates ``existing`` in place. The
            # node's schema defaults are passed so a field carrying only its
            # sentinel default (e.g. ``os_family="unknown"``) on the stub does
            # not clobber an enriched value. ``observed_fields`` is the set of
            # fields this parse *explicitly* set (pydantic ``model_fields_set``)
            # so a deliberately-observed scalar equal to its default sentinel
            # (e.g. ``tls=False`` after a prior ``tls=True``) is honoured rather
            # than masked as a no-information stub (wsN-01).
            _merge_props(
                existing,
                props,
                _schema_defaults(node),
                _observed_fields(node),
            )
        else:
            # New node
            self.G.add_node(node_id, **props)
            self._by_kind[kind].add(node_id)

        # Add new field-index entries.
        current = self.G.nodes[node_id]
        self._update_field_indices(node_id, kind, current)

    async def upsert_edge(self, edge: Edge) -> None:
        """Add or update a directed edge in the graph."""
        if not edge.provenance:
            raise ProvenanceRequiredError(
                f"Edge {edge.kind.value} {edge.src_id}->{edge.dst_id} offered without provenance"
            )
        src = str(edge.src_id)
        dst = str(edge.dst_id)
        edge_props = {
            "type": edge.kind.value,
            "engagement_id": str(edge.engagement_id),
            "provenance": [str(p) for p in edge.provenance],
            **{k: _scalar(v) for k, v in edge.properties.items()},
        }
        # networkx allows multiple calls to add_edge; it updates the props.
        self.G.add_edge(src, dst, key=edge.kind.value, **edge_props)

    # ------------------------------------------------------------------ #
    # Read                                                               #
    # ------------------------------------------------------------------ #

    async def get_nodes_by_kind(
        self, engagement_id: UUID, kind: str, *, include_secrets: bool = False
    ) -> list[dict[str, Any]]:
        """Return all nodes of ``kind`` for the engagement.

        Credential nodes are REDACTED by default (ws-04): secret fields
        (``password``/``nt_hash``/``aes256_key``/``ssh_private_key_ref``) are
        replaced with ``[ENCRYPTED]`` so a caller can never *accidentally*
        receive plaintext. The execution / persistence path that genuinely
        needs the live material passes ``include_secrets=True`` (see
        ``get_credentials_unredacted``). The serialization/encryption layer in
        ``to_json`` reads the live graph directly and is unaffected.
        """
        eid = str(engagement_id)
        redact = kind == "Credential" and not include_secrets
        out: list[dict[str, Any]] = []
        for nid in self._by_kind.get(kind, set()).copy():
            node = self.G.nodes[nid]
            if node.get("engagement_id") != eid:
                continue
            out.append(_redact_credential(dict(node)) if redact else dict(node))
        return out

    async def get_credentials_unredacted(
        self, engagement_id: UUID
    ) -> list[dict[str, Any]]:
        """Explicit unredacted Credential accessor for the execution path that
        needs the real secret material (lateral auth, persistence/encryption).

        This is the only sanctioned way to obtain plaintext credential secrets
        from a read; the generic :meth:`get_nodes_by_kind` redacts by default.
        """
        return await self.get_nodes_by_kind(
            engagement_id, "Credential", include_secrets=True
        )

    async def get_nodes_by_kind_field(
        self,
        engagement_id: UUID,
        kind: str,
        field: str,
        value: Any,
    ) -> list[dict[str, Any]]:
        """O(1) lookup using the secondary index."""
        eid = str(engagement_id)
        key = (kind, field, _index_key(value))
        return [
            dict(self.G.nodes[nid])
            for nid in self._by_kind_field.get(key, set()).copy()
            if self.G.nodes[nid].get("engagement_id") == eid
        ]

    async def get_services_for_host(
        self, engagement_id: UUID, host_id: UUID
    ) -> list[dict[str, Any]]:
        eid = str(engagement_id)
        hid = str(host_id)
        services = [
            dict(self.G.nodes[nid])
            for nid in self._by_kind_field.get(
                ("Service", "host_id", hid), set()
            ).copy()
            if self.G.nodes[nid].get("engagement_id") == eid
        ]
        services.sort(key=lambda s: s.get("port", 0))
        return services

    async def count(self, engagement_id: UUID) -> dict[str, int]:
        """Per-kind node count for the engagement."""
        eid = str(engagement_id)
        counts: dict[str, int] = {}
        for kind, nids in self._by_kind.items():
            n = sum(
                1
                for nid in nids
                if self.G.nodes[nid].get("engagement_id") == eid
            )
            if n > 0:
                counts[kind] = n
        return counts

    async def set_finding_override(
        self,
        *,
        engagement_id: UUID,
        finding_id: UUID,
        justification: str,
        by: str,
        at_iso: str,
    ) -> bool:
        """Mark a Finding as operator-overridden. Returns True iff the finding
        existed in the engagement and was updated."""
        fid = str(finding_id)
        eid = str(engagement_id)
        node = self.G.nodes.get(fid)
        if node is None:
            return False
        if node.get("engagement_id") != eid or node.get("kind") != "Finding":
            return False
        node["override_justification"] = justification
        node["override_by"] = by
        node["override_at"] = at_iso
        return True

    # ------------------------------------------------------------------ #
    # Edge queries                                                       #
    # ------------------------------------------------------------------ #

    async def get_successors_by_edge_type(
        self,
        node_id: UUID,
        edge_type: str,
    ) -> list[dict[str, Any]]:
        """Return nodes reachable from ``node_id`` via edges of the given type."""
        nid = str(node_id)
        if nid not in self.G:
            return []
        results = []
        for _, dst, data in self.G.edges(nid, data=True):
            if data.get("type") == edge_type:
                results.append(dict(self.G.nodes[dst]))
        return results

    async def get_predecessors_by_edge_type(
        self,
        node_id: UUID,
        edge_type: str,
    ) -> list[dict[str, Any]]:
        """Return nodes that have an edge of the given type pointing to ``node_id``."""
        nid = str(node_id)
        if nid not in self.G:
            return []
        results = []
        for src, _, data in self.G.in_edges(nid, data=True):
            if data.get("type") == edge_type:
                results.append(dict(self.G.nodes[src]))
        return results

    # ------------------------------------------------------------------ #
    # Cypher compatibility (removed)                                     #
    # ------------------------------------------------------------------ #

    async def run_read_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "run_read_cypher is no longer supported. Neo4j has been replaced "
            "with an in-process networkx graph. Use InMemoryGraphRepo methods "
            "(get_nodes_by_kind, get_nodes_by_kind_field, "
            "get_successors_by_edge_type) instead."
        )

    async def run_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return await self.run_read_cypher(cypher, params)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def delete_engagement(self, engagement_id: UUID) -> None:
        eid = str(engagement_id)
        to_remove = [
            nid
            for nid, props in self.G.nodes(data=True)
            if props.get("engagement_id") == eid
        ]
        for nid in to_remove:
            kind = self.G.nodes[nid].get("kind", "")
            self._remove_field_indices(nid, kind, dict(self.G.nodes[nid]))
            self._by_kind.get(kind, set()).discard(nid)
            self.G.remove_node(nid)

    async def close(self) -> None:
        """No connection to close -- in-process graph."""
        pass

    async def init_constraints(self) -> None:
        """No-op. Constraints are enforced by the Python layer."""
        pass


# Backwards-compatible alias so existing imports keep working.
Neo4jRepo = InMemoryGraphRepo


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _encrypt_credential_secrets(node_link: dict[str, Any]) -> None:
    """Encrypt Credential secret fields in a node-link-data dict, in place.

    ``node_link`` is a fresh ``nx.node_link_data`` output (a copy of the live
    graph state), so mutating it does not touch the in-memory graph.
    """
    from orchestrator import crypto

    for node in node_link.get("nodes", []):
        if node.get("kind") != "Credential":
            continue
        for field in CREDENTIAL_SECRET_FIELDS:
            value = node.get(field)
            if value is not None:
                node[field] = crypto.encrypt(value)


def _decrypt_credential_secrets(node_link: dict[str, Any]) -> None:
    """Decrypt Credential secret fields in a node-link-data dict, in place."""
    from orchestrator import crypto

    for node in node_link.get("nodes", []):
        if node.get("kind") != "Credential":
            continue
        for field in CREDENTIAL_SECRET_FIELDS:
            value = node.get(field)
            if value is None:
                continue
            decrypted = crypto.decrypt(value)
            # If it still looks like a Fernet token after a decrypt attempt,
            # the master key didn't match (rotated/restored mismatch). Don't
            # feed ciphertext to tools silently as if it were a secret --
            # surface it so the key mismatch is observable.
            if crypto.is_encrypted(decrypted):
                log.warning(
                    "world_state.decrypt_failed",
                    node_id=str(node.get("id")),
                    field=field,
                    reason="credential field still encrypted after decrypt (master key mismatch?)",
                )
            node[field] = decrypted


# Placeholder substituted for credential secret values at the read boundary.
# Mirrors ``queries.REDACTED_PLACEHOLDER`` (kept in sync; both render
# ``[ENCRYPTED]`` so LLM-facing slices never carry plaintext).
_REDACTED_PLACEHOLDER = "[ENCRYPTED]"


def _redact_credential(node: dict[str, Any]) -> dict[str, Any]:
    """Replace each populated Credential secret field with ``[ENCRYPTED]``.

    Operates on a node dict already detached from the live graph (caller
    passes a copy). Non-secret fields are untouched; ``None`` stays ``None``.
    """
    for field in CREDENTIAL_SECRET_FIELDS:
        if node.get(field) is not None:
            node[field] = _REDACTED_PLACEHOLDER
    return node


# Fields that must never be touched by a merge (identity / creation marker).
_MERGE_PROTECTED_FIELDS: frozenset[str] = frozenset({"id", "first_seen", "kind"})


def _is_empty(v: Any) -> bool:
    """True for values a low-detail re-parse would carry that must NOT
    clobber a populated field: ``None`` and empty list/dict/string."""
    if v is None:
        return True
    if isinstance(v, (list, dict, str)) and len(v) == 0:
        return True
    return False


def _merge_lists_union(existing: Any, incoming: Any) -> list[Any]:
    """Union two list-valued fields, preserving order (existing first) and
    de-duplicating. Used for list/set fields (hostnames, cpes, tags,
    provenance, affected_node_ids, ...). Non-list inputs are coerced."""
    out: list[Any] = []
    seen: set[Any] = set()
    for src in (existing, incoming):
        if src is None:
            continue
        items = src if isinstance(src, list) else [src]
        for item in items:
            try:
                key = item if not isinstance(item, (list, dict)) else json.dumps(item, sort_keys=True, default=str)
            except TypeError:
                key = repr(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out


def _schema_defaults(node: WorldStateNode) -> dict[str, Any]:
    """Serialized per-field schema defaults for ``node``'s model.

    Used by :func:`_merge_props` to recognise a field that carries only its
    *sentinel* default (e.g. ``os_family="unknown"``, ``tls=False``) on an
    incoming stub so it does not clobber an enriched value an earlier parse
    wrote. Required defaults (no default value) are omitted.
    """
    out: dict[str, Any] = {}
    for name, field in type(node).model_fields.items():
        if field.is_required():
            continue
        try:
            default = field.get_default(call_default_factory=True)
        except TypeError:
            default = field.get_default()
        out[name] = _scalar(default)
    return out


def _observed_fields(node: WorldStateNode) -> set[str]:
    """Names of fields the producing parse *explicitly* set on ``node``.

    Pydantic's ``model_fields_set`` records which fields were passed at
    construction (vs. filled from their default). ``_merge_props`` uses this
    to distinguish "field absent / not observed" (carries no information, must
    not clobber a populated value) from "observed value happens to equal the
    schema default" (e.g. a parse that genuinely saw ``Service.tls=False``):
    the latter is authoritative and is allowed to overwrite, so a prior
    ``True`` can be corrected back down to ``False`` (wsN-01).
    """
    return set(node.model_fields_set)


def _merge_props(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    defaults: dict[str, Any] | None = None,
    observed: set[str] | None = None,
) -> None:
    """Additive / monotonic in-place merge of an incoming serialized node onto
    an existing graph node (ws-01).

    Rules:
    * identity / creation fields (``id``/``first_seen``/``kind``) are never changed;
    * ``last_seen`` always advances to the incoming value (most-recent observation);
    * ``confidence`` is monotonic: never lowered (``max`` of the two);
    * list-valued fields are unioned (``provenance``, ``hostnames``, ``cpes``,
      ``tags``, ``cve``, ``affected_node_ids``, ``technologies``, ...);
    * a scalar field is overwritten when the incoming value is populated and is
      not merely the field's sentinel schema default — UNLESS the field was
      *explicitly observed* by the incoming parse (``observed``), in which case
      an authoritative value equal to its default (e.g. a parse that genuinely
      saw ``Service.tls=False`` after a prior ``True``) is honoured and applies
      (wsN-01). A stub's *unobserved* ``os_family="unknown"`` / ``tls=False``
      still cannot demote an enriched value;
    * a populated existing field is never wiped by a ``None``/empty re-parse.

    This makes a re-upsert of the same deterministic id strictly enriching for
    unobserved/default fields: a later low-detail stub can add provenance and
    advance ``last_seen`` but can never demote ``confidence`` or erase
    ``hostnames``/``os_family``/``product``/``version`` populated by an earlier
    richer parse. A genuine re-observation (the field is in ``observed``) is
    still applied, so legitimate boolean/scalar corrections propagate.
    """
    defaults = defaults or {}
    observed = observed or set()
    for key, new_val in incoming.items():
        if key in _MERGE_PROTECTED_FIELDS:
            continue
        if key == "last_seen":
            # Most-recent-wins: advance to the incoming observation time when present.
            if not _is_empty(new_val):
                existing[key] = new_val
            continue
        if key == "confidence":
            old_c = existing.get("confidence")
            try:
                if old_c is None:
                    existing["confidence"] = new_val
                elif new_val is not None:
                    existing["confidence"] = max(float(old_c), float(new_val))
            except (TypeError, ValueError):
                existing["confidence"] = new_val
            continue
        old_val = existing.get(key)
        # List/set fields: union so neither parse loses members.
        if isinstance(old_val, list) or isinstance(new_val, list):
            existing[key] = _merge_lists_union(old_val, new_val)
            continue
        if key not in existing:
            existing[key] = new_val
            continue
        # A field the incoming parse explicitly observed is authoritative, even
        # when its value equals the schema default (e.g. a real ``tls=False``
        # observation correcting a prior ``True``): the value is information,
        # not an absent-field stub, so it overwrites. An *empty* observed value
        # (None / "") is still treated as no-information and skipped so a
        # cleared field doesn't wipe a populated one.
        if key in observed and not _is_empty(new_val):
            existing[key] = new_val
            continue
        # An UNOBSERVED incoming value equal to its schema-default sentinel
        # carries no new information; never let it overwrite a populated
        # existing value. (An unobserved field is a default the producer never
        # set — a stub.)
        is_sentinel = key in defaults and new_val == defaults[key]
        if _is_empty(new_val) or is_sentinel:
            continue
        existing[key] = new_val


def _serialize_node(node: WorldStateNode) -> dict[str, Any]:
    raw = node.model_dump(mode="json")
    flat: dict[str, Any] = {}
    for k, v in raw.items():
        flat[k] = _scalar(v)
    # Store kind explicitly for index rebuilds.
    flat["kind"] = node.kind
    return flat


def _scalar(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, list):
        return [_scalar(x) for x in v]
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    if isinstance(v, UUID):
        return str(v)
    return v


def _index_key(v: Any) -> Any:
    """Normalize a value for use as a dict key in the secondary index."""
    if isinstance(v, UUID):
        return str(v)
    return v
