"""
Radix Tree KV Cache — SGLang-style prefix reuse for TinyGrad on Jetson Orin AGX.

Multi-agent scenario
--------------------
Multiple agents share a common prefix (system prompt, tool definitions, long
context doc). Without prefix sharing, each agent pays the full O(n²) prefill
cost for every re-use of those tokens.

With this cache:
  - First agent: pays full prefill, result snapshotted into the tree.
  - Every subsequent agent matching the same prefix: skips those tokens,
    restores cached KV + recurrent state, prefills only the private suffix.

Architecture (mirrors SGLang's RadixCache)
------------------------------------------
  RadixNode         — tree node; edge label = token_ids, value = ModelSnapshot
  RadixPrefixCache  — manages the tree: match_prefix / insert / LRU eviction
  ModelSnapshot     — frozen model state at a prefix boundary:
                        • attention KV cache slices  (one per FullAttentionBlock)
                        • DeltaNet recurrent states  (one per DeltaNetBlock)

Key differences from SGLang's implementation
---------------------------------------------
  • No physical KV pool / slot-index indirection: snapshots store Tensors
    directly.  The Jetson's 64 GB unified memory makes this affordable for the
    small number of distinct prefixes typical in multi-agent workloads.
  • Includes DeltaNet (linear attention) recurrent state — not just GQA KV —
    so the full hybrid model state is faithfully restored at the split point.
  • Page-size = 1 (token-granularity).  SGLang uses paged allocations for GPU
    memory management; here tinygrad handles allocation directly.
  • LRU eviction on token-count budget (default 32 k tokens of shared prefix).
  • Thread-safe via a per-cache lock for concurrent agent access.

Usage
-----
    from radix_kv_cache import RadixPrefixCache

    cache = RadixPrefixCache(max_token_budget=32_768)

    # Agent loop — each agent gets a different suffix after shared_prefix
    shared_prefix = tok.encode(SYSTEM_PROMPT)
    for agent_suffix in agent_suffixes:
        tokens = shared_prefix + tok.encode(agent_suffix)
        out_ids = list(model.generate_with_prefix_cache(tokens[:], cache))
        print(tok.decode(out_ids))

Eviction policy
---------------
  LRU leaf eviction: only unpinned leaf nodes (ref_count == 0) are eligible.
  When a leaf is evicted its parent may become a new leaf and join the pool.
  Pinning (inc_ref / dec_ref) is lock-based; the caller is responsible for
  releasing pins once the agent is done with a particular node.
"""
from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tinygrad import Tensor


# ── Model snapshot ────────────────────────────────────────────────────────────

@dataclass
class ModelSnapshot:
    """Frozen model state at a prefix boundary.

    Stores all state needed to resume exact inference from ``prefix_len``
    without re-computing any earlier tokens:

    * ``attn_kv``      — per FullAttentionBlock index → Tensor
                         shape (2, B, n_kv_heads, prefix_len, head_dim)
    * ``delta_states`` — per DeltaNetBlock: (delta_state, conv_state)
                         shapes match those returned by DeltaNetBlock.init_state()

    Memory estimate (fp16, Qwen3-Coder-Next defaults):
        12 attn blocks × 2 × 1 × n_kv_heads × 2048 × head_dim × 2 B ≈ 96 MB
        per 2 k-token prefix (dominated by attn KV; delta states are ~few MB).
    """
    prefix_len: int
    attn_kv: Dict[int, Tensor]                        # layer_idx → KV slice
    delta_states: List[Tuple[Tensor, Tensor]]          # [(ds, cs), …]


# ── Radix tree node ───────────────────────────────────────────────────────────

class RadixNode:
    """One node in the radix prefix tree.

    The edge from ``parent`` to this node is labelled by ``token_ids``.
    ``snapshot`` is stored *only* for nodes where the full edge has been
    computed (i.e. nodes inserted via ``insert()``), not for intermediate
    split-nodes that have no associated KV data.

    ``ref_count`` counts active consumers (agents) that are currently using
    this node's snapshot.  A node with ref_count > 0 is pinned and will not
    be evicted even if it is a leaf.
    """
    __slots__ = (
        "children", "parent", "token_ids", "snapshot",
        "ref_count", "last_access", "depth_tokens",
    )

    def __init__(self) -> None:
        self.children: Dict[int, RadixNode] = {}
        self.parent:   Optional[RadixNode]  = None
        self.token_ids: List[int]           = []       # edge label
        self.snapshot:  Optional[ModelSnapshot] = None
        self.ref_count: int   = 0
        self.last_access: float = time.monotonic()
        self.depth_tokens: int  = 0  # total tokens root → end of this edge

    # heapq needs a total order; break ties by id() to avoid comparing Tensors
    def __lt__(self, other: "RadixNode") -> bool:
        if self.last_access != other.last_access:
            return self.last_access < other.last_access
        return id(self) < id(other)


# ── Radix prefix cache ────────────────────────────────────────────────────────

class RadixPrefixCache:
    """Radix tree for KV-cache prefix reuse.

    Public API
    ----------
    match_prefix(tokens)          → (matched_len, snapshot | None)
    insert(tokens, snapshot)      → None
    inc_ref(node) / dec_ref(node) → pin / unpin a node against eviction
    total_cached_tokens()         → int

    Thread safety
    -------------
    All public methods acquire ``self._lock`` before mutating tree state.
    Snapshot Tensors themselves are immutable after insertion.
    """

    def __init__(self, max_token_budget: int = 32_768) -> None:
        self.max_token_budget = max_token_budget
        self._lock = threading.Lock()

        self.root = RadixNode()
        self.root.token_ids    = []
        self.root.depth_tokens = 0
        self.root.ref_count    = 1   # root is always pinned

        self._total_cached_tokens: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def match_prefix(
        self, tokens: List[int]
    ) -> Tuple[int, Optional[ModelSnapshot]]:
        """Find the longest cached prefix of ``tokens``.

        Returns ``(matched_len, snapshot)`` where ``matched_len`` is the number
        of leading tokens covered by the tree and ``snapshot`` is the
        ModelSnapshot at that depth (or None if matched_len == 0).

        A *partial* edge match (the query ends inside a node's token_ids before
        we reach a snapshotted boundary) returns the last fully-matched
        ancestor, not the mid-edge position.  This differs slightly from
        SGLang which splits nodes lazily; here we keep the tree lean and avoid
        structural mutations on read paths.
        """
        with self._lock:
            return self._match_prefix_locked(tokens)

    def insert(self, tokens: List[int], snapshot: ModelSnapshot) -> None:
        """Insert ``tokens[:snapshot.prefix_len]`` with its model snapshot.

        Handles:
          • New leaf:   tokens diverge from all existing paths.
          • Full match: path already exists; update snapshot if absent.
          • Mid-edge split: tokens share part of an edge; split that node and
            add a new branch.
        """
        with self._lock:
            self._insert_locked(tokens[: snapshot.prefix_len], snapshot)
            self._maybe_evict_locked()

    def inc_ref(self, node: RadixNode) -> None:
        """Pin a node (and all its ancestors) against LRU eviction."""
        with self._lock:
            n = node
            while n is not None and n is not self.root:
                n.ref_count += 1
                n = n.parent

    def dec_ref(self, node: RadixNode) -> None:
        """Unpin a node (and all its ancestors) from LRU eviction."""
        with self._lock:
            n = node
            while n is not None and n is not self.root:
                n.ref_count = max(0, n.ref_count - 1)
                n = n.parent

    def total_cached_tokens(self) -> int:
        with self._lock:
            return self._total_cached_tokens

    def pretty_print(self) -> None:
        """Debug: print full tree structure to stdout."""
        with self._lock:
            self._print_node(self.root, 0, "ROOT")
        print(f"Total cached tokens: {self._total_cached_tokens}")

    # ── Internal (caller holds _lock) ─────────────────────────────────────────

    def _match_prefix_locked(
        self, tokens: List[int]
    ) -> Tuple[int, Optional[ModelSnapshot]]:
        node = self.root
        pos  = 0
        best_matched  = 0
        best_snapshot: Optional[ModelSnapshot] = None
        now = time.monotonic()

        while pos < len(tokens):
            first_tok = tokens[pos]
            if first_tok not in node.children:
                break

            child = node.children[first_tok]
            n = _common_prefix_len(child.token_ids, tokens[pos:])

            if n == 0:
                break

            child.last_access = now

            if n < len(child.token_ids):
                # Partial edge match — record depth but don't descend further.
                # No snapshot exists at mid-edge positions.
                break

            # Full edge match — descend
            pos += n
            if child.snapshot is not None:
                best_matched  = pos
                best_snapshot = child.snapshot
            node = child

        return best_matched, best_snapshot

    def _insert_locked(self, tokens: List[int], snapshot: ModelSnapshot) -> None:
        node = self.root
        pos  = 0

        while pos < len(tokens):
            first_tok = tokens[pos]
            remaining = tokens[pos:]

            if first_tok not in node.children:
                # ── New leaf for the entire remaining suffix ──────────────────
                leaf = RadixNode()
                leaf.parent       = node
                leaf.token_ids    = list(remaining)
                leaf.snapshot     = snapshot
                leaf.depth_tokens = pos + len(remaining)
                leaf.last_access  = time.monotonic()
                node.children[first_tok] = leaf
                self._total_cached_tokens += len(remaining)
                return

            child = node.children[first_tok]
            n = _common_prefix_len(child.token_ids, remaining)

            if n == len(child.token_ids):
                # ── Full match — descend ──────────────────────────────────────
                child.last_access = time.monotonic()
                if child.snapshot is None:
                    child.snapshot = snapshot
                pos  += n
                node  = child
                continue

            # ── Partial match — split child at position n ─────────────────────
            #
            # Before:  parent → [A B C D E]  (child, has snapshot)
            # After:   parent → [A B] → [C D E]  (split_node, child)
            #                         → [X Y Z]  (new_leaf, for diverging suffix)
            #
            split_node              = RadixNode()
            split_node.parent       = node
            split_node.token_ids    = child.token_ids[:n]
            split_node.snapshot     = None      # no snapshot at split boundary
            split_node.depth_tokens = pos + n
            split_node.last_access  = time.monotonic()

            # Rewire child to be under split_node
            child.token_ids  = child.token_ids[n:]
            child.parent     = split_node
            split_node.children[child.token_ids[0]] = child

            # Replace old child in parent
            node.children[first_tok] = split_node

            diverge = remaining[n:]
            if diverge:
                # Add new leaf for the diverging suffix
                new_leaf              = RadixNode()
                new_leaf.parent       = split_node
                new_leaf.token_ids    = list(diverge)
                new_leaf.snapshot     = snapshot
                new_leaf.depth_tokens = pos + n + len(diverge)
                new_leaf.last_access  = time.monotonic()
                split_node.children[diverge[0]] = new_leaf
                self._total_cached_tokens += len(diverge)
            else:
                # The inserted prefix ends exactly at the split point
                split_node.snapshot = snapshot
            return

        # tokens fully consumed — path already exists, update snapshot if missing
        if node is not self.root and node.snapshot is None:
            node.snapshot = snapshot

    def _maybe_evict_locked(self) -> None:
        if self._total_cached_tokens <= self.max_token_budget:
            return
        to_free = self._total_cached_tokens - self.max_token_budget
        self._evict_lru_locked(to_free)

    def _evict_lru_locked(self, target_tokens: int) -> None:
        """Evict unpinned leaf nodes in LRU order."""
        # Collect all evictable leaves into a min-heap (oldest first)
        evictable: List[RadixNode] = []
        self._collect_evictable_leaves(self.root, evictable)
        heapq.heapify(evictable)

        freed = 0
        while freed < target_tokens and evictable:
            node = heapq.heappop(evictable)
            if node.ref_count > 0:
                continue
            freed += len(node.token_ids)
            self._delete_leaf_locked(node)
            # Parent may now be an evictable leaf
            parent = node.parent
            if (
                parent is not None
                and parent is not self.root
                and not parent.children
                and parent.ref_count == 0
            ):
                heapq.heappush(evictable, parent)

    def _collect_evictable_leaves(
        self, node: RadixNode, out: List[RadixNode]
    ) -> None:
        if not node.children and node is not self.root and node.ref_count == 0:
            out.append(node)
            return
        for child in node.children.values():
            self._collect_evictable_leaves(child, out)

    def _delete_leaf_locked(self, node: RadixNode) -> None:
        if node.parent is None:
            return
        key = node.token_ids[0]
        node.parent.children.pop(key, None)
        self._total_cached_tokens -= len(node.token_ids)
        # Drop Tensor references so tinygrad / GC can reclaim the buffer
        node.snapshot = None

    def _print_node(self, node: RadixNode, depth: int, label: str) -> None:
        indent  = "  " * depth
        snap    = "✓" if node.snapshot is not None else "✗"
        preview = node.token_ids[:8]
        ellip   = "…" if len(node.token_ids) > 8 else ""
        print(
            f"{indent}{label}: toks={preview}{ellip} "
            f"depth={node.depth_tokens} snap={snap} ref={node.ref_count}"
        )
        for child in node.children.values():
            self._print_node(child, depth + 1, f"[{child.token_ids[0]}]")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _common_prefix_len(a: List[int], b: List[int]) -> int:
    """Length of the longest common prefix between lists a and b."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
