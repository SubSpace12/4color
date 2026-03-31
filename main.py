#!/usr/bin/env python3
"""
Wandering Fifth Color: Four-Coloring Planar Graphs
Archuleta & Shapiro, ACM 1986.

Variant: Rose Petal ordering with Saturation Level = 2
"""

import random
import time
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

def four_color(graph: dict, max_restarts: int = 200, seed: int = None) -> tuple[dict | None, int]:
    """
    Four-color a planar graph using the Wandering Fifth Color algorithm.

    Parameters
    ----------
    graph        : {vertex: iterable of neighbors}
    max_restarts : maximum number of full restarts before giving up
    seed         : optional random seed for reproducibility

    Returns
    -------
    (coloring, restarts_used)
    coloring is {vertex: color in {1,2,3,4}} or None if failed
    """
    if seed is not None:
        random.seed(seed)

    vertices = list(graph.keys())
    n = len(vertices)

    if n == 0:
        return {}, 0

    # Normalise adjacency to frozensets once
    adj = {v: frozenset(graph[v]) for v in vertices}

    for attempt in range(max_restarts):
        # Rose-petal center: cycle through all vertices first, then random.
        # This guarantees a different structural ordering on each restart.
        center = vertices[attempt % n] if attempt < n else random.choice(vertices)

        result = _attempt(adj, vertices, center)
        if result is not None:
            return result, attempt   # attempt == number of restarts before success

    return None, max_restarts


def _bfs_order(adj: dict, vertices: list, center) -> list:
    """BFS from center → vertex visit order (rose petal layers)."""
    order, visited = [], {center}
    queue = deque([center])
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in adj[v]:
            if u not in visited:
                visited.add(u)
                queue.append(u)
    for v in vertices:          # handle disconnected components
        if v not in visited:
            order.append(v)
    return order


def _attempt(adj: dict, vertices: list, center) -> dict | None:
    """
    Single coloring attempt.

    Vertex scheduling: Rose Petal (BFS from center) with Saturation=2 override.
      • Any uncoloured vertex with ≥ 2 distinct colours in its neighbourhood
        is pulled to the front (most constrained first).
      • All others follow BFS order.

    Color assignment: skewed (always pick lowest available colour class).

    Impasse resolution: Wandering Fifth Color (recursive local search).
    """
    color: dict = {}           # vertex → int ∈ {1,2,3,4}
    uncolored: set = set(vertices)

    bfs_pos = {v: i for i, v in enumerate(_bfs_order(adj, vertices, center))}

    # ── helpers ──────────────────────────────────────────────────────────────

    def nbr_colors(v) -> set:
        """Distinct colours currently used by coloured neighbours of v."""
        return {color[u] for u in adj[v] if u in color}

    def first_free(v) -> int | None:
        """
        Lowest colour not blocked by any neighbour (skewed / greedy assignment).
        Returns None if all four colours are blocked (impasse).
        """
        used = nbr_colors(v)
        for c in (1, 2, 3, 4):
            if c not in used:
                return c
        return None

    def next_vertex():
        """
        Rose Petal + Saturation = 2 scheduling rule.

        Among uncoloured vertices whose neighbourhood already contains
        ≥ 2 distinct colours, select the most-saturated one
        (tie-break: smallest BFS index = closest to the petal centre).
        If no such vertex exists, take the next vertex in BFS order.
        """
        best_hi = best_lo = None
        best_hi_sat = -1

        for v in uncolored:
            s = len(nbr_colors(v))
            if s >= 2:
                if best_hi is None or s > best_hi_sat or (
                        s == best_hi_sat and bfs_pos[v] < bfs_pos[best_hi]):
                    best_hi, best_hi_sat = v, s
            else:
                if best_lo is None or bfs_pos[v] < bfs_pos[best_lo]:
                    best_lo = v

        return best_hi if best_hi is not None else best_lo

    # ── wandering fifth colour ────────────────────────────────────────────────

    def wander(start) -> bool:
        """
        Attempt to resolve the impasse at `start` without adding a 5th colour.

        The key insight: if a neighbour u has a colour c that appears only
        ONCE in v's neighbourhood, we can "move" c from u to v, freeing u
        to find a new colour elsewhere.  If u can't re-colour directly,
        we recurse — the impasse "wanders" to u, then to u's neighbour, etc.

        Termination conditions (backtrack):
          • Vertex of high degree where every colour appears ≥ 2 times
            (no candidate neighbour to displace)
          • Path crossing detection (don't revisit a vertex on the wander path)
          • All recursive branches exhausted

        Returns True if `start` ends up coloured, False if we must restart.
        """
        path: set = set()

        def recurse(v) -> bool:
            path.add(v)

            # Count colour occurrences in v's current neighbourhood
            count: dict[int, int] = {}
            for u in adj[v]:
                if u in color:
                    c = color[u]
                    count[c] = count.get(c, 0) + 1

            # Candidates: neighbours whose colour is UNIQUE in v's neighbourhood.
            # Only these can be displaced: giving v their colour creates no
            # conflict (no other neighbour has that colour).
            candidates = [u for u in adj[v]
                          if u in color and count[color[u]] == 1]

            # ── Single pass: for each candidate, try direct recolor then
            # immediately recurse if that fails, before moving to next candidate.
            for u in candidates:
                if u in path:               # don't cross our own trail
                    continue

                c_u = color[u]

                # Give v colour c_u; free u
                color[v] = c_u
                uncolored.discard(v)
                del color[u]
                uncolored.add(u)

                new_c = first_free(u)
                if new_c is not None:       # direct recolor succeeded → done
                    color[u] = new_c
                    uncolored.discard(u)
                    path.discard(v)
                    return True

                # Direct failed — immediately recurse on u
                if recurse(u):
                    path.discard(v)
                    return True

                # Both failed — undo and try next candidate
                del color[v]
                uncolored.add(v)
                color[u] = c_u
                uncolored.discard(u)

            path.discard(v)
            return False                    # backtrack

        return recurse(start)

    # ── main loop ─────────────────────────────────────────────────────────────

    while uncolored:
        v = next_vertex()
        c = first_free(v)

        if c is not None:
            color[v] = c
            uncolored.discard(v)
        else:
            # Impasse: all four colours are blocked by neighbours.
            # Try to resolve by wandering.
            if not wander(v):
                return None     # wander failed → caller will restart with new center

    return color


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify(graph: dict, coloring: dict) -> bool:
    """Return True iff coloring is a valid proper 4-colouring of graph."""
    if coloring is None:
        return False
    for v in graph:
        if v not in coloring:
            return False
        if coloring[v] not in (1, 2, 3, 4):
            return False
        for u in graph[v]:
            if coloring[u] == coloring[v]:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Graph generators
# ─────────────────────────────────────────────────────────────────────────────

def grid(rows: int, cols: int) -> dict:
    """Plain grid graph — 2-colourable (bipartite), easy warm-up."""
    g = {(r, c): set() for r in range(rows) for c in range(cols)}
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                g[(r, c)].add((r+1, c)); g[(r+1, c)].add((r, c))
            if c + 1 < cols:
                g[(r, c)].add((r, c+1)); g[(r, c+1)].add((r, c))
    return g


def triangulated_grid(rows: int, cols: int) -> dict:
    """
    Grid with one diagonal per cell added.
    Higher average degree → more impasses, better stress test.
    """
    g = grid(rows, cols)
    for r in range(rows - 1):
        for c in range(cols - 1):
            g[(r, c)].add((r+1, c+1)); g[(r+1, c+1)].add((r, c))
    return g


def dodecahedron() -> dict:
    """
    The dodecahedron: 20 vertices, 30 edges, all faces are pentagons.
    A classic planar graph that requires all 4 colours.
    """
    edges = [
        (0,1),(0,4),(0,5),(1,2),(1,6),(2,3),(2,7),(3,4),(3,8),(4,9),
        (5,10),(5,14),(6,10),(6,11),(7,11),(7,12),(8,12),(8,13),(9,13),(9,14),
        (10,15),(11,16),(12,17),(13,18),(14,19),
        (15,16),(15,19),(16,17),(17,18),(18,19),
    ]
    g = {i: set() for i in range(20)}
    for u, v in edges:
        g[u].add(v); g[v].add(u)
    return g


def petersen() -> dict:
    """
    Petersen graph — non-planar but 3-colourable; used as a sanity check.
    The algorithm should still find a valid colouring (just not necessarily 4).
    """
    g = {i: set() for i in range(10)}
    outer = [(0,1),(1,2),(2,3),(3,4),(4,0)]
    inner = [(5,7),(7,9),(9,6),(6,8),(8,5)]
    spokes = [(0,5),(1,6),(2,7),(3,8),(4,9)]
    for u, v in outer + inner + spokes:
        g[u].add(v); g[v].add(u)
    return g


def random_planar_triangulation(n: int, seed: int = None) -> dict:
    """
    Random maximal planar graph (triangulation) via incremental face splitting.

    Algorithm:
      1. Start with triangle {0,1,2}.
      2. For each new vertex v, pick a random existing triangular face {a,b,c},
         remove it, and replace it with three new triangles {v,a,b}, {v,b,c}, {v,a,c}.
      3. Add edges v-a, v-b, v-c.

    Planarity is guaranteed by construction: we only ever split existing faces,
    never add a crossing edge.  The result is a triangulation with 3n-6 edges
    (maximum for a planar graph on n vertices).
    """
    if seed is not None:
        random.seed(seed)

    if n <= 0:
        return {}
    if n == 1:
        return {0: set()}
    if n == 2:
        return {0: {1}, 1: {0}}

    g = {i: set() for i in range(n)}

    # Seed triangle
    for u, v in ((0,1),(1,2),(0,2)):
        g[u].add(v); g[v].add(u)

    if n == 3:
        return g

    # Face list: each face is a list [a, b, c]
    faces = [[0, 1, 2]]

    for v in range(3, n):
        # Pick and remove a random face (swap-with-last for O(1) removal)
        idx = random.randrange(len(faces))
        a, b, c = faces[idx]
        faces[idx] = faces[-1]
        faces.pop()

        # Split into three new faces
        faces.append([v, a, b])
        faces.append([v, b, c])
        faces.append([v, a, c])

        # Connect v to all three corners
        for u in (a, b, c):
            g[v].add(u); g[u].add(v)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def run(name: str, graph: dict, max_restarts: int = 200):
    n = len(graph)
    e = sum(len(v) for v in graph.values()) // 2

    t0 = time.perf_counter()
    col, restarts = four_color(graph, max_restarts=max_restarts, seed=0)
    dt = (time.perf_counter() - t0) * 1000

    ok = verify(graph, col)
    colors_used = sorted(set(col.values())) if col else []
    status = "✓" if ok else ("✗ INVALID" if col else "FAILED")

    print(f"  {name:<42} n={n:>5}  e={e:>6}  "
          f"colours={colors_used}  restarts={restarts:>3}  "
          f"{dt:7.1f} ms  {status}")


if __name__ == "__main__":
    print("Wandering Fifth Color  ·  Rose Petal ordering, Saturation = 2")
    print("=" * 80)

    print("\n── Structured graphs ──────────────────────────────────────────────────────")
    run("5×5 grid",                     grid(5, 5))
    run("20×20 grid",                   grid(20, 20))
    run("50×50 grid",                   grid(50, 50))
    run("20×20 triangulated grid",      triangulated_grid(20, 20))
    run("50×50 triangulated grid",      triangulated_grid(50, 50))
    run("Dodecahedron (needs 4 cols)",  dodecahedron())
    run("Petersen (non-planar, 3-col)", petersen())

    print("\n── Random planar triangulations ───────────────────────────────────────────")
    for n in (100, 500, 1000, 2500, 5000):
        run(f"Random triangulation n={n}",
            random_planar_triangulation(n, seed=42))

    print("\n── Stress test: 10 independent random triangulations, n=1000 ─────────────")
    failures = 0
    for trial in range(10):
        g = random_planar_triangulation(1000, seed=trial * 7)
        col, restarts = four_color(g, max_restarts=200, seed=trial)
        ok = verify(g, col)
        marker = "✓" if ok else "✗"
        print(f"  trial {trial+1:>2}: restarts={restarts:>3}  {marker}")
        if not ok:
            failures += 1
    print(f"  Failures: {failures}/10")