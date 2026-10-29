"""Chain linked tracklets into trajectories.

Links produced by the candidate stage are one-to-one (each tracklet has at most
one successor and at most one predecessor), so the link graph is a disjoint
union of simple paths. Each path, singletons included, becomes one trajectory.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from gap_bridging.types import Endpoint


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #
def trajectory_id(video_id: str, n: int) -> str:
    """Return the identifier of the ``n``-th trajectory (1-based) of a video.

    Parameters
    ----------
    video_id : str
        Video the trajectory belongs to.
    n : int
        1-based rank of the trajectory within the video.

    Returns
    -------
    str
        ``f"{video_id}/trajectory{n:07d}"``.
    """
    if n < 1:
        raise ValueError(f"trajectory rank must be >= 1, got {n}")
    return f"{video_id}/trajectory{n:07d}"


def bridge_ids(trajectory_id: str, members: Sequence[str]) -> list[str]:
    """Return the bridge identifiers of the gaps of a trajectory.

    Parameters
    ----------
    trajectory_id : str
        Identifier of the trajectory.
    members : sequence of str
        Ordered member tracklet ids; ``len(members) - 1`` gaps.

    Returns
    -------
    list of str
        ``f"{trajectory_id}/b{k:03d}"`` for the k-th gap, k = 1 .. len(members) - 1.
    """
    return [f"{trajectory_id}/b{k:03d}" for k in range(1, len(members))]


# --------------------------------------------------------------------------- #
# Chaining
# --------------------------------------------------------------------------- #
def _validate_links(ids: set[str], links: Sequence[tuple[str, str]],
                    endpoints: Mapping[str, tuple[Endpoint, Endpoint]]) -> tuple[dict[str, str], set[str]]:
    """Check links are one-to-one, known, same-video and forward in time."""
    successor: dict[str, str] = {}
    predecessors: set[str] = set()
    for a, b in links:
        if a not in ids or b not in ids:
            raise ValueError(f"link ({a!r}, {b!r}) references an unknown tracklet")
        if a == b:
            raise ValueError(f"link ({a!r}, {b!r}) is a self-loop (cycle)")
        if a in successor:
            raise ValueError(f"tracklet {a!r} has two successors: {successor[a]!r}, {b!r}")
        if b in predecessors:
            raise ValueError(f"tracklet {b!r} has two predecessors")
        end_a, start_b = endpoints[a][1], endpoints[b][0]
        if end_a.video_id != start_b.video_id:
            raise ValueError(f"link ({a!r}, {b!r}) crosses videos")
        if not end_a.frame < start_b.frame:
            raise ValueError(f"link ({a!r}, {b!r}) does not go forward in time "
                             f"(end {end_a.frame}, start {start_b.frame})")
        successor[a] = b
        predecessors.add(b)
    return successor, predecessors


def build_trajectories(tracklet_ids: list[str], links: list[tuple[str, str]],
                       endpoints: dict[str, tuple[Endpoint, Endpoint]]) -> dict[str, list[str]]:
    """Group tracklets into trajectories along one-to-one links.

    Parameters
    ----------
    tracklet_ids : list of str
        Every tracklet to assign; each appears in exactly one trajectory.
    links : list of (str, str)
        One-to-one links ``(a_id, b_id)``: tracklet ``b`` continues ``a``.
    endpoints : dict
        ``tracklet_id -> (start, end)`` endpoints, used for video and frame.

    Returns
    -------
    dict of str to list of str
        ``trajectory_id -> ordered member tracklet ids`` in trajectory order:
        sorted by (video_id, start_frame, first member tracklet_id) and
        numbered from 1 within each video.

    Raises
    ------
    ValueError
        On duplicate or unknown tracklet ids, a tracklet with two successors or
        two predecessors, a cycle, a cross-video link or a link that does not
        go forward in time.
    """
    ids = set(tracklet_ids)
    if len(ids) != len(tracklet_ids):
        raise ValueError("tracklet_ids contains duplicates")
    missing = ids.difference(endpoints)
    if missing:
        raise ValueError(f"no endpoints for tracklets: {sorted(missing)[:5]}")
    successor, predecessors = _validate_links(ids, links, endpoints)

    paths: list[list[str]] = []
    visited: set[str] = set()
    for head in sorted(ids.difference(predecessors)):
        path = [head]
        while path[-1] in successor:
            path.append(successor[path[-1]])
        visited.update(path)
        paths.append(path)
    if len(visited) != len(ids):
        raise ValueError(f"links contain a cycle through {sorted(ids - visited)[:5]}")

    def key(path: list[str]) -> tuple[str, int, str]:
        start = endpoints[path[0]][0]
        return (start.video_id, int(start.frame), path[0])

    out: dict[str, list[str]] = {}
    rank: dict[str, int] = {}
    for path in sorted(paths, key=key):
        video = endpoints[path[0]][0].video_id
        rank[video] = rank.get(video, 0) + 1
        out[trajectory_id(video, rank[video])] = path
    return out


def tracklet_to_trajectory(trajectories: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Invert :func:`build_trajectories`: ``tracklet_id -> trajectory_id``."""
    inverse: dict[str, str] = {}
    for traj, members in trajectories.items():
        for member in members:
            if member in inverse:
                raise ValueError(f"tracklet {member!r} is in two trajectories")
            inverse[member] = traj
    return inverse
