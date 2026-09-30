"""Multiscale scatter-component tracking with causal bulk-motion readout.

Fine-scale connected density cores are kept separate. Previously unassigned points
may attach to a core at a larger scale, without chaining through peripheral points.
All original point rows survive; membership and bulk-state weights are separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmwcore import _native


@dataclass(frozen=True)
class ClusterConfig:
    fine_radius_m: float = 0.35
    outer_radius_m: float = 0.8
    min_points: int = 3
    velocity_scale_mps: float = 0.2710796965422454  # three recorded Doppler bins
    core_power_ratio: float = 4.0


DEFAULT_CONFIG = ClusterConfig()


def density_labels(xy: np.ndarray, radius: float, minimum: int) -> np.ndarray:
    """Deterministic DBSCAN; ambiguous borders follow first core component."""
    distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    neighbours = distances <= radius
    core = neighbours.sum(axis=1) >= minimum
    labels = np.full(len(xy), -1, dtype=int)
    label = 0
    for seed in np.flatnonzero(core):
        if labels[seed] >= 0:
            continue
        queue = [seed]
        labels[seed] = label
        while queue:
            current = queue.pop()
            for other in np.flatnonzero(neighbours[current] & (labels < 0)):
                labels[other] = label
                if core[other]:
                    queue.append(other)
        label += 1
    return labels


def cluster_points(
    points: np.ndarray, method: str, config: ClusterConfig = DEFAULT_CONFIG
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Input level-world x/y/z/vr/snr; output labels, bulk weights, observations."""
    points = np.asarray(points, dtype=float).reshape(-1, 5)
    if not np.isfinite(points).all():
        raise ValueError("Points must be finite")
    if method not in {"small", "large", "two_scale", "weighted", "power_split"}:
        raise ValueError("Unknown clustering method")
    xy = points[:, :2]
    radius = config.outer_radius_m if method == "large" else config.fine_radius_m
    labels = density_labels(xy, radius, config.min_points)
    if method == "power_split":
        labels = split_supported_cores(points, labels, config)
    initial = labels.copy()
    if method in {"two_scale", "weighted", "power_split"} and np.any(labels >= 0):
        # Attach only to original members: no outer-point chains and no core merge.
        members = np.flatnonzero(initial >= 0)
        for index in np.flatnonzero(initial < 0):
            distances = np.linalg.norm(xy[members] - xy[index], axis=1)
            closest = int(np.argmin(distances))
            if distances[closest] <= config.outer_radius_m:
                labels[index] = initial[members[closest]]
    weights = np.zeros(len(points))
    observations = []
    for label in sorted(set(labels) - {-1}):
        indices = np.flatnonzero(labels == label)
        subset = points[indices]
        local = np.linalg.norm(subset[:, None, :2] - subset[None, :, :2], axis=-1)
        density = (local <= config.fine_radius_m).sum(axis=1).astype(float)
        local_weights = np.ones(len(indices))
        if method == "weighted":
            core = initial[indices] == label
            bulk_velocity = float(np.median(subset[core, 3]))
            consistency = 1 / (
                1 + ((subset[:, 3] - bulk_velocity) / config.velocity_scale_mps) ** 2
            )
            local_weights = 0.05 + density / density.max() * consistency
        local_weights /= local_weights.sum()
        weights[indices] = local_weights
        center = np.sum(subset[:, :2] * local_weights[:, None], axis=0)
        observations.append(
            dict(
                label=int(label),
                members=indices.tolist(),
                center=center.tolist(),
                radial_velocity=float(local_weights @ subset[:, 3]),
                snr_sum=float(np.power(10.0, subset[:, 4] / 10).sum()),
                rms_extent_m=float(
                    np.sqrt(local_weights @ np.sum((subset[:, :2] - center) ** 2, axis=1))
                ),
            )
        )
    return labels, weights, observations


def split_supported_cores(
    points: np.ndarray, labels: np.ndarray, config: ClusterConfig
) -> np.ndarray:
    """Break a weak bridge only when both resulting cores have independent support.

    Local relative power determines connectivity, not whether a point survives.
    Each new core needs at least min_points reliable vertices; otherwise retain
    the original component. Reassign every remaining member to a supported core.
    """
    if not len(points):
        return labels
    distances = np.linalg.norm(points[:, None, :2] - points[None, :, :2], axis=-1)
    neighbours = distances <= config.fine_radius_m
    local_peak = np.max(np.where(neighbours, points[None, :, 4], -np.inf), axis=1)
    reliable = (neighbours.sum(axis=1) >= config.min_points) & (
        points[:, 4] >= local_peak - 10 * np.log10(config.core_power_ratio)
    )
    labels = labels.copy()
    next_label = int(labels.max()) + 1
    for label in sorted(set(labels) - {-1}):
        members = np.flatnonzero(labels == label)
        remaining = set(members[reliable[members]])
        groups = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            group, queue = [seed], [seed]
            while queue:
                current = queue.pop()
                more = {p for p in remaining if neighbours[current, p]}
                remaining -= more
                group.extend(sorted(more))
                queue.extend(sorted(more))
            if len(group) >= config.min_points:
                groups.append(group)
        if len(groups) < 2:
            continue
        distance_to_core = np.array([distances[np.ix_(members, g)].min(axis=1) for g in groups])
        nearest = distance_to_core.argmin(axis=0)
        for index in range(len(groups)):
            labels[members[nearest == index]] = label if index == 0 else next_label
            next_label += index > 0
    return labels


class ObservationTracker:
    """Shared minimal alpha-beta association, for ablations only; not TI GTRACK."""

    def __init__(self, *, max_tracks: int | None = None) -> None:
        if max_tracks is not None and (
            not isinstance(max_tracks, int) or isinstance(max_tracks, bool) or max_tracks < 1
        ):
            raise ValueError("max_tracks must be positive or None")
        self.max_tracks = max_tracks
        self.tracks: dict[int, dict] = {}
        self.next_id = 0
        self.last_matches: dict[int, dict] = {}

    def assign(self, keys: list[int], distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Original distance-only assignment, retained unchanged for comparison."""
        cost = np.concatenate(
            (np.where(distance <= 0.8, distance, 1e6), np.full((len(keys), len(keys)), 0.81)),
            axis=1,
        )
        return _native.linear_sum_assignment(cost)

    def step(self, observations: list[dict], dt: float = 0.1) -> list[dict]:
        self.last_matches = {}
        keys = sorted(self.tracks)
        for track in self.tracks.values():
            track["age"] += 1
            track["position"] += track["velocity"] * dt
            track["association_position"] += track["association_velocity"] * dt
            if "covariance" in track:
                transition = np.eye(4)
                transition[:2, 2:] = np.eye(2) * dt
                acceleration = np.vstack((np.eye(2) * dt**2 / 2, np.eye(2) * dt))
                track["covariance"] = transition @ track["covariance"] @ transition.T + (
                    acceleration @ acceleration.T * 0.5**2
                )
            track["misses"] += 1
        matched = set()
        if keys and observations:
            positions = np.array([self.tracks[k]["association_position"] for k in keys])
            centers = np.array([o["center"] for o in observations])
            association_centers = np.array(
                [o.get("association_center", o["center"]) for o in observations]
            )
            distance = np.linalg.norm(positions[:, None] - association_centers[None, :], axis=-1)
            left, right = self.assign(keys, distance)
            for i, j in zip(left, right, strict=True):
                if j >= len(observations) or distance[i, j] > 0.8:
                    continue
                track = self.tracks[keys[i]]
                residual = centers[j] - track["position"]
                if "covariance" in observations[j]:
                    covariance = track["covariance"]
                    noise = np.array(observations[j]["covariance"])
                    gain = np.linalg.solve((covariance[:2, :2] + noise).T, covariance[:, :2].T).T
                    update = gain @ residual
                    track["position"] += update[:2]
                    track["velocity"] += update[2:]
                    complement = np.eye(4)
                    complement[:, :2] -= gain
                    track["covariance"] = (
                        complement @ covariance @ complement.T + gain @ noise @ gain.T
                    )
                else:
                    track["position"] += 0.65 * residual
                    track["velocity"] += 0.1 * residual / dt
                association_residual = association_centers[j] - track["association_position"]
                track["association_position"] += 0.65 * association_residual
                track["association_velocity"] += 0.1 * association_residual / dt
                track["misses"] = 0
                track["hits"] += 1
                self.last_matches[keys[i]] = observations[j]
                matched.add(int(j))
        self._allocate(observations, matched)
        self.tracks = {k: v for k, v in self.tracks.items() if v["misses"] <= 3}
        return [
            dict(id=k, xy=v["position"].tolist(), coasting=v["misses"] > 0)
            for k, v in self.tracks.items()
            if v["hits"] >= 3
        ]

    def _allocate(self, observations: list[dict], matched: set[int]) -> None:
        allocation_order = list(range(len(observations)))
        if self.max_tracks is not None:
            allocation_order.sort(
                key=lambda j: (-len(observations[j]["members"]), -observations[j]["snr_sum"], j)
            )
        for j in allocation_order:
            if self.max_tracks is not None and len(self.tracks) >= self.max_tracks:
                break
            observation = observations[j]
            if j not in matched and observation.get("allocation_allowed", True):
                self.tracks[self.next_id] = dict(
                    position=np.array(observation["center"], dtype=float),
                    velocity=np.zeros(2),
                    association_position=np.array(
                        observation.get("association_center", observation["center"]), dtype=float
                    ),
                    association_velocity=np.zeros(2),
                    misses=0,
                    hits=1,
                    age=1,
                )
                if "covariance" in observation:
                    covariance = np.eye(4) * 0.5**2
                    covariance[:2, :2] = observation["covariance"]
                    self.tracks[self.next_id]["covariance"] = covariance
                self.last_matches[self.next_id] = observation
                self.next_id += 1


def geometric_median(positions: np.ndarray) -> np.ndarray:
    """Rotation-equivariant robust center; only one to three temporal estimates."""
    center = positions.mean(axis=0)
    for _ in range(30):
        distances = np.linalg.norm(positions - center, axis=1)
        inverse = 1 / np.maximum(distances, 1e-8)
        center = inverse @ positions / inverse.sum()
    return center


class DopplerMotionTracker(ObservationTracker):
    """Causal motion estimate with unchanged two-scale association and lifetime.

    Past positions are transported to the current time before robust combination.
    Doppler corrects bulk velocity along the observed line of sight, with weaker
    influence for dispersed residuals. Raw points and local Doppler survive.
    """

    def __init__(
        self,
        *,
        doppler: bool = True,
        temporal: bool = True,
        association_guard: bool = True,
        max_tracks: int | None = None,
        height_m: float = 1.5,
        velocity_resolution_mps: float = DEFAULT_CONFIG.velocity_scale_mps / 3,
    ) -> None:
        super().__init__(max_tracks=max_tracks)
        if (
            not np.isfinite([height_m, velocity_resolution_mps]).all()
            or min(height_m, velocity_resolution_mps) <= 0
        ):
            raise ValueError("Sensor height and Doppler resolution must be positive and finite")
        self.height_m = height_m
        self.velocity_scale_mps = 3 * velocity_resolution_mps
        self.doppler = doppler
        self.temporal = temporal
        self.association_guard = association_guard
        self.position_history: dict[int, list[tuple[float, np.ndarray]]] = {}
        self.time_s = 0.0
        self.clustering_method = "two_scale"

    def step_points(self, points: np.ndarray, dt: float = 0.1) -> tuple:
        points = np.asarray(points, dtype=float).reshape(-1, 5)
        if not np.isfinite(points).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Points must be finite and dt positive")
        self.time_s += dt
        labels, weights, observations = cluster_points(points, self.clustering_method)
        output = self.step(observations, dt)
        for report in output:
            tid = report["id"]
            state = self.tracks[tid]
            velocity = state["velocity"].copy()
            measurement = self.last_matches.get(tid)
            if self.doppler and measurement is not None:
                subset = points[measurement["members"]]
                radius = np.sqrt(
                    np.sum(subset[:, :2] ** 2, axis=1) + (subset[:, 2] - self.height_m) ** 2
                )
                los = subset[:, :2] / np.maximum(radius[:, None], 1e-8)
                residual = subset[:, 3] - los @ velocity
                median = float(np.median(residual))
                scale = max(
                    self.velocity_scale_mps / 3,
                    float(1.4826 * np.median(abs(residual - median))),
                )
                noise = scale**2 + self.velocity_scale_mps**2
                h = los.mean(axis=0)
                gain = 0.5**2 * h / (0.5**2 * (h @ h) + noise)
                velocity += gain * median
                measurement["bulk_velocity_correction"] = (gain * median).tolist()
            history = self.position_history.setdefault(tid, [])
            ambiguous = measurement is not None and any(
                other is not measurement
                and np.linalg.norm(np.array(other["center"]) - measurement["center"]) <= 0.8
                for other in observations
            )
            if self.association_guard and ambiguous:
                # Several observations inside the association scale make past
                # membership uncertain. Do not mix their positions over time.
                history.clear()
            history.append((self.time_s, state["position"].copy()))
            history[:] = history[-3:]
            if self.temporal:
                transported = np.array([p + (self.time_s - time) * velocity for time, p in history])
                report["xy"] = geometric_median(transported).tolist()
            report["bulk_velocity_xy_mps"] = velocity.tolist()
            report["measurement_members"] = measurement["members"] if measurement else []
            report["temporal_samples"] = len(history) if self.temporal else 1
            report["ambiguous_neighbour"] = ambiguous
        self.position_history = {
            tid: history for tid, history in self.position_history.items() if tid in self.tracks
        }
        return labels, weights, observations, output


class RecentSupportTracker(DopplerMotionTracker):
    """Prefer recently measured confirmed tracks at the same matching cardinality.

    First retain the number of matches admitted by the original distance
    objective. Within that count, prioritize recent confirmed support and then
    spatial distance. Forcing extra matches can revive a stale limb hypothesis
    by taking a current core's measurement, so cardinality is not increased.
    """

    def __init__(
        self,
        *,
        prefer_recent: bool = True,
        split_cores: bool = True,
        temporal: bool = True,
        max_tracks: int | None = None,
        height_m: float = 1.5,
        velocity_resolution_mps: float = DEFAULT_CONFIG.velocity_scale_mps / 3,
    ) -> None:
        super().__init__(
            temporal=temporal,
            max_tracks=max_tracks,
            height_m=height_m,
            velocity_resolution_mps=velocity_resolution_mps,
        )
        self.prefer_recent = prefer_recent
        self.clustering_method = "power_split" if split_cores else "two_scale"

    def assign(self, keys: list[int], distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left, right = super().assign(keys, distance)
        if not self.prefer_recent:
            return left, right
        count = sum(
            j < distance.shape[1] and distance[i, j] <= 0.8
            for i, j in zip(left, right, strict=True)
        )
        rows, columns = distance.shape
        recent = np.array(
            [self.tracks[key]["hits"] >= 3 and self.tracks[key]["misses"] == 1 for key in keys]
        )
        # The rectangular dummy blocks enforce exactly count real assignments.
        # Every original feasible assignment remains possible.
        cost = np.full((rows + columns - count, columns + rows - count), 1e6)
        cost[:rows, :columns] = np.where(
            distance <= 0.8, distance - (count * 0.8 + 1) * recent[:, None], 1e6
        )
        cost[:rows, columns:] = 0
        cost[rows:, :columns] = 0
        left, right = _native.linear_sum_assignment(cost)
        keep = left < rows
        return left[keep], right[keep]


class ScatterBodyTracker(RecentSupportTracker):
    """Maintain scatter-component histories and explicit split-origin body hypotheses.

    A new weak component may share a body only when it and a continuing component
    jointly match distinct points of the previous observation. Signal strength
    never blocks birth. Independent strong support immediately releases a child
    from that hypothesis; proximity alone cannot attach an existing component.

    max_components bounds scatter histories. max_bodies bounds independent body
    hypotheses, including tentative/coasting ones, after split-origin checks.
    """

    def __init__(
        self,
        *,
        lineage: bool = True,
        prefer_recent: bool = True,
        split_cores: bool = True,
        temporal: bool = True,
        max_components: int | None = None,
        max_bodies: int | None = None,
        height_m: float = 1.5,
        velocity_resolution_mps: float = DEFAULT_CONFIG.velocity_scale_mps / 3,
    ) -> None:
        super().__init__(
            prefer_recent=prefer_recent,
            split_cores=split_cores,
            temporal=temporal,
            max_tracks=max_components,
            height_m=height_m,
            velocity_resolution_mps=velocity_resolution_mps,
        )
        if max_bodies is not None and (
            not isinstance(max_bodies, int) or isinstance(max_bodies, bool) or max_bodies < 1
        ):
            raise ValueError("max_bodies must be positive or None")
        self.max_bodies = max_bodies
        self.lineage = lineage
        self.previous_clouds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.parents: dict[int, int] = {}
        self.lineage_events: list[dict] = []
        self.component_support: dict[int, tuple[float, float]] = {}

    @staticmethod
    def support(points: np.ndarray) -> tuple[float, float]:
        peak_db = float(points[:, 4].max())
        signal = np.power(10.0, (points[:, 4] - peak_db) / 10)
        effective_count = signal.sum() ** 2 / (signal @ signal)
        return peak_db + 10 * float(np.log10(signal.sum())), float(effective_count)

    @staticmethod
    def independently_supported(child: tuple[float, float], parent: tuple[float, float]) -> bool:
        # A comparable number of effective scatterers OR comparable power is
        # enough to preserve an independent hypothesis. No absolute SNR gate.
        return child[0] >= parent[0] - 10 * np.log10(DEFAULT_CONFIG.core_power_ratio) - 1e-10 or (
            child[1] >= parent[1] / 2 - 1e-10
        )

    @staticmethod
    def split_support(
        previous: np.ndarray, child: np.ndarray, parent: np.ndarray
    ) -> tuple[int, int]:
        current = np.concatenate((child, parent))
        distance = np.linalg.norm(current[:, None] - previous[None], axis=-1)
        cost = np.concatenate(
            (
                np.where(distance <= DEFAULT_CONFIG.fine_radius_m, distance, 1e6),
                np.full((len(current), len(current)), len(current) + 1.0),
            ),
            axis=1,
        )
        left, right = _native.linear_sum_assignment(cost)
        matched = [
            i
            for i, j in zip(left, right, strict=True)
            if j < len(previous) and distance[i, j] <= DEFAULT_CONFIG.fine_radius_m
        ]
        return sum(i < len(child) for i in matched), sum(i >= len(child) for i in matched)

    def _update_lineage(self, points: np.ndarray, old_ids: set[int], dt: float) -> None:
        matched = self.last_matches
        support = dict(self.component_support)
        support.update({tid: self.support(points[o["members"]]) for tid, o in matched.items()})
        for child in sorted(set(self.tracks) - old_ids):
            child_points = points[matched[child]["members"], :2]
            candidates = []
            for parent, observation in matched.items():
                if parent not in self.previous_clouds or self.tracks[parent]["hits"] < 3:
                    continue
                if self.independently_supported(support[child], support[parent]):
                    continue
                previous, velocity = self.previous_clouds[parent]
                parent_points = points[observation["members"], :2]
                child_count, parent_count = self.split_support(
                    previous + dt * velocity, child_points, parent_points
                )
                if child_count >= max(
                    DEFAULT_CONFIG.min_points, len(child_points) / 2
                ) and parent_count >= max(DEFAULT_CONFIG.min_points, len(parent_points) / 2):
                    candidates.append((parent, child_count, parent_count))
            if len(candidates) == 1:
                parent, child_count, parent_count = candidates[0]
                self.parents[child] = parent
                self.lineage_events.append(
                    dict(
                        child=child,
                        parent=parent,
                        matched_child_points=int(child_count),
                        matched_parent_points=int(parent_count),
                    )
                )
        for child, parent in list(self.parents.items()):
            if (
                child not in self.tracks
                or parent not in self.tracks
                or self.independently_supported(support[child], support[parent])
            ):
                del self.parents[child]
                continue
            separation = np.linalg.norm(
                self.tracks[child]["association_position"]
                - self.tracks[parent]["association_position"]
            )
            if separation > DEFAULT_CONFIG.outer_radius_m:
                del self.parents[child]
        self.component_support = {
            tid: value for tid, value in support.items() if tid in self.tracks
        }

    def _body_root(self, tid: int) -> int:
        while tid in self.parents:
            tid = self.parents[tid]
        return tid

    def _limit_bodies(self, previous_roots: set[int]) -> None:
        if self.max_bodies is None:
            return
        roots = {self._body_root(tid) for tid in self.tracks}
        if len(roots) <= self.max_bodies:
            return

        def priority(tid: int) -> tuple:
            observation = self.last_matches.get(tid, {})
            # Keep admitted bodies until their normal expiry. For simultaneous
            # births use the existing allocation order: point count, then SNR.
            return (
                tid not in previous_roots,
                -len(observation.get("members", [])),
                -observation.get("snr_sum", 0),
                tid,
            )

        admitted = set(sorted(roots, key=priority)[: self.max_bodies])
        rejected = {tid for tid in self.tracks if self._body_root(tid) not in admitted}
        # Reject whole hypotheses in the backend, including point associations.
        # A person-count prior is not evidence for joining unrelated components.
        for tid in rejected:
            del self.tracks[tid]
            self.last_matches.pop(tid, None)
            self.position_history.pop(tid, None)
            self.component_support.pop(tid, None)
            self.parents.pop(tid, None)
        self.lineage_events = [e for e in self.lineage_events if e["child"] not in rejected]

    def step_points(self, points: np.ndarray, dt: float = 0.1) -> tuple:
        points = np.asarray(points, dtype=float).reshape(-1, 5)
        old_ids = set(self.tracks)
        previous_roots = old_ids - self.parents.keys()
        labels, weights, observations, components = super().step_points(points, dt)
        self.lineage_events = []
        if self.lineage:
            self._update_lineage(points, old_ids, dt)
        # Candidate components must reach the split test before counting bodies.
        self._limit_bodies(previous_roots)
        components = [c for c in components if c["id"] in self.tracks]
        self.previous_clouds = {
            tid: (points[o["members"], :2].copy(), self.tracks[tid]["association_velocity"].copy())
            for tid, o in self.last_matches.items()
        }
        groups: dict[int, list[dict]] = {}
        by_id = {c["id"]: c for c in components}
        for component in components:
            root = self._body_root(component["id"])
            groups.setdefault(root, []).append(component)
        bodies = [
            dict(
                by_id[root],
                component_ids=[c["id"] for c in group],
                components=group,
                body_measurement_members=[i for c in group for i in c["measurement_members"]],
            )
            for root, group in groups.items()
        ]
        return labels, weights, observations, bodies
