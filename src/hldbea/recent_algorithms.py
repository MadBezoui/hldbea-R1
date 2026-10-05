"""Audited Python adapters for post-2024 PlatEMO comparators.

The mathematical routines in this module are ports of MaOEA-HAP and FDSEA
from PlatEMO commit d25e65d1ffba58dbf4d7e1b5259786187d12968a.  PlatEMO's
research-use notice requires acknowledgement of the platform and its 2017
paper in publications using the code.  The adapters are limited to the real,
unconstrained benchmark setting used by this study and share the experiment
harness's exact function-evaluation ledger.
"""

from __future__ import annotations

import math

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


PLATEMO_COMMIT = "d25e65d1ffba58dbf4d7e1b5259786187d12968a"


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return array


def _cosine_similarity(values: np.ndarray) -> np.ndarray:
    values = _finite_matrix(values, "values")
    norms = np.linalg.norm(values, axis=1)
    safe = np.maximum(norms, np.finfo(float).eps)
    cosine = (values @ values.T) / np.outer(safe, safe)
    return np.clip(cosine, -1.0, 1.0)


def associate_angle(objectives: np.ndarray) -> np.ndarray:
    """Port PlatEMO ``AssociateAngle.m`` with clipped cosine arithmetic."""

    objectives = _finite_matrix(objectives, "objectives")
    n_points, n_obj = objectives.shape
    cosine = _cosine_similarity(objectives)
    np.fill_diagonal(cosine, 0.0)
    angles = np.sort(np.arccos(cosine), axis=1)[:, : min(n_obj, n_points)]
    weights = 1.0 / np.arange(1, angles.shape[1] + 1, dtype=float)
    return angles @ weights


def _front_one(objectives: np.ndarray) -> np.ndarray:
    return NonDominatedSorting().do(objectives, only_non_dominated_front=True)


def estimate_shape_exponent(objectives: np.ndarray) -> float:
    """Estimate the Pareto-front Lp exponent used by MaOEA-HAP."""

    objectives = _finite_matrix(objectives, "objectives")
    front = objectives[_front_one(objectives)]
    ideal = np.min(front, axis=0)
    nadir = np.max(front, axis=0)
    span = np.maximum(nadir - ideal, np.finfo(float).eps)
    normalized = (front - ideal) / span
    candidates = np.round(np.arange(0.5, 2.01, 0.1), 10)
    variation = np.empty(len(candidates), dtype=float)
    for index, exponent in enumerate(candidates):
        radius = np.sum(normalized**exponent, axis=1) ** (1.0 / exponent)
        ordered = np.sort(radius)
        n_points = len(ordered)
        q1 = ordered[max(int(np.fix(n_points * 0.25)), 1) - 1]
        q3 = ordered[max(int(np.fix(n_points * 0.75)), 1) - 1]
        upper = q3 + 1.5 * (q3 - q1)
        kept = radius[radius <= upper]
        maximum = np.max(kept) if len(kept) else 0.0
        variation[index] = 0.0 if maximum <= 0.0 else np.std(kept / maximum, ddof=1 if len(kept) > 1 else 0)
    return float(candidates[int(np.argmin(variation))])


def frequency_decode(model_parameters: np.ndarray, n_var: int) -> np.ndarray:
    """Port the cosine-series decision mapping in PlatEMO ``Cal_Dec.m``."""

    parameters = _finite_matrix(model_parameters, "model_parameters")
    if not isinstance(n_var, int) or isinstance(n_var, bool) or n_var <= 0:
        raise ValueError("n_var must be a positive integer")
    if parameters.shape[1] < 4 or (parameters.shape[1] - 2) % 2:
        raise ValueError("model parameter width must be 2*K+2")
    n_terms = (parameters.shape[1] - 2) // 2
    coordinates = np.arange(1, n_var + 1, dtype=float)
    decoded = np.repeat((parameters[:, -1] / 2.0)[:, None], n_var, axis=1)
    omega = parameters[:, -2]
    for harmonic in range(1, n_terms + 1):
        amplitude = parameters[:, 2 * harmonic]
        phase = parameters[:, 2 * harmonic + 1]
        decoded += amplitude[:, None] * np.cos(
            harmonic * omega[:, None] * coordinates[None, :] + phase[:, None]
        )
    return decoded


def _frequency_encode(decisions: np.ndarray, n_terms: int) -> np.ndarray:
    """Port the current PlatEMO ``Cal_MP.m`` parameter reconstruction."""

    decisions = _finite_matrix(decisions, "decisions")
    n_points, n_var = decisions.shape
    parameters = np.zeros((n_points, 2 * n_terms + 2), dtype=float)
    omega = 2.0 * np.pi / n_var
    parameters[:, -1] = np.sum(decisions, axis=1) / n_var
    parameters[:, -2] = omega
    # The upstream implementation deliberately retains odd low-frequency terms.
    coordinates = np.arange(1, n_var + 1, dtype=float)
    for harmonic in range(1, int(n_terms / 2) + 1, 2):
        an = 2.0 * np.sum(decisions * np.cos(harmonic * omega * coordinates), axis=1) / n_var
        bn = 2.0 * np.sum(decisions * np.sin(harmonic * omega * coordinates), axis=1) / n_var
        parameters[:, harmonic - 1] = np.hypot(an, bn)
        parameters[:, harmonic] = -np.arctan2(bn, an)
    return parameters


def _polynomial_mutation(
    decisions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    probability: float,
    eta: float,
) -> np.ndarray:
    values = np.clip(np.asarray(decisions, dtype=float), lower, upper).copy()
    site = np.random.random(values.shape) < probability
    mu = np.random.random(values.shape)
    span = np.maximum(upper - lower, np.finfo(float).eps)
    low = site & (mu <= 0.5)
    delta = (values - lower) / span
    values[low] += span[low] * (
        (2.0 * mu[low] + (1.0 - 2.0 * mu[low]) * (1.0 - delta[low]) ** (eta + 1.0))
        ** (1.0 / (eta + 1.0))
        - 1.0
    )
    high = site & (mu > 0.5)
    delta = (upper - values) / span
    values[high] += span[high] * (
        1.0
        - (
            2.0 * (1.0 - mu[high])
            + 2.0 * (mu[high] - 0.5) * (1.0 - delta[high]) ** (eta + 1.0)
        )
        ** (1.0 / (eta + 1.0))
    )
    return np.clip(values, lower, upper)


def _operator_ga(
    parents: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    crossover_probability: float,
    crossover_eta: float,
    mutation_probability: float,
    mutation_eta: float,
) -> np.ndarray:
    parents = _finite_matrix(parents, "parents")
    n_pairs = len(parents) // 2
    if n_pairs == 0:
        return np.empty((0, parents.shape[1]), dtype=float)
    parent1 = parents[:n_pairs]
    parent2 = parents[n_pairs : 2 * n_pairs]
    mu = np.random.random(parent1.shape)
    beta = np.empty_like(mu)
    low = mu <= 0.5
    beta[low] = (2.0 * mu[low]) ** (1.0 / (crossover_eta + 1.0))
    beta[~low] = (2.0 - 2.0 * mu[~low]) ** (-1.0 / (crossover_eta + 1.0))
    beta *= (-1.0) ** np.random.randint(0, 2, size=beta.shape)
    beta[np.random.random(beta.shape) < 0.5] = 1.0
    disabled = np.random.random(n_pairs) > crossover_probability
    beta[disabled] = 1.0
    offspring = np.row_stack(
        (
            (parent1 + parent2) / 2.0 + beta * (parent1 - parent2) / 2.0,
            (parent1 + parent2) / 2.0 - beta * (parent1 - parent2) / 2.0,
        )
    )
    return _polynomial_mutation(
        offspring,
        np.broadcast_to(lower, offspring.shape),
        np.broadcast_to(upper, offspring.shape),
        mutation_probability,
        mutation_eta,
    )


def _angular_front_numbers(objectives: np.ndarray, n_sort: int) -> np.ndarray:
    """Small-population branch of PlatEMO ``NDQSort.m``."""

    objectives = _finite_matrix(objectives, "objectives")
    unique, inverse, counts = np.unique(
        objectives, axis=0, return_inverse=True, return_counts=True
    )
    n_points, n_obj = unique.shape
    angle = associate_angle(unique)
    front = np.full(n_points, np.inf)
    max_front = 0
    while np.sum(counts[np.isfinite(front)]) < min(n_sort, len(inverse)):
        max_front += 1
        for i in range(n_points):
            if np.isfinite(front[i]):
                continue
            dominated = False
            for j in range(i - 1, -1, -1):
                if front[j] != max_front:
                    continue
                objective = 1
                while objective < n_obj and unique[i, objective] >= unique[j, objective]:
                    objective += 1
                dominated = objective >= n_obj and angle[i] < angle[j]
                if dominated or n_obj == 2:
                    break
            if not dominated:
                front[i] = max_front
    unresolved = ~np.isfinite(front)
    front[unresolved] = max_front + 1
    return front[inverse].astype(int)


def _update_convergence_archive(archive: Population, new: Population, size: int) -> Population:
    combined = new if len(archive) == 0 else Population.merge(archive, new)
    if len(combined) <= size:
        return combined
    objectives = np.asarray(combined.get("F"), dtype=float)
    span = np.maximum(np.ptp(objectives, axis=0), np.finfo(float).eps)
    normalized = (objectives - np.min(objectives, axis=0)) / span
    indicator = np.max(normalized[:, None, :] - normalized[None, :, :], axis=2)
    scales = np.maximum(np.max(np.abs(indicator), axis=0), np.finfo(float).eps)
    fitness = np.sum(-np.exp(-indicator / scales[None, :] / 0.05), axis=0) + 1.0
    chosen = list(range(len(combined)))
    while len(chosen) > size:
        local = int(np.argmin(fitness[chosen]))
        deleted = chosen.pop(local)
        fitness += np.exp(-indicator[deleted, :] / scales[deleted] / 0.05)
    return combined[np.asarray(chosen, dtype=int)]


def _projection_distances(objectives: np.ndarray, exponent: float, p_norm: float) -> np.ndarray:
    ideal = np.min(objectives, axis=0)
    shifted = objectives - ideal + 1e-6
    radius = np.sum(shifted**exponent, axis=1) ** (1.0 / exponent)
    projected = shifted / np.maximum(radius[:, None], np.finfo(float).eps)
    difference = np.abs(shifted - projected)
    return np.sum(difference**p_norm, axis=1) ** (1.0 / p_norm)


def _update_diversity_archive(
    archive: Population,
    new: Population,
    size: int,
    p_norm: float,
    phase: int,
    progress: float,
) -> Population:
    combined = new if len(archive) == 0 else Population.merge(archive, new)
    objectives = np.asarray(combined.get("F"), dtype=float)
    front = _angular_front_numbers(objectives, 1)
    combined = combined[front == 1]
    if len(combined) <= size:
        return combined
    objectives = np.asarray(combined.get("F"), dtype=float)
    span = np.maximum(np.ptp(objectives, axis=0), np.finfo(float).eps)
    normalized = (objectives - np.min(objectives, axis=0)) / span
    cosine = _cosine_similarity(normalized)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    exponent = estimate_shape_exponent(objectives)
    distance = _projection_distances(objectives, exponent, p_norm)
    n_points, n_obj = normalized.shape
    chosen = np.zeros(n_points, dtype=bool)
    weights = np.full((n_obj, n_obj), 1e-6) + np.eye(n_obj)
    for objective in range(n_obj):
        remaining = np.flatnonzero(~chosen)
        asf = np.max(normalized[remaining] / weights[objective], axis=1)
        asf += 0.1 * normalized[remaining, objective] / 1e-6
        chosen[remaining[int(np.argmin(asf))]] = True
    if np.sum(chosen) > size:
        selected = np.flatnonzero(chosen)
        chosen[np.random.choice(selected, np.sum(chosen) - size, replace=False)] = False
    while np.sum(chosen) < size:
        selected = np.flatnonzero(chosen)
        remaining = np.flatnonzero(~chosen)
        nearest = np.min(angle[np.ix_(remaining, selected)], axis=1)
        if phase == 0:
            angular = nearest * (1.0 + 2.0 * progress**2)
            convergence = 1.0 / np.maximum(
                distance[remaining] * (2.0 - progress**2), np.finfo(float).eps
            )
            pick = int(np.argmax(angular * convergence))
        else:
            pick = int(np.argmax(nearest))
        chosen[remaining[pick]] = True
    return combined[chosen]


def _phase_flag(population: Population, p_norm: float) -> int:
    objectives = np.asarray(population.get("F"), dtype=float)
    exponent = estimate_shape_exponent(objectives)
    return int(np.all(_projection_distances(objectives, exponent, p_norm) < 1.0))


def _crowding_distance(objectives: np.ndarray) -> np.ndarray:
    objectives = _finite_matrix(objectives, "objectives")
    n_points, n_obj = objectives.shape
    distance = np.zeros(n_points, dtype=float)
    if n_points <= 2:
        distance[:] = np.inf
        return distance
    for objective in range(n_obj):
        order = np.argsort(objectives[:, objective], kind="mergesort")
        distance[order[[0, -1]]] = np.inf
        span = objectives[order[-1], objective] - objectives[order[0], objective]
        if span > 0:
            distance[order[1:-1]] += (
                objectives[order[2:], objective] - objectives[order[:-2], objective]
            ) / span
    return distance


class MaOEAHAP(Algorithm):
    """Real-valued MaOEA-HAP adapter following the audited PlatEMO source."""

    def __init__(
        self,
        *,
        pop_size: int,
        crossover_probability: float,
        crossover_eta: float,
        mutation_probability: float,
        mutation_eta: float,
        evaluator,
        seed: int,
    ) -> None:
        super().__init__(evaluator=evaluator, seed=seed)
        self.pop_size = int(pop_size)
        self.crossover_probability = float(crossover_probability)
        self.crossover_eta = float(crossover_eta)
        self.mutation_probability = float(mutation_probability)
        self.mutation_eta = float(mutation_eta)
        self.ca = Population.empty()
        self.da = Population.empty()
        self.phase = 0

    def _initialize_infill(self):
        decisions = np.random.uniform(self.problem.xl, self.problem.xu, (self.pop_size, self.problem.n_var))
        return Population.new("X", decisions)

    def _initialize_advance(self, infills=None, **kwargs):
        self.ca = _update_convergence_archive(Population.empty(), infills, self.pop_size)
        self.da = _update_diversity_archive(
            Population.empty(), infills, self.pop_size, 1.0 / self.problem.n_obj, 0, 0.0
        )
        self.pop = self.da

    def _mating(self) -> tuple[np.ndarray, np.ndarray]:
        count = int(math.ceil(self.pop_size / 2))
        first = np.random.randint(0, len(self.ca), count)
        second = np.random.randint(0, len(self.ca), count)
        f1 = np.asarray(self.ca[first].get("F"), dtype=float)
        f2 = np.asarray(self.ca[second].get("F"), dtype=float)
        relation = np.any(f1 < f2, axis=1).astype(int) - np.any(f1 > f2, axis=1).astype(int)
        winners = np.where(relation == 1, first, second)
        random_da = np.random.randint(0, len(self.da), count)
        crossover_parents = np.row_stack((self.ca[winners].get("X"), self.da[random_da].get("X")))
        crowding = _crowding_distance(np.asarray(self.ca.get("F"), dtype=float))
        mutation_indices = []
        for _ in range(self.pop_size):
            pair = np.random.choice(len(self.ca), 2, replace=len(self.ca) < 2)
            if crowding[pair[0]] == crowding[pair[1]]:
                mutation_indices.append(pair[np.random.randint(0, 2)])
            else:
                mutation_indices.append(pair[int(crowding[pair[1]] > crowding[pair[0]])])
        return crossover_parents, np.asarray(self.ca[mutation_indices].get("X"), dtype=float)

    def _infill(self):
        crossover_parents, mutation_parents = self._mating()
        crossover = _operator_ga(
            crossover_parents,
            self.problem.xl,
            self.problem.xu,
            crossover_probability=self.crossover_probability,
            crossover_eta=self.crossover_eta,
            mutation_probability=0.0,
            mutation_eta=self.mutation_eta,
        )
        mutation = _operator_ga(
            mutation_parents,
            self.problem.xl,
            self.problem.xu,
            crossover_probability=0.0,
            crossover_eta=self.crossover_eta,
            mutation_probability=self.mutation_probability,
            mutation_eta=self.mutation_eta,
        )
        decisions = np.row_stack((crossover, mutation))
        remaining = self.evaluator.ledger.remaining
        return Population.new("X", decisions[:remaining])

    def _advance(self, infills=None, **kwargs):
        self.ca = _update_convergence_archive(self.ca, infills, self.pop_size)
        progress = self.evaluator.ledger.used / self.evaluator.ledger.max_evaluations
        self.da = _update_diversity_archive(
            self.da,
            infills,
            self.pop_size,
            1.0 / self.problem.n_obj,
            self.phase,
            progress,
        )
        self.phase = _phase_flag(self.da, 1.0 / self.problem.n_obj)
        self.pop = self.da


def _reference_survival(problem, population: Population, size: int, ref_dirs: np.ndarray) -> Population:
    if len(population) <= size:
        return population
    survival = ReferenceDirectionSurvival(ref_dirs)
    return survival.do(problem, population, n_survive=size)


def _de_operator(
    base: np.ndarray,
    parent2: np.ndarray,
    parent3: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mutation_probability: float,
    mutation_eta: float,
) -> np.ndarray:
    offspring = base + 0.5 * (parent2 - parent3)
    return _polynomial_mutation(
        offspring,
        np.broadcast_to(lower, offspring.shape),
        np.broadcast_to(upper, offspring.shape),
        mutation_probability,
        mutation_eta,
    )


class FDSEA(Algorithm):
    """Real-valued FDSEA adapter for the study's unconstrained benchmarks."""

    def __init__(
        self,
        *,
        pop_size: int,
        ref_dirs: np.ndarray,
        frequency_terms: int,
        operator: str,
        crossover_probability: float,
        crossover_eta: float,
        mutation_probability: float,
        mutation_eta: float,
        evaluator,
        seed: int,
    ) -> None:
        super().__init__(evaluator=evaluator, seed=seed)
        self.pop_size = int(pop_size)
        self.ref_dirs = np.asarray(ref_dirs, dtype=float)
        self.frequency_terms = int(frequency_terms)
        self.operator = operator
        self.crossover_probability = float(crossover_probability)
        self.crossover_eta = float(crossover_eta)
        self.mutation_probability = float(mutation_probability)
        self.mutation_eta = float(mutation_eta)
        self.gamma = 0.5
        self.population1 = Population.empty()
        self.population2 = Population.empty()

    def _normalized_to_problem(self, decisions: np.ndarray) -> np.ndarray:
        normalized = np.clip(decisions, 0.0, 1.0)
        return self.problem.xl + (self.problem.xu - self.problem.xl) * normalized

    def _problem_to_normalized(self, decisions: np.ndarray) -> np.ndarray:
        span = np.maximum(self.problem.xu - self.problem.xl, np.finfo(float).eps)
        return np.clip((decisions - self.problem.xl) / span, 0.0, 1.0)

    def _initialize_infill(self):
        if self.evaluator.ledger.max_evaluations < 2 * self.pop_size:
            raise ValueError("FDSEA requires at least 2*population_size evaluations")
        parameters = np.random.random((self.pop_size, 2 * self.frequency_terms + 2))
        frequency_decisions = self._normalized_to_problem(
            frequency_decode(parameters, self.problem.n_var)
        )
        direct_decisions = np.random.uniform(
            self.problem.xl, self.problem.xu, (self.pop_size, self.problem.n_var)
        )
        population = Population.new("X", np.row_stack((frequency_decisions, direct_decisions)))
        population.set("Stream", np.r_[np.ones(self.pop_size, dtype=int), np.full(self.pop_size, 2, dtype=int)])
        model = np.full((2 * self.pop_size, parameters.shape[1]), np.nan)
        model[: self.pop_size] = parameters
        population.set("ModelParameters", model)
        return population

    def _initialize_advance(self, infills=None, **kwargs):
        stream = infills.get("Stream")
        self.population1 = infills[stream == 1]
        self.population2 = infills[stream == 2]
        encoded = _frequency_encode(
            self._problem_to_normalized(np.asarray(self.population2.get("X"), dtype=float)),
            self.frequency_terms,
        )
        self.population2.set("ModelParameters", encoded)
        self.population1 = _reference_survival(self.problem, self.population1, self.pop_size, self.ref_dirs)
        self.population2 = _reference_survival(self.problem, self.population2, self.pop_size, self.ref_dirs)
        self.pop = _reference_survival(
            self.problem,
            Population.merge(self.population1, self.population2),
            self.pop_size,
            self.ref_dirs,
        )

    def _frequency_offspring(self) -> tuple[np.ndarray, np.ndarray]:
        models = np.asarray(self.population1.get("ModelParameters"), dtype=float)
        if self.operator == "de":
            first = models[np.random.randint(0, len(models), self.pop_size)]
            second = models[np.random.randint(0, len(models), self.pop_size)]
            third = models[np.random.randint(0, len(models), self.pop_size)]
            offspring = _de_operator(
                first,
                second,
                third,
                np.zeros(first.shape[1]),
                np.ones(first.shape[1]),
                self.mutation_probability,
                self.mutation_eta,
            )
        else:
            parents = models[np.random.randint(0, len(models), self.pop_size)]
            offspring = _operator_ga(
                parents,
                np.zeros(models.shape[1]),
                np.ones(models.shape[1]),
                crossover_probability=self.crossover_probability,
                crossover_eta=self.crossover_eta,
                mutation_probability=self.mutation_probability,
                mutation_eta=self.mutation_eta,
            )
        decisions = self._normalized_to_problem(frequency_decode(offspring, self.problem.n_var))
        return decisions, offspring

    def _direct_offspring(self) -> tuple[np.ndarray, np.ndarray]:
        decisions = np.asarray(self.population2.get("X"), dtype=float)
        order = np.random.permutation(len(decisions))
        n_ga = int(round(self.gamma * len(decisions)))
        n_ga -= n_ga % 2
        if n_ga >= len(decisions) - 1:
            n_ga = len(decisions)
        ga = np.empty((0, decisions.shape[1]))
        de = np.empty((0, decisions.shape[1]))
        if n_ga:
            group = decisions[order[:n_ga]]
            ga = _operator_ga(
                group,
                self.problem.xl,
                self.problem.xu,
                crossover_probability=self.crossover_probability,
                crossover_eta=self.crossover_eta,
                mutation_probability=self.mutation_probability,
                mutation_eta=self.mutation_eta,
            )
        if n_ga < len(decisions):
            group = decisions[order[n_ga:]]
            de = _de_operator(
                group,
                group[np.random.randint(0, len(group), len(group))],
                group[np.random.randint(0, len(group), len(group))],
                self.problem.xl,
                self.problem.xu,
                self.mutation_probability,
                self.mutation_eta,
            )
        return np.row_stack((ga, de)), np.r_[np.ones(len(ga), dtype=int), -np.ones(len(de), dtype=int)]

    def _infill(self):
        frequency_x, models = self._frequency_offspring()
        direct_x, additions = self._direct_offspring()
        decisions = np.row_stack((frequency_x, direct_x))
        remaining = min(len(decisions), self.evaluator.ledger.remaining)
        population = Population.new("X", decisions[:remaining])
        stream = np.r_[np.ones(len(frequency_x), dtype=int), np.full(len(direct_x), 2, dtype=int)][:remaining]
        population.set("Stream", stream)
        model_values = np.full((remaining, models.shape[1]), np.nan)
        freq_count = int(np.sum(stream == 1))
        model_values[:freq_count] = models[:freq_count]
        population.set("ModelParameters", model_values)
        op_add = np.zeros(remaining, dtype=int)
        op_add[freq_count:] = additions[: remaining - freq_count]
        population.set("OperatorAddition", op_add)
        return population

    def _update_gamma(self) -> None:
        additions = np.asarray(self.population2.get("OperatorAddition"), dtype=int)
        if additions.shape != (len(self.population2),):
            return
        n_points = len(additions)
        n_ga = max(1, int(round(self.gamma * n_points)))
        n_de = max(1, n_points - n_ga)
        k_ga = np.sum(additions == 1)
        k_de = np.sum(additions == -1)
        eta1 = (k_ga - k_de) / max(n_points, 1)
        eta2 = k_ga / n_ga - k_de / n_de
        f1 = (self.gamma - 0.5) ** 2 + 1.0
        f2 = 2.0 / (1.0 + np.exp(-(self.gamma - 0.5) * 8.0)) - 1.0
        sign = np.sign(self.gamma - 0.5)
        f3 = sign * (-np.exp(-sign * eta2) + 1.0) if sign else 0.0
        self.gamma = float(np.clip(self.gamma + f1 * eta1 + f2 * f3, 0.1, 0.9))

    def _advance(self, infills=None, **kwargs):
        stream = np.asarray(infills.get("Stream"), dtype=int)
        frequency = infills[stream == 1]
        direct = infills[stream == 2]
        if len(frequency):
            self.population1 = _reference_survival(
                self.problem,
                Population.merge(self.population1, frequency),
                self.pop_size,
                self.ref_dirs,
            )
        if len(direct):
            self.population2.set("OperatorAddition", np.zeros(len(self.population2), dtype=int))
            self.population2 = _reference_survival(
                self.problem,
                Population.merge(self.population2, direct),
                self.pop_size,
                self.ref_dirs,
            )
            self._update_gamma()
        exchanges = min(10, len(self.population1), len(self.population2))
        for index in np.random.randint(0, min(len(self.population1), len(self.population2)), exchanges):
            encoded = _frequency_encode(
                self._problem_to_normalized(
                    np.asarray(self.population2[index : index + 1].get("X"), dtype=float)
                ),
                self.frequency_terms,
            )[0]
            self.population2[index].set("ModelParameters", encoded)
            temporary = self.population1[index]
            self.population1[index] = self.population2[index]
            self.population2[index] = temporary
        self.pop = _reference_survival(
            self.problem,
            Population.merge(self.population1, self.population2),
            self.pop_size,
            self.ref_dirs,
        )
