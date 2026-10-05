"""Named, auditable algorithm registry."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util

from .algorithm_adapters import (
    build_hldbea,
    build_pymoo_algorithm,
    build_recent_algorithm,
)
from .evaluation import EvaluationLedger
from .run_spec import RunSpec


class AlgorithmUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class AlgorithmDescriptor:
    id: str
    display_name: str
    source: str
    version: str
    citation_key: str
    exact_budget_support: bool
    optional_dependency: str | None = None
    available: bool = True
    unavailable_reason: str | None = None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def available_algorithms() -> dict[str, AlgorithmDescriptor]:
    numba_available = _module_available("numba")
    descriptors = [
        AlgorithmDescriptor(
            "hldbea", "HLDBEA", "local implementation", "revision-core-v1", "hldbea", True
        ),
        AlgorithmDescriptor(
            "nsga2", "NSGA-II", "pymoo", "0.6.1.3", "deb2002nsga2", True
        ),
        AlgorithmDescriptor(
            "spea2", "SPEA2", "pymoo", "0.6.1.3", "zitzler2001spea2", True
        ),
        AlgorithmDescriptor(
            "nsga3", "NSGA-III", "pymoo", "0.6.1.3", "deb2014nsga3", True
        ),
        AlgorithmDescriptor(
            "rvea", "RVEA", "pymoo", "0.6.1.3", "cheng2016rvea", True
        ),
        AlgorithmDescriptor(
            "moead", "MOEA/D", "pymoo", "0.6.1.3", "zhang2007moead", True
        ),
        AlgorithmDescriptor(
            "sms_emoa", "SMS-EMOA", "pymoo", "0.6.1.3", "beume2007sms", True
        ),
        AlgorithmDescriptor(
            "age_moea",
            "AGE-MOEA",
            "pymoo",
            "0.6.1.3",
            "panichella2022agemoea",
            True,
            optional_dependency="numba",
            available=numba_available,
            unavailable_reason=None
            if numba_available
            else "optional dependency numba is not installed",
        ),
        AlgorithmDescriptor(
            "maoea_hap",
            "MaOEA-HAP",
            "PlatEMO",
            "d25e65d1ffba58dbf4d7e1b5259786187d12968a",
            "yue2026maoeahap",
            True,
        ),
        AlgorithmDescriptor(
            "fdsea",
            "FDSEA",
            "PlatEMO",
            "d25e65d1ffba58dbf4d7e1b5259786187d12968a",
            "wang2026fdsea",
            True,
        ),
    ]
    return {descriptor.id: descriptor for descriptor in descriptors}


def build_algorithm(spec: RunSpec, problem, ledger: EvaluationLedger):
    if not isinstance(spec, RunSpec):
        raise TypeError("spec must be a RunSpec")
    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be an EvaluationLedger")
    if ledger.max_evaluations != spec.evaluation_budget:
        raise ValueError("ledger budget does not match run spec")
    descriptors = available_algorithms()
    if spec.algorithm_id not in descriptors:
        raise ValueError(f"unknown algorithm id: {spec.algorithm_id}")
    descriptor = descriptors[spec.algorithm_id]
    if not descriptor.available:
        raise AlgorithmUnavailable(descriptor.unavailable_reason)
    if spec.algorithm_id == "hldbea":
        algorithm, directions = build_hldbea(spec, problem, ledger)
    elif spec.algorithm_id in {"maoea_hap", "fdsea"}:
        algorithm, directions = build_recent_algorithm(spec, problem, ledger)
    else:
        algorithm, directions = build_pymoo_algorithm(spec, problem, ledger)
    algorithm._algorithm_descriptor = descriptor
    algorithm._declared_seed = spec.seed
    algorithm._evaluation_ledger = ledger
    algorithm._reference_directions = directions
    return algorithm
