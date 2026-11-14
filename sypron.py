# Probabilistic extensions for Syphen system states.

"""Sypron keeps probability laws and numerical generation separate from Syphen's
deterministic state machinery. Random variables remain VariableElements, while
stochastic functions are represented by finite random-basis expansions. """

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Optional
from uuid import UUID

import numpy as np
import sympy as sym

from syphen import (
    FunctionElement,
    SystemState,
    SystemicSpace,
    ValueConstraints,
    VariableElement,
)

__all__ = [
    "BernoulliDistribution",
    "CategoricalDistribution",
    "Distribution",
    "ExponentialDistribution",
    "FunctionRealization",
    "GeneradorNumerico",
    "NormalDistribution",
    "RandomVariableElement",
    "StateRealization",
    "StateSimulationResult",
    "StochasticFunctionElement",
    "SypronStateModel",
    "UniformDistribution",
]


def _finite_number(value: float, subject: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{subject} debe ser finito.")
    return number

# DISTRIBUTION BASE

class Distribution(ABC):
    """Numerical law that can generate scalar realizations."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float:
        raise NotImplementedError

# DISTRIBUTION TOOLS

@dataclass(frozen=True)
class NormalDistribution(Distribution):
    mean: float = 0.0
    standard_deviation: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _finite_number(self.mean, "mean"))
        object.__setattr__(
            self,
            "standard_deviation",
            _finite_number(self.standard_deviation, "standard_deviation"),
        )
        if (
            self.standard_deviation <= 0.0
        ):
            raise ValueError("standard_deviation debe ser positivo y finito.")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.standard_deviation))

@dataclass(frozen=True)
class UniformDistribution(Distribution):
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "low", _finite_number(self.low, "low"))
        object.__setattr__(self, "high", _finite_number(self.high, "high"))
        if not isfinite(self.low) or not isfinite(self.high) or self.low >= self.high:
            raise ValueError("Se requiere low < high, ambos finitos.")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

@dataclass(frozen=True)
class ExponentialDistribution(Distribution):
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _finite_number(self.scale, "scale"))
        if not isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale debe ser positivo y finito.")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.exponential(self.scale))

@dataclass(frozen=True)
class BernoulliDistribution(Distribution):
    probability: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probability",
            _finite_number(self.probability, "probability"),
        )
        if (
            not isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError("probability debe estar entre 0 y 1.")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.binomial(1, self.probability))

@dataclass(frozen=True)
class CategoricalDistribution(Distribution):
    values: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values or len(self.values) != len(self.probabilities):
            raise ValueError("values y probabilities deben tener igual longitud no vacía.")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("Los valores categóricos deben ser finitos.")
        if len(set(self.values)) != len(self.values):
            raise ValueError("Los valores categóricos deben ser únicos.")
        if any(
            not isfinite(probability) or probability < 0.0
            for probability in self.probabilities
        ):
            raise ValueError("Las probabilidades deben ser finitas y no negativas.")
        total = sum(self.probabilities)
        if total <= 0.0 or not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Las probabilidades categóricas deben sumar 1.")
        object.__setattr__(
            self,
            "probabilities",
            tuple(probability / total for probability in self.probabilities),
        )

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.choice(self.values, p=self.probabilities))

class RandomVariableElement(VariableElement):
    """Syphen state variable whose concrete values are generated by Sypron."""

    def __init__(
        self,
        systemic_space: SystemicSpace,
        name: str,
        distribution: Distribution,
        initial_value: Optional[float] = None,
        constraints: Optional[ValueConstraints] = None,
    ) -> None:
        if not isinstance(distribution, Distribution):
            raise TypeError("distribution debe implementar Distribution.")
        super().__init__(
            systemic_space=systemic_space,
            name=name,
            initial_value=initial_value,
            constraints=constraints,
        )
        self.distribution = distribution

class StochasticFunctionElement(FunctionElement):
    """Random function represented by a finite expansion with random weights.

    A realization draws each coefficient once. The same coefficient values are
    then used at every function argument within that realization.
    """

    def __init__(
        self,
        systemic_space: SystemicSpace,
        name: str,
        arguments: Sequence[sym.Symbol],
        basis: Sequence[sym.Expr],
        coefficient_distributions: Sequence[Distribution],
    ) -> None:
        function_arguments = tuple(arguments)
        basis_functions = tuple(basis)
        distributions = tuple(coefficient_distributions)
        if not function_arguments or any(
            not isinstance(argument, sym.Symbol) for argument in function_arguments
        ):
            raise ValueError("Se requiere al menos un argumento simbólico.")
        if not basis_functions or len(basis_functions) != len(distributions):
            raise ValueError("basis y coefficient_distributions deben tener igual longitud no vacía.")
        if any(not isinstance(term, sym.Expr) for term in basis_functions):
            raise TypeError("Cada función base debe ser una expresión de SymPy.")
        if any(not isinstance(law, Distribution) for law in distributions):
            raise TypeError("Cada coeficiente requiere una Distribution.")
        if any(
            term.free_symbols - set(function_arguments)
            for term in basis_functions
        ):
            raise ValueError("Las funciones base solo pueden usar los argumentos declarados.")

        coefficient_symbols = tuple(
            sym.Dummy(f"{name}_coefficient_{index}")
            for index in range(len(basis_functions))
        )
        expression = sym.Add(
            *(
                sym.Mul(coefficient, term)
                for coefficient, term in zip(coefficient_symbols, basis_functions)
            )
        )
        super().__init__(
            systemic_space=systemic_space,
            name=name,
            arguments=(*function_arguments, *coefficient_symbols),
            expression=expression,
        )
        self.arguments = function_arguments
        self.basis = basis_functions
        self.coefficient_symbols = coefficient_symbols
        self.coefficient_distributions = distributions
        self.function = sym.Lambda(function_arguments, expression)

    def evaluate(
        self,
        *arguments: float | complex | np.ndarray,
    ) -> float | complex | np.ndarray:
        if len(arguments) != len(self.arguments):
            raise ValueError(
                f"La función '{self.name}' requiere {len(self.arguments)} argumentos."
            )
        raise RuntimeError(
            "Una función estocástica requiere una FunctionRealization; "
            "genérala con SypronStateModel."
        )

    def realize(self, generator: GeneradorNumerico) -> FunctionRealization:
        coefficients = tuple(
            generator.sample(distribution)
            for distribution in self.coefficient_distributions
        )
        return FunctionRealization(self, coefficients)

# NUMERICAL EVALUATION AND SAMPLING

@dataclass(frozen=True)
class FunctionRealization:
    element: StochasticFunctionElement
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(self.element.basis):
            raise ValueError("La cantidad de coeficientes no coincide con la base.")

    def evaluate(
        self,
        *arguments: float | complex | np.ndarray,
    ) -> float | complex | np.ndarray:
        if len(arguments) != len(self.element.arguments):
            raise ValueError(
                f"La función '{self.element.name}' requiere "
                f"{len(self.element.arguments)} argumentos."
            )
        numeric_arguments = []
        for argument in arguments:
            value = np.asarray(argument)
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError("Los argumentos de la función deben ser numéricos.")
            if not np.all(np.isfinite(value)):
                raise ValueError("Los argumentos de la función deben ser finitos.")
            numeric_arguments.append(value.item() if value.ndim == 0 else value)
        result = np.asarray(
            self.element._compiled_function(
                *numeric_arguments,
                *self.coefficients,
            )
        )
        if not np.issubdtype(result.dtype, np.number):
            raise TypeError("La función produjo un resultado no numérico.")
        if not np.all(np.isfinite(result)):
            raise ValueError("La función produjo resultados no finitos.")
        if result.ndim == 0:
            return result.item()
        return result.copy()

class GeneradorNumerico:
    """Reproducible numerical source for distribution and function draws."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def sample(self, distribution: Distribution) -> float:
        if not isinstance(distribution, Distribution):
            raise TypeError("distribution debe implementar Distribution.")
        value = float(distribution.sample(self._rng))
        if not isfinite(value):
            raise ValueError("La distribución produjo una realización no finita.")
        return value

    def reset(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

@dataclass(frozen=True)
class StateRealization:
    state_id: UUID
    values: Mapping[UUID, float]
    functions: Mapping[UUID, FunctionRealization]

@dataclass(frozen=True)
class StateSimulationResult:
    variable_ids: tuple[UUID, ...]
    variable_names: tuple[str, ...]
    variable_samples: np.ndarray
    outputs: Mapping[sym.Symbol, np.ndarray]

    @property
    def mean(self) -> Mapping[sym.Symbol, float | np.ndarray]:
        return {
            symbol: (
                float(np.mean(values))
                if np.ndim(np.mean(values, axis=0)) == 0
                else np.mean(values, axis=0)
            )
            for symbol, values in self.outputs.items()
        }

    @property
    def variance(self) -> Mapping[sym.Symbol, float | np.ndarray]:
        return {
            symbol: (
                (
                    float(np.var(values, axis=0, ddof=1))
                    if np.ndim(np.var(values, axis=0, ddof=1)) == 0
                    else np.var(values, axis=0, ddof=1)
                )
                if len(values) > 1
                else float("nan")
            )
            for symbol, values in self.outputs.items()
        }

    def probability_greater_than(
        self,
        symbol: sym.Symbol,
        threshold: float,
    ) -> float:
        if symbol not in self.outputs:
            raise KeyError("El símbolo no pertenece a los resultados simulados.")
        values = np.asarray(self.outputs[symbol])
        if values.ndim != 1:
            raise ValueError(
                "La probabilidad de excedencia requiere un resultado escalar por ensayo."
            )
        if len(values) == 0:
            raise ValueError("No hay ensayos para estimar la probabilidad.")
        numeric_threshold = _finite_number(threshold, "threshold")
        return float(np.mean(values > numeric_threshold))

    def quantile(self, probability: float) -> Mapping[sym.Symbol, np.ndarray]:
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability debe estar entre 0 y 1.")
        return {
            symbol: np.quantile(values, probability, axis=0)
            for symbol, values in self.outputs.items()
        }

# ===========================
# 
# ===========================

class SypronStateModel:
    """Probabilistic layer composed with one existing Syphen SystemState."""

    def __init__(self, state: SystemState, seed: Optional[int] = None, generator: Optional[GeneradorNumerico] = None,
    ) -> None:
        
        if not isinstance(state, SystemState):
            raise TypeError("state debe ser un SystemState de Syphen.")
        if generator is not None and not isinstance(generator, GeneradorNumerico):
            raise TypeError("generator debe ser un GeneradorNumerico.")
        self.state = state
        self.generator = generator or GeneradorNumerico(seed)

    @property
    def random_variables(self) -> tuple[RandomVariableElement, ...]:
        return tuple(
            element
            for element in self.state.elements.values()
            if isinstance(element, RandomVariableElement)
        )

    @property
    def stochastic_functions(self) -> tuple[StochasticFunctionElement, ...]:
        return tuple(
            element
            for element in self.state.elements.values()
            if isinstance(element, StochasticFunctionElement)
        )

    def draw(self) -> StateRealization:
        values = {
            element.identificator.id: self.generator.sample(element.distribution)
            for element in self.random_variables
        }
        self.state.validate_values(values)
        functions = {
            element.identificator.id: element.realize(self.generator)
            for element in self.stochastic_functions
        }
        return StateRealization(
            state_id=self.state.identificator.id,
            values=values,
            functions=functions,
        )

    def evaluate(self, realization: Optional[StateRealization] = None,) -> dict[sym.Symbol, np.ndarray]:
        
        scenario = realization or self.draw()
        if scenario.state_id != self.state.identificator.id:
            raise ValueError("La realización pertenece a otro SystemState.")
        values_by_name: dict[str, float | np.ndarray] = {
            element.name: scenario.values[element.identificator.id]
            for element in self.random_variables
        }
        return self.state.evaluate(custom_values=values_by_name)

    def evaluate_function(
        self,
        element_id: UUID,
        *arguments: float | complex | np.ndarray,
        realization: Optional[StateRealization] = None,
    ) -> float | complex | np.ndarray:
        scenario = realization or self.draw()
        if scenario.state_id != self.state.identificator.id:
            raise ValueError("La realización pertenece a otro SystemState.")
        function_realization = scenario.functions.get(element_id)
        if function_realization is None:
            raise KeyError("La función no pertenece a esta realización.")
        return function_realization.evaluate(*arguments)

    def sample(self) -> StateRealization:
        """Draw and apply one realization to the Syphen state."""
        realization = self.draw()
        self.state.set_values(dict(realization.values))
        return realization

    def simulate(self, trials: int) -> StateSimulationResult:
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
            raise ValueError("trials debe ser un entero positivo.")
        realizations = [self.draw() for _ in range(trials)]
        variables = self.random_variables
        variable_samples = np.asarray(
            [
                [realization.values[variable.identificator.id] for variable in variables]
                for realization in realizations
            ],
            dtype=float,
        ).reshape(trials, len(variables))
        evaluated = [self.evaluate(realization) for realization in realizations]
        output_symbols = tuple(evaluated[0])
        outputs = {
            symbol: np.stack([result[symbol] for result in evaluated], axis=0)
            for symbol in output_symbols
        }
        return StateSimulationResult(
            variable_ids=tuple(
                variable.identificator.id for variable in variables
            ),
            variable_names=tuple(variable.name for variable in variables),
            variable_samples=variable_samples,
            outputs=outputs,
        )
