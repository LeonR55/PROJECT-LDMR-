# ==================================
# SYSTEMS PHYSICS ENGINE (SYPHEN)
# ==================================

""" Syphen, stands for Systemic Physics Engine, it´s functionality is to model systems through Topologies, 
States, Processes, Inputs and Outputs Processing and Semantics Domain powered by a set of specialized python libraries
 """

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from fractions import Fraction
from math import isfinite
from typing import ClassVar, Any, Optional, Callable, Optional, Sequence, Union, Mapping
from abc import ABC, abstractmethod
from uuid import uuid4, UUID

import networkx as nx
import numpy as np
import sympy as sym 
import scipy as sci
import simpy as sim

import matplotlib.pyplot as plt

# =================================
# ONTOLOGY BASE
# =================================

@dataclass
class Ontology:

    class EntityType(Enum):

        NOT_DEFINED = auto()

        ELEMENT = auto()
        FUNCTION_ELEMENT = auto()
        SYSTEM_STATE = auto()

        PROCESS = auto()
        PROCESS_VALUE = auto()
        PROCESS_EXPRESSION = auto()
        PROCESS_CONDITION = auto()
        PROCESS_TRIGGER = auto()
        PROCESS_ACTION = auto()

        SYSTEM_SEMANTICS = auto()
        SYSTEM_STRUCTURE = auto()
        SYSTEM_RELATION = auto()
        STATE_COUPLING_STRUCTURE = auto()
        STATE_COUPLING_RELATION = auto()
        FLOW_BUS_STRUCTURE = auto()
        FLOW_BUS_RELATION = auto()

        TOKEN = auto()
        PORT = auto()
        SYSTEM_PROCESSOR = auto()
                
        COMPONENT = auto()
        
        SYSTEM = auto()
        
        SPACE = auto()
    
@dataclass
class Identification:

    id : UUID = field(init = False, default_factory = uuid4)
    entity : Optional[Ontology.EntityType] = None
    label : Optional[str] = None
    
class Entity(ABC):

    identificator : Identification

    def __init__(self, systemic_space : SystemicSpace) -> None:

        self.systemic_space : SystemicSpace = systemic_space
        
        self.semantic_archetype_id: Optional[UUID] = None
        self.semantic_domain_id: Optional[UUID] = None
        self.semantic_attributes: dict[str, Any] = {}
        self.semantic_tags: set[str] = set()
        
    def space_auto_registry(self) -> None:
        self.systemic_space.create(self.identificator.id, self)

# =================================
# SYSTEMS STATE
# =================================

@dataclass(frozen=True)
class ValueConstraints:
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    predicates: tuple[tuple[str, Callable[[float], bool]], ...] = ()

    def __post_init__(self) -> None:
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("El límite mínimo no puede superar al máximo.")

    def validate(
        self,
        value: Union[float, np.ndarray],
        subject: str,
    ) -> Union[float, np.ndarray]:
        try:
            numeric_value = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{subject} debe ser numérico.") from error

        if not np.all(np.isfinite(numeric_value)):
            raise ValueError(f"{subject} debe contener solo valores finitos.")
        if numeric_value.size == 0:
            raise ValueError(f"{subject} no puede estar vacío.")
        if self.minimum is not None and np.any(numeric_value < self.minimum):
            raise ValueError(f"{subject} no puede ser menor que {self.minimum}.")
        if self.maximum is not None and np.any(numeric_value > self.maximum):
            raise ValueError(f"{subject} no puede superar {self.maximum}.")

        for predicate_name, predicate in self.predicates:
            if not all(bool(predicate(float(item))) for item in numeric_value.flat):
                raise ValueError(
                    f"{subject} no cumple la restricción '{predicate_name}'."
                )

        if numeric_value.ndim == 0:
            return float(numeric_value)
        return numeric_value.copy()

class StateElement(Entity):

    def __init__(self, systemic_space : SystemicSpace) -> None:
        super().__init__(systemic_space = systemic_space)
        self.identificator : Identification = Identification(entity = Ontology.EntityType.ELEMENT)
        self._owner_state_id: Optional[UUID] = None
        self.space_auto_registry()

class VariableElement(StateElement):

    def __init__(self, systemic_space: SystemicSpace, name: str, initial_value: Optional[float] = None, 
        constraints: Optional[ValueConstraints] = None,) -> None:
        
        self.constraints = constraints if constraints is not None else ValueConstraints()
        validated_value = (
            None
            if initial_value is None
            else float(self.constraints.validate(initial_value, f"Variable '{name}'"))
        )
        super().__init__(systemic_space = systemic_space)
        self.name = name
        self.symbol : sym.Symbol = sym.Dummy(name = name)
        self._value: Optional[float] = validated_value

    @property
    def value(self) -> Optional[float]:
        raise AttributeError("Lee o modifica el valor mediante el SystemState propietario."
)


DiscreteValue = Union[str, int, bool, Enum]
StateValue = Union[float, DiscreteValue]


class DiscreteStateElement(StateElement):
    def __init__(
        self,
        systemic_space: SystemicSpace,
        name: str,
        initial_value: Optional[DiscreteValue] = None,
        allowed_values: Optional[Sequence[DiscreteValue]] = None,
    ) -> None:
        if not name:
            raise ValueError("name no puede estar vacío.")
        values = None if allowed_values is None else tuple(allowed_values)
        if values is not None:
            if not values:
                raise ValueError("allowed_values no puede estar vacío.")
            if len(set(values)) != len(values):
                raise ValueError("allowed_values no puede contener duplicados.")
            for value in values:
                self._validate_type(value)
        super().__init__(systemic_space)
        self.name = name
        self.allowed_values = values
        self._value = (
            None if initial_value is None else self.validate_value(initial_value)
        )

    @staticmethod
    def _validate_type(value: Any) -> None:
        if not isinstance(value, (str, int, bool, Enum)):
            raise TypeError("Un valor discreto debe ser str, int, bool o Enum.")

    def validate_value(self, value: DiscreteValue) -> DiscreteValue:
        self._validate_type(value)
        if self.allowed_values is not None and value not in self.allowed_values:
            raise ValueError(f"El valor discreto '{value}' no está permitido.")
        return value

class DependentExpressionElement(StateElement):

    def __init__(self, systemic_space: SystemicSpace, name: str, expression: sym.Expr, constraints: Optional[ValueConstraints] = None,) -> None:
        
        super().__init__(systemic_space = systemic_space)
        self.name : str = name
        self.symbol : sym.Symbol = sym.Dummy(name = name)
        self.expression : sym.Expr = expression
        self.constraints = constraints if constraints is not None else ValueConstraints()

class FunctionElement(StateElement):

    def __init__(self, systemic_space: SystemicSpace, name: str, arguments: Sequence[sym.Symbol], expression: sym.Expr,
                 ) -> None:
        if not name:
            raise ValueError("FunctionElement requiere un nombre no vacío.")
        if not isinstance(expression, sym.Expr):
            raise TypeError("expression debe ser una expresión simbólica de SymPy.")
        function_arguments = tuple(arguments)
        if not function_arguments:
            raise ValueError("FunctionElement requiere al menos un argumento.")
        if any(not isinstance(argument, sym.Symbol) for argument in function_arguments):
            raise TypeError("Cada argumento de la función debe ser un símbolo de SymPy.")
        if len(set(function_arguments)) != len(function_arguments):
            raise ValueError("Los argumentos de la función deben ser únicos.")
        unbound_symbols = expression.free_symbols - set(function_arguments)
        if unbound_symbols:
            names = ", ".join(sorted(str(symbol) for symbol in unbound_symbols))
            raise ValueError(
                f"La función contiene símbolos que no son argumentos: {names}."
            )

        super().__init__(systemic_space=systemic_space)
        self.identificator.entity = Ontology.EntityType.FUNCTION_ELEMENT
        self.identificator.label = name
        self.name = name
        self.arguments = function_arguments
        self.expression = expression
        self.function = sym.Lambda(self.arguments, self.expression)
        self._compiled_function = sym.lambdify(
            self.arguments,
            self.expression,
            modules=["numpy", "scipy"],
        )
        self.space_auto_registry()

    def evaluate(self, *arguments: float | complex | np.ndarray) -> float | complex | np.ndarray:
        if len(arguments) != len(self.arguments):
            raise ValueError(
                f"La función '{self.name}' requiere {len(self.arguments)} argumentos."
            )
        numeric_arguments = []
        for argument in arguments:
            value = np.asarray(argument)
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError("Los argumentos de la función deben ser numéricos.")
            if not np.all(np.isfinite(value)):
                raise ValueError("Los argumentos de la función deben ser finitos.")
            numeric_arguments.append(value.item() if value.ndim == 0 else value)

        result = np.asarray(self._compiled_function(*numeric_arguments))
        if not np.issubdtype(result.dtype, np.number):
            raise TypeError("La función debe producir un resultado numérico.")
        if not np.all(np.isfinite(result)):
            raise ValueError("La función produjo resultados no finitos.")
        if result.ndim == 0:
            return result.item()
        return result.copy()

class CompiledStateKernel:

    def __init__(self, input_symbols: list[sym.Symbol], target_symbols: list[sym.Symbol], compiled_symbols: list[sym.Symbol], compiled_function: Callable, 
    output_constraints: dict[sym.Symbol, ValueConstraints],):
        
        self.input_symbols: list[sym.Symbol] = input_symbols
        self.target_symbols: list[sym.Symbol] = target_symbols
        self.compiled_symbols: list[sym.Symbol] = compiled_symbols
        self._execute_num: Callable = compiled_function
        self._output_constraints = output_constraints

    def evaluate_to_dict(self, *args) -> dict[sym.Symbol, np.ndarray]:
        try:
            with np.errstate(invalid="raise", divide="raise", over="raise"):
                raw_outputs = self._execute_num(*args)
        except (ArithmeticError, ValueError, FloatingPointError) as error:
            raise ValueError(
                "La evaluación encontró un valor fuera del dominio numérico."
            ) from error
        if len(self.compiled_symbols) == 1:
            raw_outputs = [raw_outputs]
        if len(raw_outputs) != len(self.compiled_symbols):
            raise RuntimeError(
                "El kernel compilado devolvió una cantidad inesperada de resultados."
            )
        evaluated = {
            symbol: np.asarray(result)
            for symbol, result in zip(self.compiled_symbols, raw_outputs)
        }
        for symbol, constraints in self._output_constraints.items():
            constraints.validate(evaluated[symbol], f"Resultado '{symbol.name}'")
        return {symbol: evaluated[symbol] for symbol in self.target_symbols}

class SystemState(Entity):

    def __init__(self, systemic_space : SystemicSpace) -> None:
        super().__init__(systemic_space = systemic_space)
        self.identificator : Identification = Identification(entity = Ontology.EntityType.SYSTEM_STATE)
        self.space_auto_registry()

        self.elements : dict[UUID, StateElement] = {}
        self._variables_by_name : dict[str, VariableElement] = {}
        self._discrete_variables_by_name: dict[str, DiscreteStateElement] = {}
        self._expressions_by_symbol : dict[sym.Symbol, sym.Expr] = {}

        self.expressions : list[DependentExpressionElement] = []
        self.symbols : list[sym.Symbol] = []
        self._active_kernel: Optional[CompiledStateKernel] = None
        self._history_recorders: list[
            tuple[TemporalHistory, Callable[[], float]]
        ] = []

    def add_element(self, element : UUID) -> None:
        entity_element = self.systemic_space.get(element)

        if not isinstance(entity_element, StateElement):
            raise KeyError(f"El ID {element} no corresponde a un elemento de estado.")
        if entity_element._owner_state_id not in (None, self.identificator.id):
            raise ValueError("El elemento ya pertenece a otro SystemState.")

        if isinstance(entity_element, VariableElement):
            existing_variable = self._variables_by_name.get(entity_element.name)
            if existing_variable is not None and existing_variable is not entity_element:
                raise ValueError(f"Ya existe una variable llamada '{entity_element.name}' en este estado.")
            if entity_element.name in self._discrete_variables_by_name:
                raise ValueError(f"Ya existe un elemento llamado '{entity_element.name}' en este estado.")
        if isinstance(entity_element, DiscreteStateElement):
            existing_discrete = self._discrete_variables_by_name.get(
                entity_element.name
            )
            if existing_discrete is not None and existing_discrete is not entity_element:
                raise ValueError(f"Ya existe un elemento llamado '{entity_element.name}' en este estado.")
            if entity_element.name in self._variables_by_name:
                raise ValueError(f"Ya existe una variable llamada '{entity_element.name}' en este estado.")

        entity_element._owner_state_id = self.identificator.id
        self.elements[element] = entity_element
        if isinstance(entity_element, VariableElement):
            self._variables_by_name[entity_element.name] = entity_element
            self._active_kernel = None

        elif isinstance(entity_element, DiscreteStateElement):
            self._discrete_variables_by_name[entity_element.name] = entity_element

        elif isinstance(entity_element, DependentExpressionElement):
            if entity_element not in self.expressions:
                self.expressions.append(entity_element)
            self._expressions_by_symbol[entity_element.symbol] = entity_element.expression
            self._active_kernel = None

    def read_value(self, element: UUID) -> float:
        entity_element = self.elements.get(element)
        if not isinstance(entity_element, VariableElement):
            raise KeyError("Solo se puede leer una variable que pertenezca a este estado.")
        if entity_element._value is None:
            raise ValueError(f"La variable '{entity_element.name}' no tiene valor asignado.")
        return float(
            entity_element.constraints.validate(
                entity_element._value,
                f"Variable '{entity_element.name}'",
            )
        )

    def read_discrete_value(self, element: UUID) -> DiscreteValue:
        entity_element = self.elements.get(element)
        if not isinstance(entity_element, DiscreteStateElement):
            raise KeyError(
                "Solo se puede leer un DiscreteStateElement de este estado."
            )
        if entity_element._value is None:
            raise ValueError(
                f"El elemento discreto '{entity_element.name}' no tiene valor asignado."
            )
        return entity_element.validate_value(entity_element._value)

    def set_discrete_value(self, element: UUID, value: DiscreteValue) -> None:
        entity_element = self.elements.get(element)
        if not isinstance(entity_element, DiscreteStateElement):
            raise KeyError(
                "Solo se puede asignar valor a un DiscreteStateElement de este estado."
            )
        validated_value = entity_element.validate_value(value)
        self._commit_state_updates({element: validated_value})

    def read_element(
        self,
        element: UUID,
    ) -> Union[float, np.ndarray, sym.Lambda, DiscreteValue]:
        entity_element = self.elements.get(element)
        if isinstance(entity_element, VariableElement):
            return self.read_value(element)
        if isinstance(entity_element, DiscreteStateElement):
            return self.read_discrete_value(element)
        if isinstance(entity_element, DependentExpressionElement):
            kernel = self._active_kernel
            if kernel is None or entity_element.symbol not in kernel.target_symbols:
                raise RuntimeError(
                    f"La expresión '{entity_element.name}' debe incluirse en "
                    "compile_state() antes de consultarla."
                )
            return self.evaluate()[entity_element.symbol].copy()
        if isinstance(entity_element, FunctionElement):
            return entity_element.function
        raise KeyError("El elemento no es una variable o expresión de este estado.")

    def evaluate_function(
        self,
        element: UUID,
        *arguments: float | complex | np.ndarray,
    ) -> float | complex | np.ndarray:
        entity_element = self.elements.get(element)
        if not isinstance(entity_element, FunctionElement):
            raise KeyError("El elemento solicitado no es una función de este estado.")
        return entity_element.evaluate(*arguments)

    def set_value(self, element: UUID, value: float) -> None:
        self.set_values({element: value})

    def validate_values(self, updates: dict[UUID, float]) -> dict[UUID, float]:
        validated_updates: dict[UUID, float] = {}
        candidate_values: dict[str, float] = {}
        for element, value in updates.items():
            entity_element = self.elements.get(element)
            if not isinstance(entity_element, VariableElement):
                raise KeyError("Solo se puede asignar valor a variables de este estado.")
            validated_value = entity_element.constraints.validate(
                value,
                f"Variable '{entity_element.name}'",
            )
            if not isinstance(validated_value, float):
                raise TypeError(
                    "Una VariableElement solo puede almacenar un valor escalar."
                )
            validated_updates[element] = validated_value
            candidate_values[entity_element.name] = validated_value

        if (
            self._active_kernel is not None
            and all(
                variable._value is not None or variable.name in candidate_values
                for variable in self._variables_by_name.values()
            )
        ):
            self.evaluate(custom_values = candidate_values)
        return validated_updates

    def validate_state_updates(
        self,
        updates: Mapping[UUID, StateValue],
    ) -> dict[UUID, StateValue]:
        numeric_updates: dict[UUID, float] = {}
        discrete_updates: dict[UUID, DiscreteValue] = {}
        for element_id, value in updates.items():
            element = self.elements.get(element_id)
            if isinstance(element, VariableElement):
                if isinstance(value, (str, bool, Enum)) or not isinstance(
                    value, (int, float, np.number)
                ):
                    raise TypeError("Una VariableElement requiere un valor numérico.")
                numeric_updates[element_id] = float(value)
            elif isinstance(element, DiscreteStateElement):
                if not isinstance(value, (str, int, bool, Enum)):
                    raise TypeError(
                        "Un DiscreteStateElement requiere un valor categórico."
                    )
                discrete_updates[element_id] = value
            else:
                raise KeyError(
                    "Cada elemento debe pertenecer al estado y ser asignable."
                )
        validated: dict[UUID, StateValue] = dict(self.validate_values(numeric_updates))
        for element_id, value in discrete_updates.items():
            element = self.elements[element_id]
            if not isinstance(element, DiscreteStateElement):
                raise TypeError("El elemento discreto dejó de pertenecer al estado.")
            validated[element_id] = element.validate_value(value)
        return validated

    def set_values(self, updates: dict[UUID, float]) -> None:
        validated_updates = self.validate_values(updates)
        self._commit_validated_values(validated_updates)

    def _commit_state_updates(
        self,
        updates: Mapping[UUID, StateValue],
        record_history: bool = True,
    ) -> None:
        validated_updates = self.validate_state_updates(updates)
        discrete_updates: dict[UUID, DiscreteValue] = {}
        numeric_updates: dict[UUID, float] = {}
        for element_id, value in validated_updates.items():
            element = self.elements[element_id]
            if isinstance(element, VariableElement):
                if not isinstance(value, float):
                    raise TypeError("Una VariableElement requiere un valor numérico.")
                numeric_updates[element_id] = value
            elif isinstance(element, DiscreteStateElement):
                if not isinstance(value, (str, int, bool, Enum)):
                    raise TypeError("Un DiscreteStateElement requiere un valor categórico.")
                discrete_updates[element_id] = value
            else:
                raise TypeError("El elemento no es una variable asignable.")
        self._commit_validated_values(numeric_updates, record_history=False)
        for element_id, value in discrete_updates.items():
            element = self.elements[element_id]
            if not isinstance(element, DiscreteStateElement):
                raise TypeError("El elemento discreto dejó de pertenecer al estado.")
            element._value = value
        if record_history:
            for history, time_provider in tuple(self._history_recorders):
                now = time_provider()
                for element_id, value in discrete_updates.items():
                    history.record_discrete(
                        self.identificator.id,
                        element_id,
                        value,
                        now,
                    )
                history.record_state(self, now)

    def _commit_validated_values(
        self,
        updates: dict[UUID, float],
        record_history: bool = True,
    ) -> None:
        for element, value in updates.items():
            variable = self.elements[element]
            if not isinstance(variable, VariableElement):
                raise TypeError("La variable validada dejó de pertenecer al estado.")
            variable._value = value
        if record_history:
            for history, time_provider in tuple(self._history_recorders):
                history.record_state(self, time_provider())

    def compile_state(self, targets: dict[sym.Symbol, sym.Expr], custom_mappings: Optional[dict[str, Callable[..., Any]]] = None) -> CompiledStateKernel:
        """
        Compila el mapa de expresiones algebraicas. Admite un diccionario 'custom_mappings'
        para enlazar funciones u operadores avanzados de NumPy/SciPy que SymPy deba traducir.
        """
        self.symbols = [var.symbol for var in self._variables_by_name.values()]
        substitutions = {sym_key: expr_val for sym_key, expr_val in self._expressions_by_symbol.items()}
        expanded_substitutions: dict[sym.Symbol, sym.Expr] = {}

        def resolve_expression(symbol: sym.Symbol, resolving: set[sym.Symbol]) -> sym.Expr:
            if symbol in expanded_substitutions:
                return expanded_substitutions[symbol]
            if symbol in resolving:
                raise ValueError(f"Dependencia circular detectada en la expresión '{symbol}'.")

            expression = substitutions[symbol]
            dependencies = {
                dependency: resolve_expression(dependency, resolving | {symbol})
                for dependency in expression.free_symbols
                if isinstance(dependency, sym.Symbol) and dependency in substitutions
            }
            resolved_expression = expression.xreplace(dependencies)
            expanded_substitutions[symbol] = resolved_expression
            return resolved_expression

        for symbol in substitutions:
            resolve_expression(symbol, set())
        
        if not targets:
            raise ValueError("Se requiere al menos un objetivo para compilar el estado.")

        target_symbols = list(targets.keys())
        raw_expressions = list(targets.values())
        flat_expressions = [expr.xreplace(expanded_substitutions) for expr in raw_expressions]

        variables_by_symbol = {
            variable.symbol: variable for variable in self._variables_by_name.values()
        }
        expressions_by_symbol = {
            expression.symbol: expression for expression in self.expressions
        }
        owned_symbols = set(variables_by_symbol) | set(expressions_by_symbol)
        owned_variable_symbols = set(variables_by_symbol)
        expressions_to_validate = dict(expanded_substitutions)
        for target_symbol, expression in zip(target_symbols, flat_expressions):
            if target_symbol not in owned_symbols:
                raise ValueError(
                    f"El objetivo '{target_symbol}' no pertenece a este SystemState."
                )
            if target_symbol in expressions_by_symbol:
                expressions_to_validate[target_symbol] = expression

        for expression_symbol, expression in expressions_to_validate.items():
            foreign_symbols = expression.free_symbols - owned_variable_symbols
            if foreign_symbols:
                names = ", ".join(sorted(str(symbol) for symbol in foreign_symbols))
                raise ValueError(
                    f"La expresión '{expression_symbol}' depende de símbolos ajenos "
                    f"al estado: {names}. Usa StateTransducer para consultarlos."
                )

        compiled_symbols = list(target_symbols)
        compiled_expressions = list(flat_expressions)
        output_constraints: dict[sym.Symbol, ValueConstraints] = {}
        for symbol in target_symbols:
            target_element = expressions_by_symbol.get(symbol)
            if target_element is None:
                target_element = variables_by_symbol[symbol]
            output_constraints[symbol] = target_element.constraints

        for expression_symbol, expression in expressions_to_validate.items():
            if expression_symbol not in output_constraints:
                compiled_symbols.append(expression_symbol)
                compiled_expressions.append(expression)
                output_constraints[expression_symbol] = (
                    expressions_by_symbol[expression_symbol].constraints
                )

        # EXTENSIÓN: Construcción dinámica de la cadena de traducción de backend
        # Inyecta diccionarios de usuario sobre los backends estándar de cómputo científico

        compile_modules: list[Union[str, dict[str, Callable[..., Any]]]] = ['numpy', 'scipy']

        if custom_mappings: compile_modules.insert(0, custom_mappings)
        compiled_fn = sym.lambdify(
            args=self.symbols,
            expr=compiled_expressions,
            modules=compile_modules,
        )
        self._active_kernel = CompiledStateKernel(input_symbols=self.symbols, target_symbols=target_symbols, compiled_symbols=compiled_symbols, 
        compiled_function=compiled_fn, output_constraints=output_constraints,)

        return self._active_kernel

    def evaluate(self, custom_values: Optional[dict[str, Union[float, np.ndarray]]] = None) -> dict[sym.Symbol, np.ndarray]:
        if self._active_kernel is None:
            raise RuntimeError("Operación Inválida: Se debe invocar 'compile_state()' antes de evaluar el sistema.")

        custom_values = custom_values or {}
        unknown_names = custom_values.keys() - self._variables_by_name.keys()
        if unknown_names:
            names = ", ".join(sorted(unknown_names))
            raise KeyError(
                f"Las variables {names} no pertenecen a este estado. "
                "Usa StateTransducer para consultarlas."
            )

        input_args = []
        
        for sym_in in self.symbols:
            name = sym_in.name
            variable = self._variables_by_name[name]
            if name in custom_values:
                value = custom_values[name]
            else:
                if variable._value is None:
                    raise ValueError(f"Falta de Datos: La variable '{name}' no posee un valor asignado.")
                value = variable._value
            input_args.append(
                variable.constraints.validate(value, f"Variable '{name}'")
            )

        return self._active_kernel.evaluate_to_dict(*input_args)

    # EXTENSIÓN: Gestión de memoria y persistencia del Estado (Consolidación Temporal)
    def update_state_values(self, evaluation_results: dict[sym.Symbol, np.ndarray]) -> None:
        """
        Toma los mapas numéricos devueltos por .evaluate() y los escribe de regreso
        en las VariableElement independientes del sistema, actualizando su memoria histórica.
        """
        variables_by_symbol = {
            variable.symbol: variable for variable in self._variables_by_name.values()
        }
        updates: dict[VariableElement, float] = {}
        for sym_target, num_value in evaluation_results.items():
            variable = variables_by_symbol.get(sym_target)
            if variable is None:
                if sym_target in self._expressions_by_symbol:
                    continue
                raise KeyError(
                    f"El resultado '{sym_target}' no pertenece a este SystemState."
                )
            # For vector simulations, persist the last continuous state.
            value = float(num_value[-1]) if num_value.ndim > 0 else float(num_value)
            validated_value = variable.constraints.validate(
                value,
                f"Variable '{variable.name}'",
            )
            if not isinstance(validated_value, float):
                raise TypeError("Una VariableElement solo puede almacenar un valor escalar.")
            updates[variable] = validated_value

        candidate_values = {
            variable.name: value for variable, value in updates.items()
        }
        if (
            self._active_kernel is not None
            and all(
                variable._value is not None or variable in updates
                for variable in self._variables_by_name.values()
            )
        ):
            self.evaluate(custom_values = candidate_values)
        for variable, value in updates.items():
            variable._value = value

# =============================
# SYSTEMS TEMPORAL MANAGEMENT
# =============================

@dataclass(frozen=True)
class TrajectorySample:
    simulation_time: float
    state_id: UUID
    values: Mapping[UUID, Union[float, DiscreteValue]]

@dataclass(frozen=True)
class DiscreteStateEvent:
    simulation_time: float
    state_id: UUID
    element_id: UUID
    value: Union[str, int, bool, Enum]

class TemporalHistory:
    """In-memory numeric trajectories and explicitly recorded discrete changes."""

    def __init__(self) -> None:
        self.samples: list[TrajectorySample] = []
        self.discrete_events: list[DiscreteStateEvent] = []
        self._attached_states: set[UUID] = set()

    def attach_state(
        self,
        state: SystemState,
        time_provider: Callable[[], float],
    ) -> None:
        if not isinstance(state, SystemState):
            raise TypeError("state debe ser SystemState.")
        if not callable(time_provider):
            raise TypeError("time_provider debe ser invocable.")
        if state.identificator.id in self._attached_states:
            return
        state._history_recorders.append((self, time_provider))
        self._attached_states.add(state.identificator.id)
        self.record_state(state, time_provider())

    def record_state(
        self,
        state: SystemState,
        simulation_time: float,
    ) -> TrajectorySample:
        values: dict[UUID, Union[float, DiscreteValue]] = {
            element_id: float(element._value)
            for element_id, element in state.elements.items()
            if isinstance(element, VariableElement) and element._value is not None
        }
        values.update(
            {
                element_id: element._value
                for element_id, element in state.elements.items()
                if isinstance(element, DiscreteStateElement)
                and element._value is not None
            }
        )
        return self.record_values(state.identificator.id, simulation_time, values)

    def record_values(
        self,
        state_id: UUID,
        simulation_time: float,
        values: Mapping[UUID, Union[float, DiscreteValue]],
    ) -> TrajectorySample:
        time_value = float(simulation_time)
        if not isfinite(time_value):
            raise ValueError("simulation_time debe ser finito.")
        validated_values: dict[UUID, Union[float, DiscreteValue]] = {}
        for element_id, value in values.items():
            if isinstance(value, (str, bool, Enum)):
                validated_values[element_id] = value
            elif isinstance(value, int):
                validated_values[element_id] = value
            else:
                numeric_value = float(value)
                if not isfinite(numeric_value):
                    raise ValueError("Los valores de trayectoria deben ser finitos.")
                validated_values[element_id] = numeric_value
        sample = TrajectorySample(time_value, state_id, validated_values)
        self.samples.append(sample)
        return sample

    def record_discrete(
        self,
        state_id: UUID,
        element_id: UUID,
        value: DiscreteValue,
        simulation_time: float,
    ) -> DiscreteStateEvent:
        DiscreteStateElement._validate_type(value)
        time_value = float(simulation_time)
        if not isfinite(time_value):
            raise ValueError("simulation_time debe ser finito.")
        event = DiscreteStateEvent(time_value, state_id, element_id, value)
        self.discrete_events.append(event)
        return event

    def trajectory(self, state_id: UUID, element_id: UUID) -> ODETrajectory:
        samples = [
            (sample.simulation_time, sample.values[element_id])
            for sample in self.samples
            if sample.state_id == state_id and element_id in sample.values
        ]
        if not samples:
            raise KeyError("No hay muestras para el estado y elemento indicados.")
        if any(
            isinstance(value, (bool, Enum))
            or not isinstance(value, (int, float))
            for _, value in samples
        ):
            raise TypeError("La trayectoria solicitada no es numérica.")
        times = np.asarray([time_value for time_value, _ in samples], dtype=float)
        values = np.asarray([value for _, value in samples], dtype=float)
        return ODETrajectory(times, {element_id: values})

@dataclass(frozen=True)
class SimulationClock:
    """Maps simulation seconds to an aware calendar origin."""

    origin: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.origin, datetime):
            raise TypeError("origin debe ser datetime.")
        if self.origin.tzinfo is None or self.origin.utcoffset() is None:
            raise ValueError("origin debe incluir una zona horaria.")
        round_trip = self.origin.astimezone(timezone.utc).astimezone(
            self.origin.tzinfo
        )
        if round_trip.replace(tzinfo=None) != self.origin.replace(tzinfo=None):
            raise ValueError("origin representa una hora local inexistente.")

    def to_datetime(self, simulation_time: float) -> datetime:
        time_value = float(simulation_time)
        if not isfinite(time_value):
            raise ValueError("simulation_time debe ser finito.")
        instant = self.origin.astimezone(timezone.utc) + timedelta(
            seconds=time_value
        )
        return instant.astimezone(self.origin.tzinfo)

    def to_simulation_time(self, instant: datetime) -> float:
        if not isinstance(instant, datetime):
            raise TypeError("instant debe ser datetime.")
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("instant debe incluir una zona horaria.")
        round_trip = instant.astimezone(timezone.utc).astimezone(instant.tzinfo)
        if round_trip.replace(tzinfo=None) != instant.replace(tzinfo=None):
            raise ValueError("instant representa una hora local inexistente.")
        return (
            instant.astimezone(timezone.utc)
            - self.origin.astimezone(timezone.utc)
        ).total_seconds()

@dataclass(frozen=True)
class ODETrajectory:
    times: np.ndarray
    values: dict[UUID, np.ndarray]

class ODEIntegrator:
    """Integrate variables across states; derivative-vector order follows variable_ids."""

    def __init__(
        self,
        state: Union[SystemState, Sequence[SystemState]],
        variable_ids: Sequence[UUID],
        derivatives: Callable[
            [float, np.ndarray],
            Union[Sequence[float], np.ndarray],
        ],
    ) -> None:
        states = (state,) if isinstance(state, SystemState) else tuple(state)
        if not states or any(not isinstance(candidate, SystemState) for candidate in states):
            raise TypeError("state debe ser SystemState o una secuencia no vacía de estados.")
        if len({candidate.identificator.id for candidate in states}) != len(states):
            raise ValueError("La secuencia de estados no puede contener duplicados.")
        if any(
            candidate.systemic_space is not states[0].systemic_space
            for candidate in states
        ):
            raise ValueError("Los estados ODE deben pertenecer al mismo SystemicSpace.")
        variables = tuple(variable_ids)
        if not variables or len(set(variables)) != len(variables):
            raise ValueError("variable_ids debe contener variables únicas y no vacías.")
        if not callable(derivatives):
            raise TypeError("derivatives debe ser invocable.")
        state_by_element = {
            element_id: candidate
            for candidate in states
            for element_id in candidate.elements
        }
        for element_id in variables:
            owner = state_by_element.get(element_id)
            if owner is None or not isinstance(owner.elements.get(element_id), VariableElement):
                raise KeyError("Cada variable debe pertenecer a uno de los estados indicados.")
        self.states = states
        self.state = states[0]
        self._state_by_element = state_by_element
        self.variable_ids = variables
        self.derivatives = derivatives

    def integrate(
        self,
        t_span: tuple[float, float],
        t_eval: Optional[Sequence[float]] = None,
        method: str = "RK45",
        rtol: float = 1e-3,
        atol: float = 1e-6,
        commit: bool = True,
    ) -> ODETrajectory:
        """Integrate over seconds and commit only after the whole path validates."""
        if len(t_span) != 2:
            raise ValueError("t_span debe contener inicio y fin.")
        start, end = (float(value) for value in t_span)
        if not isfinite(start) or not isfinite(end) or end <= start:
            raise ValueError("t_span requiere tiempos finitos con inicio < fin.")
        if not isfinite(rtol) or rtol <= 0 or not isfinite(atol) or atol <= 0:
            raise ValueError("rtol y atol deben ser finitos y positivos.")
        evaluation_times = None if t_eval is None else np.asarray(t_eval, dtype=float)
        if evaluation_times is not None and (
            evaluation_times.ndim != 1
            or len(evaluation_times) == 0
            or not np.all(np.isfinite(evaluation_times))
            or np.any(evaluation_times < start)
            or np.any(evaluation_times > end)
            or np.any(np.diff(evaluation_times) <= 0)
        ):
            raise ValueError("t_eval debe ser una secuencia creciente dentro de t_span.")
        if (
            commit
            and evaluation_times is not None
            and evaluation_times[-1] < end
        ):
            evaluation_times = np.append(evaluation_times, end)

        initial_values = np.asarray(
            [
                self._state_by_element[element_id].read_value(element_id)
                for element_id in self.variable_ids
            ],
            dtype=float,
        )

        def evaluate_derivatives(time_value: float, values: np.ndarray) -> np.ndarray:
            result = np.asarray(self.derivatives(time_value, values.copy()), dtype=float)
            if result.shape != initial_values.shape:
                raise ValueError("derivatives debe devolver un valor por variable.")
            if not np.all(np.isfinite(result)):
                raise ValueError("derivatives produjo valores no finitos.")
            return result

        result = sci.integrate.solve_ivp(
            evaluate_derivatives,
            (start, end),
            initial_values,
            method=method,
            t_eval=evaluation_times,
            rtol=rtol,
            atol=atol,
        )
        if not result.success:
            raise RuntimeError(f"La integración ODE falló: {result.message}")
        values_by_id = {
            element_id: np.asarray(result.y[index], dtype=float).copy()
            for index, element_id in enumerate(self.variable_ids)
        }
        for element_id, values in values_by_id.items():
            variable = self._state_by_element[element_id].elements[element_id]
            if not isinstance(variable, VariableElement):
                raise TypeError("Una variable ODE dejó de pertenecer al estado.")
            variable.constraints.validate(values, f"Trayectoria '{variable.name}'")

        if commit:
            updates_by_state: dict[SystemState, dict[UUID, float]] = {}
            for element_id, values in values_by_id.items():
                updates_by_state.setdefault(
                    self._state_by_element[element_id],
                    {},
                )[element_id] = float(values[-1])
            validated_by_state = {
                state: state.validate_values(updates)
                for state, updates in updates_by_state.items()
            }
            for state, updates in validated_by_state.items():
                state._commit_validated_values(updates, record_history=False)
            for state in self.states:
                state_values = {
                    element_id: float(element._value)
                    for element_id, element in state.elements.items()
                    if isinstance(element, VariableElement)
                    and element._value is not None
                }
                for history, _ in tuple(state._history_recorders):
                    for index, time_value in enumerate(result.t):
                        sample_values = dict(state_values)
                        sample_values.update(
                            {
                                element_id: float(values[index])
                                for element_id, values in values_by_id.items()
                                if self._state_by_element[element_id] is state
                            }
                        )
                        history.record_values(
                            state.identificator.id,
                            float(time_value),
                            sample_values,
                        )

        return ODETrajectory(np.asarray(result.t, dtype=float).copy(), values_by_id)

class StateTransducer:
    """Bridge between two states with explicitly granted variable write access."""

    def __init__(self, consumer: SystemState, provider: SystemState, writable_elements: Optional[set[UUID]] = None,) -> None:
        
        if consumer is provider:
            raise ValueError("StateTransducer requiere dos SystemState distintos.")
        
        self._consumer = consumer
        self._provider = provider
        self._writable_elements = frozenset(writable_elements or ())
        for element_id in self._writable_elements:
            if not isinstance(provider.elements.get(element_id), VariableElement):
                raise ValueError(
                    "Los permisos de escritura solo pueden asignarse a variables "
                    "del estado proveedor."
                )

    def read(self, element: UUID) -> Union[float, np.ndarray, sym.Lambda]:
        entity_element = self._provider.elements.get(element)
        if entity_element is None:
            raise KeyError("El elemento solicitado no pertenece al estado proveedor.")

        if isinstance(entity_element, VariableElement):
            return self._provider.read_value(element)

        if isinstance(entity_element, FunctionElement):
            return entity_element.function

        if isinstance(entity_element, DependentExpressionElement):
            kernel = self._provider._active_kernel
            if kernel is None or entity_element.symbol not in kernel.target_symbols:
                raise RuntimeError(
                    f"La expresión '{entity_element.name}' debe incluirse en "
                    "compile_state() del estado proveedor antes de consultarla."
                )
            return self._provider.evaluate()[entity_element.symbol].copy()

        raise TypeError("El tipo de elemento solicitado no se puede leer.")

    def write(self, element: UUID, value: float) -> None:
        if element not in self._writable_elements:
            raise PermissionError(
                "El puente no tiene permiso de escritura para esta variable."
            )
        self._provider.set_value(element, value)

# =======================
# SYSTEMS PROCESS
# =======================

@dataclass
class TokenProcessingResult:
    token_updates: dict[UUID, float] = field(default_factory=dict)
    processor_updates: dict[UUID, float] = field(default_factory=dict)

class ProcessTriggerKind(Enum):
    MANUAL = auto()
    TOKEN_RECEIVED = auto()
    CONDITION = auto()
    INTERVAL = auto()
    CALENDAR = auto()

class ProcessValueScope(Enum):
    SYSTEM_STATE = auto()
    TOKEN_STATE = auto()
    SIMULATION_TIME = auto()

class ProcessExpressionOperator(Enum):
    CONSTANT = auto()
    DISCRETE_CONSTANT = auto()
    READ = auto()
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    NEGATE = auto()
    MIN = auto()
    MAX = auto()

class ProcessConditionOperator(Enum):

    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()

class ProcessActionKind(Enum):

    SET_STATE_VALUE = auto()
    SET_TOKEN_VALUE = auto()
    EMIT_TOKEN = auto()
    APPLY_STATE_COUPLING = auto()
    SET_COMPONENT_VALUE = auto()
    PROPAGATE_STATE_COUPLINGS = auto()
    ADVANCE_FLOW_BUS = auto()

class ProcessValueRef(Entity):
    
    def __init__(
        self,
        systemic_space: SystemicSpace,
        scope: ProcessValueScope,
        element_id: Optional[UUID] = None,
        state_id: Optional[UUID] = None,
    ) -> None:
        super().__init__(systemic_space)
        if scope is ProcessValueScope.SYSTEM_STATE and (
            state_id is None or element_id is None
        ):
            raise ValueError(
                "Una referencia al estado del sistema requiere state_id y element_id."
            )
        if scope is ProcessValueScope.TOKEN_STATE and state_id is not None:
            raise ValueError("Una referencia al token toma su estado del token activador.")
        if scope is ProcessValueScope.TOKEN_STATE and element_id is None:
            raise ValueError("Una lectura del token requiere element_id.")
        if scope is ProcessValueScope.SIMULATION_TIME and (
            element_id is not None or state_id is not None
        ):
            raise ValueError("SIMULATION_TIME no requiere estado ni elemento.")
        self.identificator = Identification(entity=Ontology.EntityType.PROCESS_VALUE)
        self.scope = scope
        self.element_id = element_id
        self.state_id = state_id
        self.space_auto_registry()

class ProcessExpression(Entity):
    def __init__(
        self,
        systemic_space: SystemicSpace,
        operator: ProcessExpressionOperator,
        operands: tuple[UUID, ...] = (),
        value: Optional[StateValue] = None,
        value_ref_id: Optional[UUID] = None,
    ) -> None:
        super().__init__(systemic_space)
        self.identificator = Identification(entity=Ontology.EntityType.PROCESS_EXPRESSION)
        self.operator = operator
        self.operands = operands
        self.value = value
        self.value_ref_id = value_ref_id
        self.space_auto_registry()

    def evaluate(
        self,
        system: System,
        token: Optional[Token],
        now: float = 0.0,
    ) -> StateValue:
        if self.operator in (
            ProcessExpressionOperator.CONSTANT,
            ProcessExpressionOperator.DISCRETE_CONSTANT,
        ):
            if self.value is None or self.operands or self.value_ref_id is not None:
                raise ValueError(
                    f"La expresión {self.operator.name} requiere solo un valor."
                )
            if self.operator is ProcessExpressionOperator.CONSTANT:
                if isinstance(self.value, (str, Enum)) or not isinstance(
                    self.value, (int, float, np.number)
                ):
                    raise TypeError("CONSTANT requiere un valor numérico.")
                result: StateValue = float(self.value)
            else:
                if not isinstance(self.value, (str, int, bool, Enum)):
                    raise TypeError("DISCRETE_CONSTANT requiere un valor categórico.")
                result = self.value
        elif self.operator is ProcessExpressionOperator.READ:
            if self.value_ref_id is None or self.operands or self.value is not None:
                raise ValueError("La expresión READ requiere solo value_ref_id.")
            reference = self.systemic_space.get(self.value_ref_id)
            if not isinstance(reference, ProcessValueRef):
                raise TypeError("value_ref_id debe referir a ProcessValueRef.")
            if reference.scope is ProcessValueScope.SIMULATION_TIME:
                result = now
                if not np.isfinite(result):
                    raise ValueError("El tiempo del proceso debe ser finito.")
                return float(result)
            if reference.scope is ProcessValueScope.SYSTEM_STATE:
                if reference.state_id is None:
                    raise ValueError("La referencia al estado requiere state_id.")
                state = system.states.get(reference.state_id)
                if state is None:
                    raise KeyError("El estado leído no pertenece al System del proceso.")
            else:
                if token is None:
                    raise RuntimeError("La lectura del estado del token requiere token_id.")
                state = token.charge
                if not isinstance(state, SystemState):
                    raise TypeError("El token activador no contiene un SystemState.")
            if reference.element_id is None:
                raise ValueError("La referencia de estado carece de element_id.")
            referenced_element = state.elements.get(reference.element_id)
            if isinstance(referenced_element, DiscreteStateElement):
                return state.read_discrete_value(reference.element_id)
            raw_value = state.read_element(reference.element_id)
            if isinstance(raw_value, np.ndarray):
                if raw_value.ndim != 0:
                    raise ValueError(
                        "Process requiere valores escalares en esta versión."
                    )
            elif not isinstance(raw_value, (int, float, np.number)):
                raise TypeError("ProcessExpression solo puede leer valores numéricos.")
            if np.asarray(raw_value).ndim != 0:
                raise ValueError("Process requiere valores escalares en esta versión.")
            result = float(raw_value)
        else:
            if self.value is not None or self.value_ref_id is not None:
                raise ValueError("Solo CONSTANT y READ aceptan un valor directo.")
            values = []
            for operand_id in self.operands:
                operand = self.systemic_space.get(operand_id)
                if not isinstance(operand, ProcessExpression):
                    raise TypeError("Los operandos deben referir a ProcessExpression.")
                values.append(operand.evaluate(system, token, now))
            if any(
                isinstance(value, (str, bool, Enum))
                or not isinstance(value, (int, float, np.number))
                for value in values
            ):
                raise TypeError("Las operaciones aritméticas requieren operandos numéricos.")
            values = [float(value) for value in values]
            arities = {
                ProcessExpressionOperator.NEGATE: 1,
                ProcessExpressionOperator.ADD: 2,
                ProcessExpressionOperator.SUBTRACT: 2,
                ProcessExpressionOperator.MULTIPLY: 2,
                ProcessExpressionOperator.DIVIDE: 2,
                ProcessExpressionOperator.MODULO: 2,
                ProcessExpressionOperator.POWER: 2,
                ProcessExpressionOperator.MIN: 2,
                ProcessExpressionOperator.MAX: 2,
            }
            if len(values) != arities[self.operator]:
                raise ValueError(
                    f"{self.operator.name} requiere {arities[self.operator]} operandos."
                )
            if self.operator is ProcessExpressionOperator.ADD:
                result = values[0] + values[1]
            elif self.operator is ProcessExpressionOperator.SUBTRACT:
                result = values[0] - values[1]
            elif self.operator is ProcessExpressionOperator.MULTIPLY:
                result = values[0] * values[1]
            elif self.operator is ProcessExpressionOperator.DIVIDE:
                result = values[0] / values[1]
            elif self.operator is ProcessExpressionOperator.MODULO:
                result = values[0] % values[1]
            elif self.operator is ProcessExpressionOperator.POWER:
                with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                    result = np.power(values[0], values[1])
            elif self.operator is ProcessExpressionOperator.NEGATE:
                result = -values[0]
            elif self.operator is ProcessExpressionOperator.MIN:
                result = min(values)
            else:
                result = max(values)
        if isinstance(result, (str, bool, Enum)):
            return result
        if isinstance(result, int):
            return result
        if not isinstance(result, (int, float, np.number)):
            raise TypeError("La expresión del proceso produjo un valor no escalar.")
        if not np.isfinite(result):
            raise ValueError("La expresión del proceso produjo un valor no finito.")
        return float(result)

class ProcessCondition(Entity):
    def __init__(
        self,
        systemic_space: SystemicSpace,
        operator: ProcessConditionOperator,
        operands: tuple[UUID, ...],
    ) -> None:
        super().__init__(systemic_space)
        self.identificator = Identification(entity=Ontology.EntityType.PROCESS_CONDITION)
        self.operator = operator
        self.operands = operands
        self.space_auto_registry()

    def evaluate(
        self,
        system: System,
        token: Optional[Token],
        now: float = 0.0,
    ) -> bool:
        expected_arity = {
            ProcessConditionOperator.NOT: 1,
            ProcessConditionOperator.AND: 2,
            ProcessConditionOperator.OR: 2,
            ProcessConditionOperator.EQUAL: 2,
            ProcessConditionOperator.NOT_EQUAL: 2,
            ProcessConditionOperator.LESS: 2,
            ProcessConditionOperator.LESS_EQUAL: 2,
            ProcessConditionOperator.GREATER: 2,
            ProcessConditionOperator.GREATER_EQUAL: 2,
        }[self.operator]
        if len(self.operands) != expected_arity:
            raise ValueError(
                f"{self.operator.name} requiere {expected_arity} operandos."
            )
        values = []
        for operand_id in self.operands:
            operand = self.systemic_space.get(operand_id)
            if isinstance(operand, ProcessExpression):
                values.append(operand.evaluate(system, token, now))
            elif isinstance(operand, ProcessCondition):
                values.append(operand.evaluate(system, token, now))
            else:
                raise TypeError("Los operandos deben ser expresiones o condiciones.")
        if self.operator in (
            ProcessConditionOperator.LESS,
            ProcessConditionOperator.LESS_EQUAL,
            ProcessConditionOperator.GREATER,
            ProcessConditionOperator.GREATER_EQUAL,
        ) and any(
            isinstance(value, (str, bool, Enum))
            or not isinstance(value, (int, float, np.number))
            for value in values
        ):
            raise TypeError("Las comparaciones de orden requieren valores numéricos.")
        if self.operator is ProcessConditionOperator.NOT:
            return not bool(values[0])
        if self.operator is ProcessConditionOperator.AND:
            return bool(values[0]) and bool(values[1])
        if self.operator is ProcessConditionOperator.OR:
            return bool(values[0]) or bool(values[1])
        if self.operator is ProcessConditionOperator.EQUAL:
            return values[0] == values[1]
        if self.operator is ProcessConditionOperator.NOT_EQUAL:
            return values[0] != values[1]
        if self.operator is ProcessConditionOperator.LESS:
            return values[0] < values[1]
        if self.operator is ProcessConditionOperator.LESS_EQUAL:
            return values[0] <= values[1]
        if self.operator is ProcessConditionOperator.GREATER:
            return values[0] > values[1]
        return values[0] >= values[1]

class ProcessTrigger(Entity):
    def __init__(
        self,
        systemic_space: SystemicSpace,
        kind: ProcessTriggerKind,
        processor_id: Optional[UUID] = None,
        condition_id: Optional[UUID] = None,
        interval: Optional[float] = None,
        initial_delay: Optional[float] = None,
        scheduled_at: Optional[datetime] = None,
    ) -> None:
        super().__init__(systemic_space)
        if kind is ProcessTriggerKind.TOKEN_RECEIVED and processor_id is None:
            raise ValueError("TOKEN_RECEIVED requiere processor_id.")
        if kind is ProcessTriggerKind.CONDITION and condition_id is None:
            raise ValueError("CONDITION requiere condition_id.")
        if kind is ProcessTriggerKind.INTERVAL:
            if interval is None or not np.isfinite(interval) or interval <= 0:
                raise ValueError("INTERVAL requiere un intervalo finito y positivo.")
            if initial_delay is not None and (
                not np.isfinite(initial_delay) or initial_delay < 0
            ):
                raise ValueError("initial_delay debe ser finito y no negativo.")
        elif interval is not None:
            raise ValueError("interval solo aplica al trigger INTERVAL.")
        elif initial_delay is not None:
            raise ValueError("initial_delay solo aplica al trigger INTERVAL.")
        if kind is ProcessTriggerKind.CALENDAR:
            if not isinstance(scheduled_at, datetime):
                raise ValueError("CALENDAR requiere scheduled_at como datetime.")
            SimulationClock(scheduled_at)
        elif scheduled_at is not None:
            raise ValueError("scheduled_at solo aplica al trigger CALENDAR.")
        self.identificator = Identification(entity=Ontology.EntityType.PROCESS_TRIGGER)
        self.kind = kind
        self.processor_id = processor_id
        self.condition_id = condition_id
        self.interval = interval
        self.initial_delay = initial_delay
        self.scheduled_at = scheduled_at
        self.space_auto_registry()

class ProcessAction(Entity):
    def __init__(
        self, systemic_space: SystemicSpace, kind: ProcessActionKind, expression_id: Optional[UUID] = None, state_id: Optional[UUID] = None,
        element_id: Optional[UUID] = None, processor_id: Optional[UUID] = None, port_id: Optional[UUID] = None, token_id: Optional[UUID] = None,
        structure_id: Optional[UUID] = None, relation_id: Optional[UUID] = None,
        component_id: Optional[UUID] = None, duration: Optional[float] = None,) -> None:
        super().__init__(systemic_space)

        if kind in (
            ProcessActionKind.SET_STATE_VALUE,
            ProcessActionKind.SET_TOKEN_VALUE,
            ProcessActionKind.SET_COMPONENT_VALUE,
        ) and (expression_id is None or element_id is None):
            raise ValueError("Una acción SET requiere expression_id y element_id.")
        if kind is ProcessActionKind.SET_STATE_VALUE and state_id is None:
            raise ValueError("SET_STATE_VALUE requiere state_id.")
        if kind is ProcessActionKind.SET_TOKEN_VALUE and state_id is not None:
            raise ValueError("SET_TOKEN_VALUE usa el token activador como estado.")
        if kind is ProcessActionKind.SET_COMPONENT_VALUE and component_id is None:
            raise ValueError("SET_COMPONENT_VALUE requiere component_id.")
        if kind is ProcessActionKind.EMIT_TOKEN and (
            processor_id is None or port_id is None
        ):
            raise ValueError("EMIT_TOKEN requiere processor_id y port_id.")
        if kind is ProcessActionKind.APPLY_STATE_COUPLING and (
            structure_id is None or relation_id is None
        ):
            raise ValueError(
                "APPLY_STATE_COUPLING requiere structure_id y relation_id."
            )
        if kind is ProcessActionKind.PROPAGATE_STATE_COUPLINGS and structure_id is None:
            raise ValueError(
                "PROPAGATE_STATE_COUPLINGS requiere structure_id."
            )
        if kind is ProcessActionKind.ADVANCE_FLOW_BUS:
            if structure_id is None or duration is None:
                raise ValueError(
                    "ADVANCE_FLOW_BUS requiere structure_id y duration positivo."
                )
            if isinstance(duration, bool) or not isinstance(
                duration, (int, float, np.number)
            ):
                raise TypeError("duration debe ser un número.")
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError("duration debe ser finito y positivo.")
        elif duration is not None:
            raise ValueError("duration solo aplica a ADVANCE_FLOW_BUS.")
        
        self.identificator = Identification(entity=Ontology.EntityType.PROCESS_ACTION)
        self.kind = kind
        self.expression_id = expression_id
        self.state_id = state_id
        self.element_id = element_id
        self.processor_id = processor_id
        self.port_id = port_id
        self.token_id = token_id
        self.structure_id = structure_id
        self.relation_id = relation_id
        self.component_id = component_id
        self.duration = duration
        self.space_auto_registry()

class Process(Entity):
    
    def __init__(self, systemic_space: SystemicSpace, system_id: UUID, label: Optional[str] = None,) -> None:

        super().__init__(systemic_space)
        self.identificator = Identification(entity=Ontology.EntityType.PROCESS,label=label,)

        self.system_id = system_id
        self.trigger_ids: list[UUID] = []
        self.action_ids: list[UUID] = []
        self.space_auto_registry()

    def add_trigger(self, trigger_id: UUID) -> None:
        trigger = self.systemic_space.get(trigger_id)
        if not isinstance(trigger, ProcessTrigger):
            raise KeyError("El ID indicado no es un ProcessTrigger.")
        if trigger_id not in self.trigger_ids:
            self.trigger_ids.append(trigger_id)

    def add_action(self, action_id: UUID) -> None:
        action = self.systemic_space.get(action_id)
        if not isinstance(action, ProcessAction):
            raise KeyError("El ID indicado no es un ProcessAction.")
        if action_id not in self.action_ids:
            self.action_ids.append(action_id)

    def execute(self, trigger_id: UUID, token_id: Optional[UUID] = None, now: float = 0.0,) -> bool:

        system = self.systemic_space.get(self.system_id)
        trigger = self.systemic_space.get(trigger_id)
        if not np.isfinite(now):
            raise ValueError("now debe ser un tiempo de simulación finito.")

        if not isinstance(system, System) or self.identificator.id not in system.processes:
            raise ValueError("Process debe estar suscrito al System declarado.")
        if trigger_id not in self.trigger_ids or not isinstance(trigger, ProcessTrigger):
            raise ValueError("El trigger no pertenece a este Process.")
        
        token = self.systemic_space.get(token_id) if token_id is not None else None
        
        if token_id is not None and not isinstance(token, Token):
            raise KeyError("token_id no corresponde a un Token.")
        if trigger.kind is ProcessTriggerKind.TOKEN_RECEIVED:
            if trigger.processor_id is None:
                raise ValueError("TOKEN_RECEIVED requiere processor_id.")
            receiver = system.processors.get(trigger.processor_id)

            if (receiver is None or token_id is None or token_id not in receiver.token_deposit):
                raise ValueError("TOKEN_RECEIVED solo se ejecuta para un token depositado en el processor configurado.")
            
        if trigger.condition_id is not None:
            condition = self.systemic_space.get(trigger.condition_id)
            if not isinstance(condition, ProcessCondition):
                raise TypeError("condition_id debe referir a ProcessCondition.")
            if not condition.evaluate(system, token, now):
                return False

        updates: dict[SystemState, dict[UUID, StateValue]] = {}
        emissions: list[tuple[SystemProcessor, UUID, UUID]] = []
        coupling_actions: list[tuple[StateCouplingStructure, UUID]] = []
        propagation_actions: list[StateCouplingStructure] = []
        flow_actions: list[tuple[FlowBusStructure, float]] = []
        
        for action_id in self.action_ids:
            action = self.systemic_space.get(action_id)
            if not isinstance(action, ProcessAction):
                raise TypeError("Process contiene una acción inválida.")
            if action.kind in (
                ProcessActionKind.SET_STATE_VALUE,
                ProcessActionKind.SET_TOKEN_VALUE,
                ProcessActionKind.SET_COMPONENT_VALUE,
            ):

                if action.expression_id is None or action.element_id is None:
                    raise ValueError("Una acción SET requiere expression_id y element_id.")
                expression = self.systemic_space.get(action.expression_id)
                if not isinstance(expression, ProcessExpression):
                    raise TypeError("expression_id debe referir a ProcessExpression.")
                if action.kind is ProcessActionKind.SET_STATE_VALUE:
                    if action.state_id is None:
                        raise ValueError("SET_STATE_VALUE requiere state_id.")
                    state = system.states.get(action.state_id)
                    if state is None:
                        raise KeyError("El estado destino no pertenece al System.")
                elif action.kind is ProcessActionKind.SET_COMPONENT_VALUE:
                    if action.component_id is None:
                        raise ValueError("SET_COMPONENT_VALUE requiere component_id.")
                    component = self.systemic_space.get(action.component_id)
                    if not isinstance(component, Component):
                        raise KeyError("component_id no corresponde a un Component.")
                    if not any(
                        action.component_id in structure.components
                        for structure in system.structures.values()
                    ):
                        raise ValueError(
                            "El Component destino debe pertenecer a una estructura del System."
                        )
                    state = component.system_state
                else:
                    if token is None or not isinstance(token.charge, SystemState):
                        raise RuntimeError("SET_TOKEN_VALUE requiere un token con SystemState.")
                    state = token.charge
                value = expression.evaluate(system, token, now)
                updates.setdefault(state, {})[action.element_id] = value
            elif action.kind is ProcessActionKind.APPLY_STATE_COUPLING:
                if action.structure_id is None or action.relation_id is None:
                    raise ValueError(
                        "APPLY_STATE_COUPLING requiere structure_id y relation_id."
                    )
                structure = system.structures.get(action.structure_id)
                if not isinstance(structure, StateCouplingStructure):
                    raise KeyError(
                        "La estructura de acoplamiento no pertenece al System."
                    )
                if action.relation_id not in structure.relations:
                    raise KeyError(
                        "La relación no pertenece a la estructura de acoplamiento."
                    )
                coupling_actions.append((structure, action.relation_id))
            elif action.kind is ProcessActionKind.PROPAGATE_STATE_COUPLINGS:
                
                if action.structure_id is None:
                    raise ValueError(
                        "PROPAGATE_STATE_COUPLINGS requiere structure_id."
                    )
                structure = system.structures.get(action.structure_id)
                if not isinstance(structure, StateCouplingStructure):
                    raise KeyError(
                        "La estructura de acoplamiento no pertenece al System."
                    )
                propagation_actions.append(structure)
            elif action.kind is ProcessActionKind.ADVANCE_FLOW_BUS:
                if action.structure_id is None or action.duration is None:
                    raise ValueError(
                        "ADVANCE_FLOW_BUS requiere structure_id y duration."
                    )
                structure = system.structures.get(action.structure_id)
                if not isinstance(structure, FlowBusStructure):
                    raise KeyError("El FlowBus no pertenece al System.")
                flow_actions.append((structure, action.duration))
            else:
                if action.processor_id is None or action.port_id is None:
                    raise ValueError("EMIT_TOKEN requiere processor_id y port_id.")
                processor = system.processors.get(action.processor_id)
                if processor is None or action.port_id not in processor.ports:
                    raise KeyError("El processor o port de EMIT_TOKEN no pertenece al System.")
                emitted_token_id = action.token_id or token_id
                if emitted_token_id is None:
                    raise ValueError("EMIT_TOKEN requiere token_id o un token activador.")
                if emitted_token_id not in processor.token_deposit:
                    raise KeyError("El processor no tiene depositado el token a emitir.")
                emissions.append((processor, action.port_id, emitted_token_id))

        emission_token_ids = [token_id for _, _, token_id in emissions]
        if len(emission_token_ids) != len(set(emission_token_ids)):
            raise ValueError("Un Process no puede emitir el mismo token más de una vez.")

        validated_updates = {
            state: state.validate_state_updates(state_updates)
            for state, state_updates in updates.items()
        }

        for state, state_updates in validated_updates.items():
            state._commit_state_updates(state_updates)
        for structure, relation_id in coupling_actions:
            structure.apply_relation(relation_id)
        for structure in propagation_actions:
            structure.propagate()
        for structure, duration in flow_actions:
            structure.advance(duration, now=now)
        for processor, port_id, emitted_token_id in emissions:
            processor.emit_token(emitted_token_id, port_id)
        return True

# =================================
# SYSTEM PROCESSOR
# =================================

class Token(Entity):

    def __init__(self, systemic_space : SystemicSpace, charged : bool = False, label: Optional[str] = None):

        super().__init__(systemic_space = systemic_space)
        self.identificator : Identification = Identification(
            entity = Ontology.EntityType.TOKEN,
            label = label,
        )
        self.space_auto_registry()

        self.charge : System | SystemState = System(systemic_space = systemic_space) if charged else SystemState(systemic_space = systemic_space)

    def inject_description(self, system : UUID, system_element : UUID) -> None:
        system_entity : System = self.systemic_space.get(system)
        system_entity.suscribe_systemic_description(system_element)

class FlowCohort(Token):
    """A routed token representing a homogeneous group of discrete agents."""

    def __init__(
        self,
        systemic_space: SystemicSpace,
        population: int,
        attributes: Optional[Mapping[str, StateValue]] = None,
        label: Optional[str] = None,
    ) -> None:
        if isinstance(population, bool) or not isinstance(population, int) or population <= 0:
            raise ValueError("population debe ser un entero positivo.")
        super().__init__(systemic_space, label=label)
        if not isinstance(self.charge, SystemState):
            raise RuntimeError("Una cohorte requiere un SystemState.")
        self.population = population
        self.attribute_elements: dict[str, UUID] = {}
        for name, value in (attributes or {}).items():
            if not name or not name.strip():
                raise ValueError("Los atributos de una cohorte requieren nombres.")
            if isinstance(value, (str, bool, Enum)):
                element = DiscreteStateElement(
                    systemic_space,
                    name,
                    initial_value=value,
                )
            elif isinstance(value, (int, float, np.number)):
                if not np.isfinite(value):
                    raise ValueError(f"El atributo '{name}' debe ser finito.")
                element = VariableElement(systemic_space, name, float(value))
            else:
                raise TypeError(
                    "Los atributos de cohorte deben ser valores escalares numéricos o categóricos."
                )
            self.charge.add_element(element.identificator.id)
            self.attribute_elements[name] = element.identificator.id

class Port(Entity):

    def __init__(self, systemic_space : SystemicSpace, destination : UUID) -> None:

        super().__init__(systemic_space = systemic_space)
        self.identificator : Identification = Identification(entity = Ontology.EntityType.PORT)
        self.space_auto_registry()

        self.token_queue : dict[UUID, Token] = {}
        self.destination_processor : UUID = destination

        self.validate_destination(destination)

    def validate_destination(self, sor_processor : UUID) -> None:
        if self.systemic_space.validator.validate_entity_class(sor_processor, SystemProcessor) is False:
            raise KeyError("Only system processors can be assigned")
        
    def clear_out_port(self) -> None:
        destination : SystemProcessor = self.systemic_space.get(self.destination_processor)
        for token_id, token in tuple(self.token_queue.items()):
            destination.receive_token(token_id, token)
            del self.token_queue[token_id]

class SystemProcessor(Entity):

    def __init__(self, systemic_space : SystemicSpace) -> None:

        super().__init__(systemic_space = systemic_space)
        self.identificator : Identification = Identification(entity = Ontology.EntityType.SYSTEM_PROCESSOR)
        self.space_auto_registry()

        self.ports : dict[UUID, Port] = {}
        self.token_deposit : dict[UUID, Token] = {}
        self.system_state: Optional[SystemState] = None
        self.system_id: Optional[UUID] = None

    def add_port(self, port : UUID) -> None:
        if self.systemic_space.validator.validate_entity_class(port, Port) is False:
            raise KeyError("The provided id does not correspond to any port")
        self.ports[port] = self.systemic_space.get(port)

    def bind_state(self, state: SystemState) -> None:
        if state.systemic_space is not self.systemic_space:
            raise ValueError("El estado del procesador debe pertenecer al mismo espacio.")
        self.system_state = state

    def receive_token(self, token_id: UUID, token: Token) -> None:
        if token.identificator.id != token_id:
            raise ValueError("El ID no coincide con el token recibido.")
        if token.systemic_space is not self.systemic_space:
            raise ValueError("El token pertenece a otro SystemicSpace.")
        if token_id in self.token_deposit:
            raise ValueError("El procesador ya contiene este token.")
        self.token_deposit[token_id] = token
        if self.system_id is not None:
            system = self.systemic_space.get(self.system_id)
            if not isinstance(system, System):
                raise RuntimeError("El System vinculado al processor no es válido.")
            system.trigger_processes(
                ProcessTriggerKind.TOKEN_RECEIVED,
                processor_id=self.identificator.id,
                token_id=token_id,
                now=system.simulation_time,
            )

    def process_token(self, token_id: UUID, transition: Callable[[StateTransducer, StateTransducer],TokenProcessingResult,], 
    token_writable_elements: Optional[set[UUID]] = None, processor_writable_elements: Optional[set[UUID]] = None,) -> TokenProcessingResult:
    
        """Apply one domain rule to a deposited token and this processor's state.

        The transition reads through token_state and processor_state bridges, then
        returns proposed writes. Both states validate the proposals before either
        state is mutated.
        """
        token = self.token_deposit.get(token_id)

        if token is None:
            raise KeyError("El procesador no contiene el token indicado.")
        if not isinstance(token.charge, SystemState):
            raise TypeError("El token debe contener un SystemState para procesarse.")
        if self.system_state is None:
            raise RuntimeError("El procesador debe vincular su SystemState antes de procesar.")

        token_state = token.charge
        processor_state = self.system_state
        
        if token_state.systemic_space is not self.systemic_space:
            raise ValueError("El estado del token pertenece a otro espacio sistémico.")

        token_bridge = StateTransducer(consumer=processor_state, provider=token_state,)
        processor_bridge = StateTransducer(consumer=token_state, provider=processor_state,)

        result = transition(token_bridge, processor_bridge)

        if not isinstance(result, TokenProcessingResult):
            raise TypeError("La regla debe devolver una instancia de TokenProcessingResult.")

        token_permissions = set(token_writable_elements or ())
        processor_permissions = set(processor_writable_elements or ())
        unauthorized_token_updates = result.token_updates.keys() - token_permissions
        unauthorized_processor_updates = (
            result.processor_updates.keys() - processor_permissions
        )
        if unauthorized_token_updates or unauthorized_processor_updates:
            raise PermissionError(
                "La regla propuso cambios fuera de los permisos de escritura."
            )

        validated_token_updates = token_state.validate_values(result.token_updates)
        validated_processor_updates = processor_state.validate_values(
            result.processor_updates
        )

        token_state._commit_validated_values(validated_token_updates)
        processor_state._commit_validated_values(validated_processor_updates)
        return result

    # GENERATIVE
    def generate_token(self, charged: bool = False, label: Optional[str] = None) -> Token:
        genesis_token = Token(
            systemic_space=self.systemic_space,
            charged=charged,
            label=label,
        )
        self.token_deposit[genesis_token.identificator.id] = genesis_token
        return genesis_token

    def generate_cohort(
        self,
        population: int,
        attributes: Optional[Mapping[str, StateValue]] = None,
        label: Optional[str] = None,
    ) -> FlowCohort:
        cohort = FlowCohort(
            self.systemic_space,
            population,
            attributes,
            label,
        )
        self.token_deposit[cohort.identificator.id] = cohort
        return cohort

    def merge_cohorts(self, cohort_ids: Sequence[UUID]) -> FlowCohort:
        identifiers = tuple(cohort_ids)
        if len(identifiers) < 2 or len(set(identifiers)) != len(identifiers):
            raise ValueError("merge_cohorts requiere al menos dos IDs únicos.")
        cohorts = []
        for cohort_id in identifiers:
            cohort = self.token_deposit.get(cohort_id)
            if not isinstance(cohort, FlowCohort):
                raise KeyError("Todas las cohortes deben estar depositadas aquí.")
            cohorts.append(cohort)
        reference_state = cohorts[0].charge
        if not isinstance(reference_state, SystemState):
            raise RuntimeError("Una cohorte dejó de contener un SystemState.")
        reference_values = {
            element.name: reference_state.read_element(element_id)
            for element_id, element in reference_state.elements.items()
            if isinstance(element, (VariableElement, DiscreteStateElement))
        }
        for cohort in cohorts[1:]:
            state = cohort.charge
            if not isinstance(state, SystemState):
                raise RuntimeError("Una cohorte dejó de contener un SystemState.")
            values = {
                element.name: state.read_element(element_id)
                for element_id, element in state.elements.items()
                if isinstance(element, (VariableElement, DiscreteStateElement))
            }
            if values != reference_values:
                raise ValueError("Solo se pueden fusionar cohortes con el mismo estado.")
        cohorts[0].population = sum(cohort.population for cohort in cohorts)
        for cohort in cohorts[1:]:
            self.token_deposit.pop(cohort.identificator.id)
        return cohorts[0]

    def split_cohort(
        self,
        cohort_id: UUID,
        population: int,
        label: Optional[str] = None,
    ) -> FlowCohort:
        cohort = self.token_deposit.get(cohort_id)
        if not isinstance(cohort, FlowCohort):
            raise KeyError("La cohorte debe estar depositada en este processor.")
        if (
            isinstance(population, bool)
            or not isinstance(population, int)
            or population <= 0
            or population >= cohort.population
        ):
            raise ValueError(
                "population debe ser positivo y menor que la cohorte original."
            )
        state = cohort.charge
        if not isinstance(state, SystemState):
            raise RuntimeError("Una cohorte dejó de contener un SystemState.")
        attributes: dict[str, StateValue] = {}
        for element_id, element in state.elements.items():
            if isinstance(element, VariableElement):
                attributes[element.name] = state.read_value(element_id)
            elif isinstance(element, DiscreteStateElement):
                attributes[element.name] = state.read_discrete_value(element_id)
        split = self.generate_cohort(population, attributes, label)
        split_state = split.charge
        if not isinstance(split_state, SystemState):
            raise RuntimeError("La cohorte dividida debe contener un SystemState.")
        for element in state.elements.values():
            if not isinstance(element, (VariableElement, DiscreteStateElement)):
                continue
            split_element_id = split.attribute_elements[element.name]
            split_element = split_state.elements[split_element_id]
            if isinstance(element, VariableElement) and isinstance(
                split_element,
                VariableElement,
            ):
                split_element.constraints = element.constraints
            elif isinstance(element, DiscreteStateElement) and isinstance(
                split_element,
                DiscreteStateElement,
            ):
                split_element.allowed_values = element.allowed_values
        cohort.population -= population
        return split

    # CONSERVATIVE / ROUTING

    def store_token(self, token_id : UUID) -> None:
        if self.systemic_space.validator.validate_entity_class(token_id, Token) is False:
            raise KeyError("Can only store Tokens")
        self.token_deposit[token_id] = self.systemic_space.get(token_id)

    def emit_token(self, token_id: UUID, port : UUID) -> None:
        emission_port : Port = self.ports[port]
        emission_port.token_queue[token_id] = self.token_deposit[token_id]
        self.token_deposit.pop(token_id)

    def obliterate_token(self, token_id : UUID) -> None:
        self.token_deposit.pop(token_id)

# ============================
# SYSTEM STRUCTURES
# ============================

class Component(Entity):

    def __init__(
        self,
        systemic_space: SystemicSpace,
        is_system: bool = False,
        label: Optional[str] = None,
    ) -> None:

        super().__init__(systemic_space = systemic_space)
        self.identificator : Identification = Identification(
            entity=Ontology.EntityType.COMPONENT,
            label=label,
        )
        self.space_auto_registry()

        self.system_state : SystemState = SystemState(self.systemic_space)

        self.system : Optional[System] = System(systemic_space = systemic_space) if is_system else None

class SystemRelation(Entity):

    def __init__(
        self,
        systemic_space: SystemicSpace,
        source_id: UUID,
        target_id: UUID,
        label: Optional[str] = None,
    ) -> None:
        
        super().__init__(systemic_space=systemic_space)

        source = systemic_space.get(source_id)
        target = systemic_space.get(target_id)
        
        if not isinstance(source, Entity) or not isinstance(target, Entity):
            raise KeyError("Los extremos de una relación deben existir en SystemicSpace.")
        
        self.identificator = Identification(
            entity=Ontology.EntityType.SYSTEM_RELATION,
            label=label,
        )
        self.source_id = source_id
        self.target_id = target_id
        self.space_auto_registry()

class SystemStructure(Entity):

    def __init__(self, systemic_space : SystemicSpace) -> None:

        super().__init__(systemic_space = systemic_space)
        self.identificator : Identification = Identification(entity = Ontology.EntityType.SYSTEM_STRUCTURE)
        self.space_auto_registry()

        self.system_id: Optional[UUID] = None
        self.nodes: dict[UUID, Entity] = {}
        self.components : dict[UUID, Component] = {}
        self.relations: dict[UUID, SystemRelation] = {}
        self.topology : nx.MultiDiGraph = nx.MultiDiGraph()

    def bind_to_system(self, system_id: UUID) -> None:
        system = self.systemic_space.get(system_id)
        if not isinstance(system, System):
            raise KeyError("El ID indicado no corresponde a un System.")
        if self.system_id not in (None, system_id):
            raise ValueError("SystemStructure ya pertenece a otro System.")
        if system.systemic_space is not self.systemic_space:
            raise ValueError("SystemStructure y System deben compartir SystemicSpace.")
        self.system_id = system_id

    def add_node(self, entity_id: UUID) -> None:
        entity = self.systemic_space.get(entity_id)
        if not isinstance(entity, Entity):
            raise KeyError("El ID no corresponde a una entidad de SystemicSpace.")
        if entity.systemic_space is not self.systemic_space:
            raise ValueError("El nodo debe pertenecer al mismo SystemicSpace.")
        if entity_id in self.nodes:
            return
        self.nodes[entity_id] = entity
        self.topology.add_node(entity_id)
        if isinstance(entity, Component):
            self.components[entity_id] = entity
            if self.system_id is not None:
                system = self.systemic_space.get(self.system_id)
                if isinstance(system, System):
                    system._attach_history_state(entity.system_state)

    def add_component(self, component : UUID) -> None:
        if not isinstance(self.systemic_space.get(component), Component):
            raise KeyError("The id does not match a Component in the systemic space.")
        self.add_node(component)

    def add_relation(self, relation_id: UUID) -> None:
        relation = self.systemic_space.get(relation_id)
        if not isinstance(relation, SystemRelation):
            raise KeyError("El ID no corresponde a una SystemRelation.")
        if relation.systemic_space is not self.systemic_space:
            raise ValueError("La relación debe pertenecer al mismo SystemicSpace.")
        if relation.source_id not in self.nodes or relation.target_id not in self.nodes:
            raise ValueError("Los dos extremos deben estar suscritos a SystemStructure.")
        if relation_id in self.relations:
            return
        self.relations[relation_id] = relation
        self.topology.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation_id,
            relation_id=relation_id,
        )

    def remove_relation(self, relation_id: UUID) -> None:
        relation = self.relations.pop(relation_id, None)
        if relation is None:
            raise KeyError("La relación no pertenece a SystemStructure.")
        self.topology.remove_edge(
            relation.source_id,
            relation.target_id,
            key=relation_id,
        )

    def remove_node(self, entity_id: UUID) -> None:
        if entity_id not in self.nodes:
            raise KeyError("El nodo no pertenece a SystemStructure.")
        incident_relations = tuple(
            relation_id
            for relation_id, relation in self.relations.items()
            if entity_id in (relation.source_id, relation.target_id)
        )
        for relation_id in incident_relations:
            self.remove_relation(relation_id)
        del self.nodes[entity_id]
        self.components.pop(entity_id, None)
        self.topology.remove_node(entity_id)

    def has_path(self, source_id: UUID, target_id: UUID) -> bool:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError("Los extremos de la consulta deben pertenecer a SystemStructure.")
        return nx.has_path(self.topology, source_id, target_id)

    def relations_between(
        self,
        source_id: UUID,
        target_id: UUID,
    ) -> tuple[SystemRelation, ...]:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError("Los extremos de la consulta deben pertenecer a SystemStructure.")
        relation_ids = self.topology.get_edge_data(source_id, target_id, default={})
        return tuple(self.relations[edge["relation_id"]] for edge in relation_ids.values())

class StateMappingRelation(SystemRelation):

    def __init__(
        self,
        systemic_space: SystemicSpace,
        source_component_id: UUID,
        source_element_id: UUID,
        target_component_id: UUID,
        target_element_id: UUID,
        transform: Callable[..., Any],
        input_constraints: Optional[ValueConstraints] = None,
        output_constraints: Optional[ValueConstraints] = None,
        label: Optional[str] = None,
    ) -> None:
        source_component = systemic_space.get(source_component_id)
        target_component = systemic_space.get(target_component_id)
        if not isinstance(source_component, Component):
            raise TypeError("El origen del acoplamiento debe ser un Component.")
        if not isinstance(target_component, Component):
            raise TypeError("El destino del acoplamiento debe ser un Component.")
        if source_component is target_component:
            raise ValueError("El acoplamiento debe conectar dos Components distintos.")
        source_element = source_component.system_state.elements.get(source_element_id)
        target_element = target_component.system_state.elements.get(target_element_id)
        if not isinstance(source_element, (VariableElement, DiscreteStateElement)):
            raise TypeError("El origen debe ser un elemento escalar de estado.")
        if not isinstance(target_element, (VariableElement, DiscreteStateElement)):
            raise TypeError("El destino debe ser un elemento escalar de estado.")
        if not callable(transform):
            raise TypeError("transform debe ser una función escalar invocable.")
        if input_constraints is not None and not isinstance(
            source_element, VariableElement
        ):
            raise TypeError("input_constraints solo aplica a variables numéricas.")
        if output_constraints is not None and not isinstance(
            target_element, VariableElement
        ):
            raise TypeError("output_constraints solo aplica a variables numéricas.")
        if input_constraints is not None and not isinstance(
            input_constraints, ValueConstraints
        ):
            raise TypeError("input_constraints debe ser ValueConstraints.")
        if output_constraints is not None and not isinstance(
            output_constraints, ValueConstraints
        ):
            raise TypeError("output_constraints debe ser ValueConstraints.")

        super().__init__(
            systemic_space,
            source_component_id,
            target_component_id,
            label,
        )
        self.identificator.entity = Ontology.EntityType.STATE_COUPLING_RELATION
        self.source_element_id = source_element_id
        self.target_element_id = target_element_id
        self.transform: Callable[..., Any] = transform
        self.input_constraints = input_constraints or (
            source_element.constraints
            if isinstance(source_element, VariableElement)
            else None
        )
        self.output_constraints = output_constraints or (
            target_element.constraints
            if isinstance(target_element, VariableElement)
            else None
        )

    def evaluate(self, source_value: Any = None) -> StateValue:
        source_component = self.systemic_space.get(self.source_id)
        target_component = self.systemic_space.get(self.target_id)
        if not isinstance(source_component, Component) or not isinstance(
            target_component, Component
        ):
            raise TypeError("Los extremos del acoplamiento deben seguir siendo Components.")

        source_state = source_component.system_state
        target_state = target_component.system_state
        source_element = source_state.elements.get(self.source_element_id)
        target_element = target_state.elements.get(self.target_element_id)
        if source_value is None:
            if isinstance(source_element, VariableElement):
                source_value = source_state.read_value(self.source_element_id)
            elif isinstance(source_element, DiscreteStateElement):
                source_value = source_state.read_discrete_value(self.source_element_id)
            else:
                raise TypeError("El elemento fuente dejó de ser un estado escalar.")
        if isinstance(source_element, VariableElement):
            if isinstance(source_value, (str, bool, Enum)) or not isinstance(
                source_value, (int, float, np.number)
            ):
                raise TypeError("El origen numérico requiere un valor numérico.")
            if self.input_constraints is None:
                raise RuntimeError("Faltan restricciones para el elemento numérico.")
            validated_source = self.input_constraints.validate(
                float(source_value),
                "Valor de entrada del acoplamiento",
            )
            if isinstance(validated_source, np.ndarray):
                if validated_source.ndim != 0:
                    raise TypeError("El origen del acoplamiento debe ser escalar.")
                validated_source = validated_source.item()
            if isinstance(validated_source, (bool, str, Enum)) or not isinstance(
                validated_source, (int, float, np.number)
            ):
                raise TypeError("El origen del acoplamiento debe ser numérico.")
            source_value = float(validated_source)
        elif isinstance(source_element, DiscreteStateElement):
            if not isinstance(source_value, (str, int, bool, Enum)):
                raise TypeError("El origen categórico requiere un valor discreto.")
            source_value = source_element.validate_value(source_value)
        else:
            raise TypeError("El elemento fuente dejó de ser un estado escalar.")
        transformed_value = self.transform(source_value)
        if isinstance(target_element, VariableElement):
            if isinstance(transformed_value, (str, bool, Enum)) or not isinstance(
                transformed_value, (int, float, np.number)
            ):
                raise TypeError("El destino numérico requiere un valor numérico.")
            if self.output_constraints is None:
                raise RuntimeError("Faltan restricciones para el elemento destino.")
            validated_value = self.output_constraints.validate(
                float(transformed_value),
                f"Variable '{target_element.name}'",
            )
            if isinstance(validated_value, np.ndarray):
                if validated_value.ndim != 0:
                    raise TypeError("El acoplamiento debe producir un escalar numérico.")
                validated_value = validated_value.item()
            if isinstance(validated_value, (bool, str, Enum)) or not isinstance(
                validated_value, (int, float, np.number)
            ):
                raise TypeError("El acoplamiento debe producir un escalar numérico.")
            return float(validated_value)
        if isinstance(target_element, DiscreteStateElement):
            if not isinstance(transformed_value, (str, int, bool, Enum)):
                raise TypeError("El destino categórico requiere un valor discreto.")
            return target_element.validate_value(transformed_value)
        raise TypeError("El elemento destino dejó de ser un estado escalar.")

    def apply(self) -> StateValue:
        value = self.evaluate()
        source_component = self.systemic_space.get(self.source_id)
        target_component = self.systemic_space.get(self.target_id)
        if not isinstance(source_component, Component) or not isinstance(
            target_component, Component
        ):
            raise TypeError("Los extremos del acoplamiento deben ser Components.")
        target_component.system_state._commit_state_updates(
            {self.target_element_id: value}
        )
        return value

class StateCouplingRelation(StateMappingRelation):
    def __init__(
        self,
        systemic_space: SystemicSpace,
        source_component_id: UUID,
        source_element_id: UUID,
        target_component_id: UUID,
        target_element_id: UUID,
        transform: Callable[[float], float],
        input_constraints: Optional[ValueConstraints] = None,
        output_constraints: Optional[ValueConstraints] = None,
        label: Optional[str] = None,
    ) -> None:
        source = systemic_space.get(source_component_id)
        target = systemic_space.get(target_component_id)
        if not isinstance(source, Component) or not isinstance(target, Component):
            raise TypeError("Los extremos del acoplamiento deben ser Components.")
        if not isinstance(source.system_state.elements.get(source_element_id), VariableElement):
            raise TypeError("El origen debe ser una VariableElement.")
        if not isinstance(target.system_state.elements.get(target_element_id), VariableElement):
            raise TypeError("El destino debe ser una VariableElement.")
        super().__init__(
            systemic_space,
            source_component_id,
            source_element_id,
            target_component_id,
            target_element_id,
            transform,
            input_constraints,
            output_constraints,
            label,
        )

    def evaluate(self, source_value: Optional[float] = None) -> float:
        value = super().evaluate(source_value)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError("StateCouplingRelation debe producir un valor numérico.")
        return float(value)

    def apply(self) -> float:
        value = super().apply()
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError("StateCouplingRelation debe producir un valor numérico.")
        return float(value)

class CategoricalStateCouplingRelation(StateMappingRelation):
    def __init__(
        self,
        systemic_space: SystemicSpace,
        source_component_id: UUID,
        source_element_id: UUID,
        target_component_id: UUID,
        target_element_id: UUID,
        transform: Callable[[DiscreteValue], DiscreteValue],
        label: Optional[str] = None,
    ) -> None:
        source = systemic_space.get(source_component_id)
        target = systemic_space.get(target_component_id)
        if not isinstance(source, Component) or not isinstance(target, Component):
            raise TypeError("Los extremos del acoplamiento deben ser Components.")
        if not isinstance(
            source.system_state.elements.get(source_element_id),
            DiscreteStateElement,
        ):
            raise TypeError("El origen debe ser un DiscreteStateElement.")
        if not isinstance(
            target.system_state.elements.get(target_element_id),
            DiscreteStateElement,
        ):
            raise TypeError("El destino debe ser un DiscreteStateElement.")
        super().__init__(
            systemic_space,
            source_component_id,
            source_element_id,
            target_component_id,
            target_element_id,
            transform,
            label=label,
        )

    def evaluate(
        self,
        source_value: Optional[DiscreteValue] = None,
    ) -> DiscreteValue:
        value = super().evaluate(source_value)
        if not isinstance(value, (str, int, bool, Enum)):
            raise TypeError("El acoplamiento categórico debe producir un valor discreto.")
        return value

    def apply(self) -> DiscreteValue:
        value = super().apply()
        if not isinstance(value, (str, int, bool, Enum)):
            raise TypeError("El acoplamiento categórico debe producir un valor discreto.")
        return value

class StateCouplingStructure(SystemStructure):

    def __init__(self, systemic_space: SystemicSpace) -> None:
        super().__init__(systemic_space)
        self.identificator.entity = Ontology.EntityType.STATE_COUPLING_STRUCTURE

    def add_relation(self, relation_id: UUID) -> None:
        relation = self.systemic_space.get(relation_id)
        if not isinstance(relation, StateMappingRelation):
            raise TypeError(
                "StateCouplingStructure solo admite relaciones de mapeo de estado."
            )
        super().add_relation(relation_id)

    def apply_relation(self, relation_id: UUID) -> StateValue:
        relation = self.relations.get(relation_id)
        if not isinstance(relation, StateMappingRelation):
            raise KeyError("El acoplamiento no pertenece a esta estructura.")
        return relation.apply()

    def propagate(
        self,
        max_iterations: int = 100,
        atol: float = 1e-10,
        rtol: float = 1e-8,
    ) -> None:
        """Propagate acyclic mappings in order and cyclic mappings to a fixed point."""
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations debe ser un entero positivo.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or value < 0
            for value in (atol, rtol)
        ):
            raise ValueError("atol y rtol deben ser finitos y no negativos.")
        try:
            ordered_nodes = tuple(nx.topological_sort(self.topology))
        except nx.NetworkXUnfeasible:
            self._propagate_cyclic(max_iterations, atol, rtol)
            return

        for source_id in ordered_nodes:
            for target_id in self.topology.successors(source_id):
                for edge in self.topology[source_id][target_id].values():
                    self.apply_relation(edge["relation_id"])

    def _propagate_cyclic(
        self,
        max_iterations: int,
        atol: float,
        rtol: float,
    ) -> None:
        relations = tuple(
            relation
            for relation in self.relations.values()
            if isinstance(relation, StateMappingRelation)
        )
        target_keys = [
            (relation.target_id, relation.target_element_id)
            for relation in relations
        ]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError(
                "La propagación cíclica requiere una sola relación escritora "
                "por elemento destino."
            )

        values: dict[tuple[UUID, UUID], StateValue] = {}
        for relation in relations:
            for component_id, element_id in (
                (relation.source_id, relation.source_element_id),
                (relation.target_id, relation.target_element_id),
            ):
                component = self.systemic_space.get(component_id)
                if not isinstance(component, Component):
                    raise TypeError("Los extremos del acoplamiento deben ser Components.")
                element = component.system_state.elements.get(element_id)
                if isinstance(element, VariableElement):
                    value: StateValue = component.system_state.read_value(element_id)
                elif isinstance(element, DiscreteStateElement):
                    value = component.system_state.read_discrete_value(element_id)
                else:
                    raise TypeError("Cada relación debe conectar estados escalares.")
                values[(component_id, element_id)] = value

        converged = False
        for _ in range(max_iterations):
            next_values = dict(values)
            for relation in relations:
                source_key = (relation.source_id, relation.source_element_id)
                target_key = (relation.target_id, relation.target_element_id)
                next_values[target_key] = relation.evaluate(values[source_key])

            converged = True
            for key, next_value in next_values.items():
                previous_value = values[key]
                if (
                    isinstance(previous_value, (int, float, np.number))
                    and not isinstance(previous_value, bool)
                    and isinstance(next_value, (int, float, np.number))
                    and not isinstance(next_value, bool)
                ):
                    if not np.isclose(
                        float(previous_value),
                        float(next_value),
                        atol=atol,
                        rtol=rtol,
                    ):
                        converged = False
                        break
                elif previous_value != next_value:
                    converged = False
                    break
            values = next_values
            if converged:
                break

        if not converged:
            raise RuntimeError(
                f"La propagación cíclica no convergió en {max_iterations} iteraciones."
            )

        updates_by_state: dict[SystemState, dict[UUID, StateValue]] = {}
        for (component_id, element_id), value in values.items():
            component = self.systemic_space.get(component_id)
            if not isinstance(component, Component):
                raise TypeError("Los extremos del acoplamiento deben ser Components.")
            updates_by_state.setdefault(component.system_state, {})[element_id] = value
        validated_updates = {
            state: state.validate_state_updates(updates)
            for state, updates in updates_by_state.items()
        }
        for state, updates in validated_updates.items():
            state._commit_state_updates(updates)

class FlowBusRelation(SystemRelation):

    def __init__(
        self,
        systemic_space: SystemicSpace,
        source_id: UUID,
        target_id: UUID,
        port_id: Optional[UUID] = None,
        label: Optional[str] = None,
    ) -> None:
        source = systemic_space.get(source_id)
        target = systemic_space.get(target_id)
        supported_endpoints = (Component, System, SystemProcessor)
        if not isinstance(source, supported_endpoints):
            raise TypeError(
                "El origen del flujo debe ser Component, System o SystemProcessor."
            )
        if not isinstance(target, supported_endpoints):
            raise TypeError(
                "El destino del flujo debe ser Component, System o SystemProcessor."
            )
        if port_id is not None and not isinstance(systemic_space.get(port_id), Port):
            raise KeyError("port_id debe referir a un Port del SystemicSpace.")

        super().__init__(systemic_space, source_id, target_id, label)
        self.identificator.entity = Ontology.EntityType.FLOW_BUS_RELATION
        self.port_id = port_id

class FlowBusStructure(SystemStructure):

    def __init__(self, systemic_space: SystemicSpace) -> None:
        super().__init__(systemic_space)
        self.identificator.entity = Ontology.EntityType.FLOW_BUS_STRUCTURE

    def add_relation(self, relation_id: UUID) -> None:
        relation = self.systemic_space.get(relation_id)
        if not isinstance(relation, FlowBusRelation):
            raise TypeError("FlowBusStructure solo admite FlowBusRelation.")
        if relation.port_id is not None:
            port = self.systemic_space.get(relation.port_id)
            if not isinstance(port, Port):
                raise KeyError("El Port de FlowBusRelation dejó de existir.")
            source = self.systemic_space.get(relation.source_id)
            if isinstance(source, SystemProcessor) and relation.port_id not in source.ports:
                raise ValueError("El Port debe pertenecer al processor de origen.")
            if isinstance(self.systemic_space.get(relation.target_id), SystemProcessor):
                if port.destination_processor != relation.target_id:
                    raise ValueError(
                        "El destino de la relación debe coincidir con el destino del Port."
                    )
        super().add_relation(relation_id)

    def ports_between(
        self,
        source_id: UUID,
        target_id: UUID,
    ) -> tuple[Port, ...]:
        relations = self.relations_between(source_id, target_id)
        ports = []
        for relation in relations:
            if not isinstance(relation, FlowBusRelation) or relation.port_id is None:
                continue
            port = self.systemic_space.get(relation.port_id)
            if not isinstance(port, Port):
                raise KeyError("El Port asociado a la relación dejó de existir.")
            ports.append(port)
        return tuple(ports)

    def advance(self, duration: float, now: float = 0.0) -> dict[UUID, float]:
        """Transfer all continuous flows simultaneously, conserving each quantity."""
        if isinstance(duration, bool) or not isinstance(
            duration, (int, float, np.number)
        ):
            raise TypeError("duration debe ser numérico.")
        if isinstance(now, bool) or not isinstance(now, (int, float, np.number)):
            raise TypeError("now debe ser numérico.")
        interval = float(duration)
        simulation_time = float(now)
        if not isfinite(interval) or interval <= 0:
            raise ValueError("duration debe ser finito y positivo.")
        if not isfinite(simulation_time) or simulation_time < 0:
            raise ValueError("now debe ser finito y no negativo.")

        flow_relations = tuple(
            relation
            for relation in self.relations.values()
            if isinstance(relation, ContinuousFlowRelation)
        )
        if not flow_relations:
            return {}

        snapshots: dict[tuple[SystemState, UUID], float] = {}
        transfers: dict[UUID, float] = {}
        deltas: dict[tuple[SystemState, UUID], float] = {}
        outflows: dict[tuple[SystemState, UUID], float] = {}
        for relation in flow_relations:
            source = self.systemic_space.get(relation.source_id)
            target = self.systemic_space.get(relation.target_id)
            if not isinstance(source, SystemProcessor) or not isinstance(
                target, SystemProcessor
            ):
                raise TypeError("Los extremos del flujo continuo deben ser processors.")
            if source.system_state is None or target.system_state is None:
                raise RuntimeError("Ambos processors deben tener un SystemState vinculado.")
            source_key = (source.system_state, relation.source_element_id)
            target_key = (target.system_state, relation.target_element_id)
            for state, element_id in (source_key, target_key):
                if (state, element_id) not in snapshots:
                    snapshots[(state, element_id)] = state.read_value(element_id)

            source_value = snapshots[source_key]
            target_value = snapshots[target_key]
            if source_value < 0 or target_value < 0:
                raise ValueError("Los inventarios de flujo no pueden ser negativos.")
            rate = relation.rate
            evaluated_rate = (
                rate(simulation_time, source_value, target_value)
                if callable(rate)
                else rate
            )
            if isinstance(evaluated_rate, bool) or not isinstance(
                evaluated_rate, (int, float, np.number)
            ):
                raise TypeError("La tasa de flujo debe ser numérica.")
            if not np.isfinite(evaluated_rate) or evaluated_rate < 0:
                raise ValueError("La tasa de flujo debe ser finita y no negativa.")
            amount = float(evaluated_rate) * interval
            if not isfinite(amount):
                raise ValueError("La cantidad transferida debe ser finita.")
            transfers[relation.identificator.id] = amount
            outflows[source_key] = outflows.get(source_key, 0.0) + amount
            deltas[source_key] = deltas.get(source_key, 0.0) - amount
            deltas[target_key] = deltas.get(target_key, 0.0) + amount

        for key, amount in outflows.items():
            available = snapshots[key]
            if amount > available:
                raise ValueError(
                    "La cantidad solicitada excede el inventario disponible."
                )

        updates_by_state: dict[SystemState, dict[UUID, float]] = {}
        for (state, element_id), delta in deltas.items():
            updates_by_state.setdefault(state, {})[element_id] = (
                snapshots[(state, element_id)] + delta
            )
        validated_updates = {
            state: state.validate_values(updates)
            for state, updates in updates_by_state.items()
        }
        for state, updates in validated_updates.items():
            state._commit_validated_values(updates)
        return transfers

class ContinuousFlowRelation(FlowBusRelation):
    """Directed quantity transfer at a rate measured in quantity units per second."""

    def __init__(
        self,
        systemic_space: SystemicSpace,
        source_processor_id: UUID,
        source_element_id: UUID,
        target_processor_id: UUID,
        target_element_id: UUID,
        rate: Union[float, Callable[[float, float, float], float]],
        label: Optional[str] = None,
        port_id: Optional[UUID] = None,
    ) -> None:
        source = systemic_space.get(source_processor_id)
        target = systemic_space.get(target_processor_id)
        if not isinstance(source, SystemProcessor) or not isinstance(
            target, SystemProcessor
        ):
            raise TypeError("Un flujo continuo requiere processors como extremos.")
        if source.system_state is None or target.system_state is None:
            raise RuntimeError("Los processors requieren SystemState antes del flujo.")
        if not isinstance(source.system_state.elements.get(source_element_id), VariableElement):
            raise TypeError("El origen del flujo debe ser una variable numérica.")
        if not isinstance(target.system_state.elements.get(target_element_id), VariableElement):
            raise TypeError("El destino del flujo debe ser una variable numérica.")
        if callable(rate):
            flow_rate = rate
        elif isinstance(rate, (int, float, np.number)) and not isinstance(rate, bool):
            if not np.isfinite(rate) or rate < 0:
                raise ValueError("rate debe ser finito y no negativo.")
            flow_rate = float(rate)
        else:
            raise TypeError("rate debe ser una tasa numérica o una función de estado.")
        super().__init__(
            systemic_space,
            source_processor_id,
            target_processor_id,
            port_id=port_id,
            label=label,
        )
        self.identificator.entity = Ontology.EntityType.FLOW_BUS_RELATION
        self.source_element_id = source_element_id
        self.target_element_id = target_element_id
        self.rate = flow_rate

# ======================================
# SYSTEMS SEMANTIC DOMAIN
# =====================================

class SemanticEntityKind(Enum):

    COMPONENT = auto()
    SYSTEM = auto()

class SemanticValueType(Enum):
    
    ANY = auto()
    NUMBER = auto()
    INTEGER = auto()
    STRING = auto()
    BOOLEAN = auto()

@dataclass(frozen=True)
class SemanticAttribute:
    name: str
    value_type: SemanticValueType = SemanticValueType.ANY
    default: Any = None
    required: bool = False
    choices: tuple[Any, ...] = ()
    unit_id: Optional[UUID] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("El atributo semántico requiere un nombre.")
        if not isinstance(self.value_type, SemanticValueType):
            raise TypeError("value_type debe ser SemanticValueType.")
        if self.choices:
            for choice in self.choices:
                self.validate(choice)
        if self.default is not None:
            self.validate(self.default)
        elif self.required:
            raise ValueError("Un atributo requerido debe declarar un valor predeterminado.")

    def validate(self, value: Any) -> Any:
        valid = True
        if value is None:
            valid = not self.required
        elif self.value_type is SemanticValueType.NUMBER:
            valid = isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
            valid = valid and isfinite(float(value))
        elif self.value_type is SemanticValueType.INTEGER:
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif self.value_type is SemanticValueType.STRING:
            valid = isinstance(value, str)
        elif self.value_type is SemanticValueType.BOOLEAN:
            valid = isinstance(value, bool)
        if not valid:
            raise TypeError(
                f"El atributo '{self.name}' requiere un valor de tipo "
                f"{self.value_type.name}."
            )
        if self.choices and value not in self.choices:
            raise ValueError(
                f"El valor de '{self.name}' debe pertenecer a {self.choices!r}."
            )
        return deepcopy(value)

@dataclass
class SemanticArchetype:
    name: str
    kind: SemanticEntityKind
    parent_id: Optional[UUID] = None
    id: UUID = field(default_factory=uuid4)
    attributes: dict[str, SemanticAttribute] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)

@dataclass
class SemanticSet:
    name: str
    id: UUID = field(default_factory=uuid4)
    members: set[UUID] = field(default_factory=set)

@dataclass(frozen=True)
class QualitativeRegion:
    lower: float
    upper: float
    category: str
    include_lower: bool = True
    include_upper: bool = False

@dataclass
class QualitativeDomain:
    name: str
    categories: tuple[str, ...]
    id: UUID = field(default_factory=uuid4)
    value_map: dict[Any, str] = field(default_factory=dict)
    regions: list[QualitativeRegion] = field(default_factory=list)

@dataclass
class QualitativeAutomaton:
    name: str
    states: frozenset[str]
    input_symbols: frozenset[str]
    initial_state: str
    transitions: dict[tuple[str, str], str] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)

    def step(self, state: str, input_symbol: str) -> str:
        if state not in self.states:
            raise KeyError(f"El estado '{state}' no pertenece al autómata.")
        if input_symbol not in self.input_symbols:
            raise KeyError(f"La entrada '{input_symbol}' no pertenece al autómata.")
        try:
            return self.transitions[(state, input_symbol)]
        except KeyError as error:
            raise ValueError(
                f"No hay transición para ({state!r}, {input_symbol!r})."
            ) from error

@dataclass(frozen=True)
class SemanticUnit:
    name: str
    symbol: str
    dimension: tuple[tuple[str, Fraction], ...]
    scale: float = 1.0
    offset: float = 0.0
    id: UUID = field(default_factory=uuid4)

class SystemSemantics(Entity):

    def __init__(self, systemic_space: SystemicSpace, owner_system_id: Optional[UUID] = None, label: Optional[str] = None,) -> None:
        
        super().__init__(systemic_space)
        self.identificator = Identification(entity=Ontology.EntityType.SYSTEM_SEMANTICS, label=label,)

        self.owner_system_id = owner_system_id

        self.archetypes: dict[UUID, SemanticArchetype] = {}
        self.instances: dict[UUID, Entity] = {}
        
        self.sets: dict[UUID, SemanticSet] = {}
        self.disjoint_set_pairs: set[frozenset[UUID]] = set()
        self.parts_by_whole: dict[UUID, set[UUID]] = {}
        
        self.qualitative_domains: dict[UUID, QualitativeDomain] = {}
        
        self.automata: dict[UUID, QualitativeAutomaton] = {}
        
        self.base_dimensions: set[str] = set()
        self.units: dict[UUID, SemanticUnit] = {}
        self.state_variable_units: dict[UUID, dict[UUID, UUID]] = {}
        
        self.space_auto_registry()

    @staticmethod
    def _validate_name(name: str, subject: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{subject} requiere un nombre no vacío.")
        return name.strip()

    @staticmethod
    def _validate_tag(tag: str) -> str:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("Las etiquetas deben ser texto no vacío.")
        normalized = tag.strip()
        if any(not segment.strip() for segment in normalized.split(">")):
            raise ValueError("Las etiquetas jerárquicas no pueden tener segmentos vacíos.")
        return normalized

# ARCHETYPES AND INSTANCES

    def define_archetype(self, name: str, kind: SemanticEntityKind,*, parent_id: Optional[UUID] = None, 
        tags: tuple[str, ...] = (),) -> SemanticArchetype:

        name = self._validate_name(name, "El arquetipo")
        if not isinstance(kind, SemanticEntityKind):
            raise TypeError("kind debe ser SemanticEntityKind.")
        parent = self.archetypes.get(parent_id) if parent_id is not None else None
        if parent_id is not None and parent is None:
            raise KeyError("El arquetipo padre no pertenece a este dominio semántico.")
        if parent is not None and parent.kind is not kind:
            raise TypeError("Un arquetipo solo puede heredar de otro del mismo tipo.")
        
        archetype = SemanticArchetype(name=name, kind=kind, parent_id=parent_id, 
            tags={self._validate_tag(tag) for tag in tags},)
        
        self.archetypes[archetype.id] = archetype
        return archetype

    def _archetype_attributes(self, archetype_id: UUID, resolving: Optional[set[UUID]] = None,
        ) -> dict[str, SemanticAttribute]:
        
        archetype = self.archetypes.get(archetype_id)
        if archetype is None:
            raise KeyError("El arquetipo no pertenece a este dominio semántico.")
        resolving = set() if resolving is None else resolving
        if archetype_id in resolving:
            raise ValueError("Se detectó una herencia circular de arquetipos.")
        inherited = (
            self._archetype_attributes(archetype.parent_id, resolving | {archetype_id})
            if archetype.parent_id is not None
            else {}
        )
        return {**inherited, **archetype.attributes}

    def _archetype_tags(self, archetype_id: UUID) -> set[str]:
        archetype = self.archetypes.get(archetype_id)
        if archetype is None:
            raise KeyError("El arquetipo no pertenece a este dominio semántico.")
        tags = set(archetype.tags)
        if archetype.parent_id is not None:
            tags.update(self._archetype_tags(archetype.parent_id))
        return tags

    def add_attribute(
        self,
        archetype_id: UUID,
        attribute: SemanticAttribute,
    ) -> None:
        archetype = self.archetypes.get(archetype_id)
        if archetype is None:
            raise KeyError("El arquetipo no pertenece a este dominio semántico.")
        if not isinstance(attribute, SemanticAttribute):
            raise TypeError("attribute debe ser SemanticAttribute.")
        if attribute.unit_id is not None and attribute.unit_id not in self.units:
            raise KeyError("La unidad del atributo no pertenece a este dominio.")
        archetype.attributes[attribute.name] = attribute

    def define_archetype_from_instance(
        self,
        name: str,
        entity_id: UUID,
        *,
        tags: tuple[str, ...] = (),
    ) -> SemanticArchetype:
        entity = self.systemic_space.get(entity_id)
        if not isinstance(entity, (Component, System)):
            raise TypeError("Solo se pueden usar System o Component como prototipos.")
        kind = (
            SemanticEntityKind.SYSTEM
            if isinstance(entity, System)
            else SemanticEntityKind.COMPONENT
        )
        combined_tags = set(entity.semantic_tags)
        combined_tags.update(self._validate_tag(tag) for tag in tags)
        archetype = self.define_archetype(name, kind, tags=tuple(combined_tags))
        for attribute_name, value in entity.semantic_attributes.items():
            value_type = (
                SemanticValueType.BOOLEAN
                if isinstance(value, bool)
                else SemanticValueType.INTEGER
                if isinstance(value, int)
                else SemanticValueType.NUMBER
                if isinstance(value, (float, np.number))
                else SemanticValueType.STRING
                if isinstance(value, str)
                else SemanticValueType.ANY
            )
            self.add_attribute(
                archetype.id,
                SemanticAttribute(
                    name=attribute_name,
                    value_type=value_type,
                    default=deepcopy(value),
                ),
            )
        entity.semantic_archetype_id = archetype.id
        entity.semantic_domain_id = self.identificator.id
        self.instances[entity_id] = entity
        return archetype

    def instantiate(
        self,
        archetype_id: UUID,
        *,
        label: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
        tags: tuple[str, ...] = (),
    ) -> Component | System:
        archetype = self.archetypes.get(archetype_id)
        if archetype is None:
            raise KeyError("El arquetipo no pertenece a este dominio semántico.")
        specifications = self._archetype_attributes(archetype_id)
        values = {
            name: specification.validate(deepcopy(specification.default))
            for name, specification in specifications.items()
            if specification.default is not None
        }
        overrides = dict(attributes or {})
        unknown = overrides.keys() - specifications.keys()
        if unknown:
            raise KeyError(f"Atributos no declarados en el arquetipo: {sorted(unknown)}.")
        for name, value in overrides.items():
            values[name] = specifications[name].validate(value)
        missing = [
            name
            for name, specification in specifications.items()
            if specification.required and name not in values
        ]
        if missing:
            raise ValueError(f"Faltan atributos requeridos: {', '.join(sorted(missing))}.")

        instance_tags = self._archetype_tags(archetype_id)
        instance_tags.update(self._validate_tag(tag) for tag in tags)
        instance = (
            System(self.systemic_space, label=label or archetype.name)
            if archetype.kind is SemanticEntityKind.SYSTEM
            else Component(self.systemic_space, label=label or archetype.name)
        )
        instance.semantic_domain_id = self.identificator.id
        instance.semantic_archetype_id = archetype_id
        instance.semantic_attributes = deepcopy(values)
        instance.semantic_tags = instance_tags
        self.instances[instance.identificator.id] = instance
        return instance

    def replicate(self, entity_id: UUID,*, label: Optional[str] = None, attributes: Optional[dict[str, Any]] = None,
        tags: tuple[str, ...] = (),) -> Component | System:

        entity = self.systemic_space.get(entity_id)
        if not isinstance(entity, (Component, System)):
            raise TypeError("Solo se pueden replicar System o Component.")
        archetype_id = entity.semantic_archetype_id
        if archetype_id not in self.archetypes:
            archetype = self.define_archetype_from_instance(
                f"{entity.identificator.label or type(entity).__name__} archetype", entity_id,)
            archetype_id = archetype.id
        return self.instantiate(archetype_id, label=label, attributes=attributes, tags=tags,)

# CATEGORICAL TAGGING AND SEMANTIC LABELS

    def add_tag(self, entity_id: UUID, tag: str) -> None:
        entity = self.systemic_space.get(entity_id)
        if not isinstance(entity, Entity):
            raise KeyError("La entidad no existe en SystemicSpace.")
        entity.semantic_tags.add(self._validate_tag(tag))
        if entity_id in self.instances:
            self.instances[entity_id] = entity

    def remove_tag(self, entity_id: UUID, tag: str) -> None:
        entity = self.systemic_space.get(entity_id)
        if not isinstance(entity, Entity):
            raise KeyError("La entidad no existe en SystemicSpace.")
        entity.semantic_tags.discard(self._validate_tag(tag))

    def entities_with_tag(self, tag: str) -> tuple[Entity, ...]:
        normalized = self._validate_tag(tag)
        entities = (*self.systemic_space.source.component.values(), *self.systemic_space.source.system.values(),)
        return tuple(entity for entity in entities if normalized in entity.semantic_tags)

# SETS MANAGEMENT 

    def define_set(self, name: str) -> SemanticSet:
        semantic_set = SemanticSet(self._validate_name(name, "El conjunto"))
        self.sets[semantic_set.id] = semantic_set
        return semantic_set

    def add_member(self, set_id: UUID, entity_id: UUID) -> None:
        semantic_set = self.sets.get(set_id)
        if semantic_set is None:
            raise KeyError("El conjunto no pertenece a este dominio semántico.")
        if not isinstance(self.systemic_space.get(entity_id), Entity):
            raise KeyError("El miembro debe existir en SystemicSpace.")
        for disjoint_pair in self.disjoint_set_pairs:
            if set_id in disjoint_pair:
                other_id = next(item for item in disjoint_pair if item != set_id)
                if entity_id in self.sets[other_id].members:
                    raise ValueError("El miembro ya pertenece a un conjunto declarado disjunto.")
        semantic_set.members.add(entity_id)

    def declare_disjoint(self, first_id: UUID, second_id: UUID) -> None:
        first = self.sets.get(first_id)
        second = self.sets.get(second_id)
        if first is None or second is None:
            raise KeyError("Ambos conjuntos deben pertenecer a este dominio semántico.")
        if first_id == second_id:
            raise ValueError("Un conjunto no puede ser disjunto de sí mismo.")
        if first.members & second.members:
            raise ValueError("Los conjuntos ya comparten miembros.")
        self.disjoint_set_pairs.add(frozenset((first_id, second_id)))

    def intersection(self, first_id: UUID, second_id: UUID) -> frozenset[UUID]:
        first = self.sets.get(first_id)
        second = self.sets.get(second_id)
        if first is None or second is None:
            raise KeyError("Ambos conjuntos deben pertenecer a este dominio semántico.")
        return frozenset(first.members & second.members)

    def define_partition(
        self,
        universe_id: UUID,
        partition_ids: tuple[UUID, ...],
    ) -> None:
        universe = self.sets.get(universe_id)
        partitions = [self.sets.get(set_id) for set_id in partition_ids]
        if universe is None or not partitions or any(item is None for item in partitions):
            raise KeyError("El universo y las particiones deben existir en este dominio.")
        if len(set(partition_ids)) != len(partition_ids):
            raise ValueError("La partición contiene conjuntos duplicados.")
        members: set[UUID] = set()
        for partition in partitions:
            if partition is None:
                raise RuntimeError("Partición inválida.")
            if not partition.members <= universe.members:
                raise ValueError("Cada conjunto de la partición debe ser subconjunto del universo.")
            if members & partition.members:
                raise ValueError("Los conjuntos de una partición deben ser disjuntos.")
            members.update(partition.members)
        if members != universe.members:
            raise ValueError("Los conjuntos no cubren exactamente el universo.")
        for index, first_id in enumerate(partition_ids):
            for second_id in partition_ids[index + 1:]:
                self.disjoint_set_pairs.add(frozenset((first_id, second_id)))

# MEREOLOGICAL MANAGEMENT

    def add_part(self, whole_id: UUID, part_id: UUID) -> None:
        whole = self.systemic_space.get(whole_id)
        part = self.systemic_space.get(part_id)
        if not isinstance(whole, Entity) or not isinstance(part, Entity):
            raise KeyError("El todo y la parte deben existir en SystemicSpace.")
        if whole_id == part_id or self.is_part_of(whole_id, part_id):
            raise ValueError("La relación parte-todo produciría un ciclo mereológico.")
        self.parts_by_whole.setdefault(whole_id, set()).add(part_id)

    def is_part_of(self, part_id: UUID, whole_id: UUID) -> bool:
        pending = list(self.parts_by_whole.get(whole_id, ()))
        visited: set[UUID] = set()
        while pending:
            current = pending.pop()
            if current == part_id:
                return True
            if current not in visited:
                visited.add(current)
                pending.extend(self.parts_by_whole.get(current, ()))
        return False

# QUALITATIVE STATES

    def define_qualitative_domain(
        self,
        name: str,
        categories: tuple[str, ...],
    ) -> QualitativeDomain:
        normalized = tuple(self._validate_name(item, "La categoría") for item in categories)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("El dominio cualitativo requiere categorías únicas.")
        domain = QualitativeDomain(
            self._validate_name(name, "El dominio cualitativo"),
            normalized,
        )
        self.qualitative_domains[domain.id] = domain
        return domain

    def map_qualitative_value(
        self,
        domain_id: UUID,
        value: Any,
        category: str,
    ) -> None:
        domain = self.qualitative_domains.get(domain_id)
        if domain is None:
            raise KeyError("El dominio cualitativo no pertenece a este sistema.")
        if category not in domain.categories:
            raise KeyError("La categoría no pertenece al dominio cualitativo.")
        try:
            domain.value_map[value] = category
        except TypeError as error:
            raise TypeError("El valor mapeado debe ser hashable.") from error

    def map_qualitative_interval(
        self,
        domain_id: UUID,
        lower: float,
        upper: float,
        category: str,
        *,
        include_lower: bool = True,
        include_upper: bool = False,
    ) -> None:
        domain = self.qualitative_domains.get(domain_id)
        if domain is None:
            raise KeyError("El dominio cualitativo no pertenece a este sistema.")
        if category not in domain.categories:
            raise KeyError("La categoría no pertenece al dominio cualitativo.")
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError("El intervalo requiere límites finitos lower < upper.")
        for region in domain.regions:
            overlap = max(lower, region.lower) < min(upper, region.upper)
            touching = (
                upper == region.lower and include_upper and region.include_lower
            ) or (
                region.upper == lower and region.include_upper and include_lower
            )
            if overlap or touching:
                raise ValueError("Los intervalos cualitativos no pueden solaparse.")
        domain.regions.append(
            QualitativeRegion(lower, upper, category, include_lower, include_upper)
        )

    def classify_value(self, domain_id: UUID, value: Any) -> str:
        domain = self.qualitative_domains.get(domain_id)
        if domain is None:
            raise KeyError("El dominio cualitativo no pertenece a este sistema.")
        try:
            category = domain.value_map.get(value)
        except TypeError:
            category = None
        if category is not None:
            return category
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            for region in domain.regions:
                above_lower = value > region.lower or (
                    region.include_lower and value == region.lower
                )
                below_upper = value < region.upper or (
                    region.include_upper and value == region.upper
                )
                if above_lower and below_upper:
                    return region.category
        raise KeyError(f"El valor {value!r} no tiene categoría en '{domain.name}'.")

    def define_automaton(
        self,
        name: str,
        states: tuple[str, ...],
        input_symbols: tuple[str, ...],
        initial_state: str,
        transitions: dict[tuple[str, str], str],
    ) -> QualitativeAutomaton:
        state_set = frozenset(states)
        input_set = frozenset(input_symbols)
        if not state_set or len(state_set) != len(states):
            raise ValueError("El autómata requiere estados únicos y no vacíos.")
        if initial_state not in state_set:
            raise ValueError("El estado inicial debe pertenecer al autómata.")
        if len(input_set) != len(input_symbols):
            raise ValueError("Las entradas del autómata deben ser únicas.")
        for (source, input_symbol), target in transitions.items():
            if source not in state_set or target not in state_set:
                raise ValueError("Las transiciones deben referir a estados declarados.")
            if input_symbol not in input_set:
                raise ValueError("Las transiciones deben referir a entradas declaradas.")
        automaton = QualitativeAutomaton(
            name=self._validate_name(name, "El autómata"),
            states=state_set,
            input_symbols=input_set,
            initial_state=initial_state,
            transitions=dict(transitions),
        )
        self.automata[automaton.id] = automaton
        return automaton

    def add_base_dimension(self, name: str) -> None:
        self.base_dimensions.add(self._validate_name(name, "La dimensión base"))

    def define_unit(
        self,
        name: str,
        symbol: str,
        dimension: dict[str, int | Fraction],
        *,
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> SemanticUnit:
        if not dimension:
            normalized_dimension: tuple[tuple[str, Fraction], ...] = ()
        else:
            if any(base not in self.base_dimensions for base in dimension):
                raise KeyError("La dimensión de unidad usa bases no registradas.")
            normalized_dimension = tuple(
                sorted(
                    (base, Fraction(exponent))
                    for base, exponent in dimension.items()
                    if exponent != 0
                )
            )
        if not isfinite(scale) or scale <= 0 or not isfinite(offset):
            raise ValueError("scale debe ser positivo y offset finito.")
        unit = SemanticUnit(
            self._validate_name(name, "La unidad"),
            self._validate_name(symbol, "El símbolo de unidad"),
            normalized_dimension,
            scale,
            offset,
        )
        self.units[unit.id] = unit
        return unit

    def bind_state_variable(
        self,
        state: SystemState,
        element_id: UUID,
        unit_id: UUID,
    ) -> None:
        if not isinstance(state, SystemState) or state.systemic_space is not self.systemic_space:
            raise ValueError("El estado debe pertenecer al mismo SystemicSpace.")
        if not isinstance(state.elements.get(element_id), VariableElement):
            raise KeyError("element_id debe identificar una variable del estado.")
        if unit_id not in self.units:
            raise KeyError("La unidad no pertenece a este dominio semántico.")
        self.state_variable_units.setdefault(state.identificator.id, {})[element_id] = unit_id

    def _dimension_of(self, expression: sym.Expr, state: SystemState,) -> dict[str, Fraction]:
        
        bindings = self.state_variable_units.get(state.identificator.id, {})
        symbols = {element.symbol: element for element in state.elements.values() if isinstance(element, VariableElement)}

        def combine(
            first: dict[str, Fraction],
            second: dict[str, Fraction],
            factor: Fraction,
        ) -> dict[str, Fraction]:
            result = dict(first)
            for base, exponent in second.items():
                result[base] = result.get(base, Fraction(0)) + factor * exponent
                if result[base] == 0:
                    del result[base]
            return result

        def visit(term: sym.Expr) -> dict[str, Fraction]:
            if term.is_Number:
                return {}
            if term.is_Symbol:
                element = symbols.get(term)
                if element is None:
                    raise ValueError(f"El símbolo '{term}' no pertenece al estado.")
                unit_id = bindings.get(element.identificator.id)
                if unit_id is None:
                    raise ValueError(
                        f"La variable '{element.name}' no tiene unidad semántica asignada."
                    )
                return dict(self.units[unit_id].dimension)
            if term.is_Add:
                dimensions = [visit(argument) for argument in term.args]
                if any(dimension != dimensions[0] for dimension in dimensions[1:]):
                    raise ValueError("Suma/resta dimensionalmente incompatible.")
                return dimensions[0]
            if term.is_Mul:
                result: dict[str, Fraction] = {}
                for argument in term.args:
                    result = combine(result, visit(argument), Fraction(1))
                return result
            if term.is_Pow:
                base, exponent = term.args
                if not exponent.is_Rational:
                    raise ValueError("El exponente dimensional debe ser racional.")
                power = Fraction(int(exponent.p), int(exponent.q))
                return {
                    dimension: dimension_power * power
                    for dimension, dimension_power in visit(base).items()
                    if dimension_power * power != 0
                }
            if term.is_Function:
                if any(visit(argument) for argument in term.args):
                    raise ValueError(
                        f"La función '{term.func}' requiere argumentos adimensionales."
                    )
                return {}
            raise TypeError(f"No se admite '{type(term).__name__}' en análisis dimensional.")

        if not isinstance(expression, sym.Expr):
            raise TypeError("La expresión debe ser una expresión algebraica de SymPy.")
        return visit(expression)

    def validate_expression_dimensions(
        self,
        state: SystemState,
        expression: sym.Expr,
    ) -> tuple[tuple[str, Fraction], ...]:
        if state.systemic_space is not self.systemic_space:
            raise ValueError("El estado debe pertenecer al mismo SystemicSpace.")
        return tuple(sorted(self._dimension_of(expression, state).items()))

    def compile_state(self, state: SystemState, targets: dict[sym.Symbol, sym.Expr], 
        custom_mappings: Optional[dict[str, Callable[..., Any]]] = None,) -> CompiledStateKernel:
        for target_symbol, expression in targets.items():
            expression_dimension = self._dimension_of(expression, state)
            target = next(
                (
                    element
                    for element in state.elements.values()
                    if isinstance(element, VariableElement)
                    and element.symbol == target_symbol
                ),
                None,
            )
            if target is not None:
                unit_id = self.state_variable_units.get(
                    state.identificator.id,
                    {},
                ).get(target.identificator.id)
                if unit_id is None:
                    raise ValueError(
                        f"La variable objetivo '{target.name}' no tiene unidad semántica."
                    )
                if expression_dimension != dict(self.units[unit_id].dimension):
                    raise ValueError(
                        f"La expresión para '{target.name}' no coincide con su dimensión."
                    )
        return state.compile_state(targets, custom_mappings=custom_mappings)

# ======================
# SYSTEM
# ======================

class System(Entity):

# System Attributes

    def __init__(self, systemic_space: SystemicSpace, label: Optional[str] = None,) -> None:

        super().__init__(systemic_space)
        self.identificator: Identification = Identification(label=label or "System", entity=Ontology.EntityType.SYSTEM,)
        self.space_auto_registry()

        self.structures : dict[UUID, SystemStructure] = {}
        self.states : dict[UUID, SystemState] = {}
        self.processors : dict[UUID, SystemProcessor] = {}
        self.processes: dict[UUID, Process] = {}
        self.semantics_domains: dict[UUID, SystemSemantics] = {}

        self.semantics = SystemSemantics(systemic_space, owner_system_id=self.identificator.id, label=f"{self.identificator.label} semantics",)
        self.semantics_domains[self.semantics.identificator.id] = self.semantics

        self._simpy_environment: Optional[sim.Environment] = None
        self._simpy_scheduled: set[tuple[UUID, UUID]] = set()
        self._condition_edges: dict[tuple[UUID, UUID], bool] = {}
        self._condition_poll_scheduled = False
        self._condition_poll_interval: Optional[float] = None
        self._simulation_time = 0.0
        self._calendar_clock: Optional[SimulationClock] = None
        self.history: Optional[TemporalHistory] = None

        self.SYSTEMIC_ATRIBUTES : dict[Any, Any] = {
            SystemStructure: self.structures,
            SystemState: self.states,
            SystemProcessor: self.processors,
            Process: self.processes,
            SystemSemantics: self.semantics_domains,
        }

    def suscribe_systemic_description(self, description : UUID) -> None:
        description_entity = self.systemic_space.get(description)
        if description_entity is None:
            raise KeyError("La entidad no existe en el SystemicSpace.")
        description_type = type(description_entity)
        description_funnel = next((self.SYSTEMIC_ATRIBUTES[entity_type] for entity_type in description_type.__mro__ if entity_type in self.SYSTEMIC_ATRIBUTES), None,)
        if description_funnel is None:
            raise TypeError(f"{description_type.__name__} no es una descripción de System.")
        if isinstance(description_entity, Process) and (
            description_entity.system_id != self.identificator.id):
            raise ValueError("El Process fue declarado para otro System.")
        if isinstance(description_entity, SystemSemantics):
            if description_entity.owner_system_id not in (None, self.identificator.id):
                raise ValueError("SystemSemantics ya pertenece a otro System.")
            description_entity.owner_system_id = self.identificator.id
        if isinstance(description_entity, SystemStructure):
            description_entity.bind_to_system(self.identificator.id)
        description_funnel[description] = description_entity
        if isinstance(description_entity, SystemProcessor):
            description_entity.system_id = self.identificator.id
        if isinstance(description_entity, SystemState):
            self._attach_history_state(description_entity)
        elif isinstance(description_entity, SystemStructure):
            for component in description_entity.components.values():
                self._attach_history_state(component.system_state)

    @property
    def simulation_time(self) -> float:
        if self._simpy_environment is not None:
            return float(self._simpy_environment.now)
        return self._simulation_time

    @property
    def simulation_datetime(self) -> Optional[datetime]:
        if self._calendar_clock is None:
            return None
        return self._calendar_clock.to_datetime(self.simulation_time)

    def enable_history(self) -> TemporalHistory:
        if self.history is None:
            self.history = TemporalHistory()
        for state in self._states_for_history():
            self._attach_history_state(state)
        return self.history

    def _states_for_history(self) -> tuple[SystemState, ...]:
        states = dict(self.states)
        for structure in self.structures.values():
            for component in structure.components.values():
                state = component.system_state
                states[state.identificator.id] = state
        return tuple(states.values())

    def _attach_history_state(self, state: SystemState) -> None:
        if self.history is not None:
            self.history.attach_state(state, lambda: self.simulation_time)

    def record_discrete(
        self,
        state_id: UUID,
        element_id: UUID,
        value: DiscreteValue,
        simulation_time: Optional[float] = None,
    ) -> DiscreteStateEvent:
        if self.history is None:
            raise RuntimeError("Activa el historial con enable_history() primero.")
        state = next(
            (
                candidate
                for candidate in self._states_for_history()
                if candidate.identificator.id == state_id
            ),
            None,
        )
        if state is None:
            raise KeyError("state_id no pertenece a este System.")
        if element_id not in state.elements:
            raise KeyError("element_id no pertenece al SystemState indicado.")
        now = self.simulation_time if simulation_time is None else simulation_time
        return self.history.record_discrete(state_id, element_id, value, now)

# System Processes Control related Functions

    def run_process(self, process_id: UUID, trigger_id: UUID, token_id: Optional[UUID] = None, now: float = 0.0,) -> bool:
        process = self.processes.get(process_id)
        if process is None:
            raise KeyError("El Process no pertenece a este System.")
        if not np.isfinite(now):
            raise ValueError("now debe ser un tiempo de simulación finito.")
        self._simulation_time = float(now)
        return process.execute(trigger_id, token_id=token_id, now=now)

    def trigger_processes(self, kind: ProcessTriggerKind, processor_id: Optional[UUID] = None, 
    token_id: Optional[UUID] = None, now: float = 0.0,) -> int:
        if not np.isfinite(now):
            raise ValueError("now debe ser un tiempo de simulación finito.")
        self._simulation_time = float(now)
        executed = 0

        for process in tuple(self.processes.values()):
            for trigger_id in tuple(process.trigger_ids):
                trigger = self.systemic_space.get(trigger_id)
                if not isinstance(trigger, ProcessTrigger) or trigger.kind is not kind:
                    continue
                if kind is ProcessTriggerKind.TOKEN_RECEIVED:
                    if trigger.processor_id != processor_id:
                        continue
                    if token_id is None:
                        raise ValueError("TOKEN_RECEIVED requiere token_id.")
                if kind is ProcessTriggerKind.INTERVAL:
                    continue
                if kind is ProcessTriggerKind.CALENDAR:
                    continue
                if kind is ProcessTriggerKind.CONDITION:
                    if trigger.condition_id is None:
                        raise ValueError("CONDITION requiere condition_id.")
                    condition = self.systemic_space.get(trigger.condition_id)
                    if not isinstance(condition, ProcessCondition):
                        raise TypeError("condition_id debe referir a ProcessCondition.")
                    active = condition.evaluate(
                        self,
                        None,
                        now,
                    )
                    edge_key = (process.identificator.id, trigger_id)
                    previous = self._condition_edges.get(edge_key, False)
                    self._condition_edges[edge_key] = active
                    if not active or previous:
                        continue
                if process.execute(trigger_id, token_id=token_id, now=now):
                    executed += 1
        return executed

    def run_condition_processes(self, now: float = 0.0) -> int:
        return self.trigger_processes(
            ProcessTriggerKind.CONDITION,
            now=now,
        )

    def run_simpy(
        self,
        until: float,
        environment: Optional[sim.Environment] = None,
        calendar_origin: Optional[datetime] = None,
        condition_interval: float = 1.0,
    ) -> sim.Environment:
        if not np.isfinite(until) or until <= 0:
            raise ValueError("until debe ser un tiempo finito mayor que cero.")
        if not np.isfinite(condition_interval) or condition_interval <= 0:
            raise ValueError("condition_interval debe ser finito y positivo.")
        if environment is None:
            environment = sim.Environment()
        requested_clock = (
            SimulationClock(calendar_origin)
            if calendar_origin is not None
            else None
        )
        if until <= environment.now:
            raise ValueError("until debe ser mayor que el tiempo actual de SimPy.")
        if calendar_origin is not None:
            if (
                self._calendar_clock is not None
                and requested_clock is not None
                and self._calendar_clock.origin != requested_clock.origin
            ):
                raise ValueError("calendar_origin no puede cambiar durante la simulación.")
        if environment is not self._simpy_environment:
            self._simpy_environment = environment
            self._simpy_scheduled.clear()
            self._condition_edges.clear()
            self._condition_poll_scheduled = False
            self._condition_poll_interval = None
        if requested_clock is not None:
            self._calendar_clock = requested_clock

        scheduled_intervals: list[tuple[Process, UUID, float, float]] = []
        scheduled_calendar: list[tuple[Process, UUID, float]] = []
        has_condition_triggers = False
        for process in self.processes.values():
            for trigger_id in process.trigger_ids:
                if (process.identificator.id, trigger_id) in self._simpy_scheduled:
                    continue
                trigger = self.systemic_space.get(trigger_id)
                if isinstance(trigger, ProcessTrigger) and trigger.kind is ProcessTriggerKind.INTERVAL:
                    if trigger.interval is None:
                        raise ValueError("Un trigger INTERVAL requiere interval.")
                    initial_delay = (
                        trigger.interval
                        if trigger.initial_delay is None
                        else trigger.initial_delay
                    )
                    scheduled_intervals.append(
                        (process, trigger_id, trigger.interval, initial_delay)
                    )
                elif (
                    isinstance(trigger, ProcessTrigger)
                    and trigger.kind is ProcessTriggerKind.CALENDAR
                ):
                    if trigger.scheduled_at is None:
                        raise ValueError("Un trigger CALENDAR requiere scheduled_at.")
                    if self._calendar_clock is None:
                        raise ValueError(
                            "CALENDAR requiere calendar_origin al iniciar la simulación."
                        )
                    scheduled_time = self._calendar_clock.to_simulation_time(
                        trigger.scheduled_at
                    )
                    if scheduled_time < environment.now:
                        raise ValueError("El evento CALENDAR está en el pasado.")
                    scheduled_calendar.append(
                        (process, trigger_id, scheduled_time - environment.now)
                    )
                elif (
                    isinstance(trigger, ProcessTrigger)
                    and trigger.kind is ProcessTriggerKind.CONDITION
                ):
                    has_condition_triggers = True
        if (
            has_condition_triggers
            and self._condition_poll_scheduled
            and self._condition_poll_interval != float(condition_interval)
        ):
            raise ValueError(
                "condition_interval no puede cambiar durante una simulación."
            )

        def repeat(process: Process, trigger_id: UUID, interval: float, initial_delay: float,) -> Any:
            yield environment.timeout(initial_delay)
            while True:
                process.execute(trigger_id, now=environment.now)
                yield environment.timeout(interval)

        for process, trigger_id, interval, initial_delay in scheduled_intervals:
            environment.process(
                repeat(process, trigger_id, interval, initial_delay)
            )
            self._simpy_scheduled.add((process.identificator.id, trigger_id))
        for process, trigger_id, delay in scheduled_calendar:
            def at_calendar_time(
                scheduled_process: Process,
                scheduled_trigger_id: UUID,
                scheduled_delay: float,
            ) -> Any:
                yield environment.timeout(scheduled_delay)
                scheduled_process.execute(
                    scheduled_trigger_id,
                    now=environment.now,
                )

            environment.process(
                at_calendar_time(process, trigger_id, delay)
            )
            self._simpy_scheduled.add((process.identificator.id, trigger_id))
        if has_condition_triggers and not self._condition_poll_scheduled:
            self._condition_poll_interval = float(condition_interval)

            def poll_conditions() -> Any:
                while True:
                    self.run_condition_processes(now=environment.now)
                    yield environment.timeout(condition_interval)

            environment.process(poll_conditions())
            self._condition_poll_scheduled = True
        environment.run(until=until)
        self._simulation_time = float(environment.now)
        return environment
    
# ==========================
# SYSTEMS SPACES
# ==========================

@dataclass
class Source:

    state_element : dict[UUID, StateElement] = field(default_factory = dict)
    process: dict[UUID, Process] = field(default_factory=dict)
    process_value: dict[UUID, ProcessValueRef] = field(default_factory=dict)
    process_expression: dict[UUID, ProcessExpression] = field(default_factory=dict)
    process_condition: dict[UUID, ProcessCondition] = field(default_factory=dict)
    process_trigger: dict[UUID, ProcessTrigger] = field(default_factory=dict)
    process_action: dict[UUID, ProcessAction] = field(default_factory=dict)

    variable_element : dict[UUID, VariableElement] = field(default_factory= dict)
    expression_element : dict[UUID, DependentExpressionElement] = field(default_factory = dict)

    system_state : dict[UUID, SystemState] = field(default_factory = dict)
    system_structure : dict[UUID, SystemStructure] = field(default_factory = dict)
    system_relation: dict[UUID, SystemRelation] = field(default_factory=dict)
    system_semantics : dict[UUID, SystemSemantics] = field(default_factory = dict)

    token : dict[UUID, Token] = field(default_factory = dict)
    port : dict[UUID, Port] = field(default_factory = dict)
    system_processor : dict[UUID, SystemProcessor] = field(default_factory = dict)

    component : dict[UUID, Component] = field(default_factory = dict)

    system : dict[UUID, System] = field(default_factory = dict)

    def funnel(self, entity: Entity) -> dict[UUID, Any]:

        destinations: dict[type, dict[UUID, Any]] = {

            StateElement: self.state_element,
            VariableElement : self.variable_element,
            DependentExpressionElement : self.expression_element,
            Process: self.process,
            ProcessValueRef: self.process_value,
            ProcessExpression: self.process_expression,
            ProcessCondition: self.process_condition,
            ProcessTrigger: self.process_trigger,
            ProcessAction: self.process_action,

            SystemState: self.system_state,
            SystemStructure: self.system_structure,
            SystemRelation: self.system_relation,
            SystemSemantics: self.system_semantics,

            SystemProcessor: self.system_processor,
            Token : self.token,
            Port : self.port,

            Component: self.component, 
            System: self.system,
        }
    
        entity_type = type(entity)
    
        try:
            return destinations[entity_type]
        except KeyError:
            for base_type in entity_type.__mro__[1:]:
                if base_type in destinations:
                    return destinations[base_type]
            raise TypeError(
                f"No existe un diccionario en SpaceSource para "
                f"{entity_type.__name__}."
            ) from None    

# WITHOUT IMPLEMENTATION
class FlatSpaceTopology:

    graph : nx.MultiDiGraph
    systemic_space : SystemicSpace

    max_z : Optional[int]
    min_z : Optional[int]

    def __init__(self, systemic_space : SystemicSpace) -> None:
        self.graph = nx.MultiDiGraph()
        self.systemic_space : SystemicSpace = systemic_space

    def suscribe_node(self, component_or_system : UUID, z_criterion : int) -> None:
        self.graph.add_node(component_or_system, z_level = z_criterion)

    def create_mereolation(self, system: UUID, component: UUID):
        # Access both Z levels
        system_z = self.graph.nodes[system]["z_level"]
        component_z = self.graph.nodes[component]["z_level"]
        
        # Operation Lock 
        if system_z <= component_z:
            raise ValueError(
                f"Cannot create relation: system Z-level ({system_z}) "
                f"must be strictly greater than component Z-level ({component_z})."
            )
            
        # Generate Mereolation
        self.graph.add_edge(system, component, relation_type = "Mereological")

    def create_flat_relation(self, component_A : UUID, component_B : UUID, relation_type : str):
        self.graph.add_edge(component_A, component_B, relation = relation_type)

    # SYSTEM OPERATIONS

    def suscribe_system_components(self):
        pass

# WITHOUT IMPLEMENTATION 
class SpaceMainTopology:

    def __init__(self, systemic_space : SystemicSpace) -> None:
        self.graph : nx.MultiDiGraph = nx.MultiDiGraph()
        self.systemic_space : SystemicSpace = systemic_space

class ValidationEngine:

    def __init__(self, systemic_space):

        self.systemic_space : SystemicSpace = systemic_space

    # Validación de Existencia en Registros 
    def validate_existence(self, entity : UUID) -> bool:
        for entity_dict in vars(self.systemic_space.source).values():
            if entity in entity_dict:
                return True
        return False

    # Validación de Requerimentos de Tipos de Entidad 
    def validate_entity_class(self, entity : UUID, expected_entity_type : Any) -> bool:
        validation = self.systemic_space.get(entity)
        return isinstance(validation, expected_entity_type)

    # Validación de Unico
    def validate_uniqueness(self, entity : UUID, entity_Type : Ontology.EntityType) -> None:
        pass

class SystemicSpace:

    identificator : Identification

    validator : ValidationEngine

    source : Source
    space_main_topology : SpaceMainTopology
    flat_space : FlatSpaceTopology 

    def __init__(self):

        self.identificator = Identification(entity = Ontology.EntityType.SPACE)
        self.validator = ValidationEngine(systemic_space = self)

        self.source = Source()

        self.space_main_topology = SpaceMainTopology(systemic_space = self)
        self.flat_space = FlatSpaceTopology(systemic_space = self)

    # CRUD SOURCE MANAGING
    def create(self, ID : UUID , entity : Entity) -> None:
        destination = self.source.funnel(entity)
        destination[ID] = entity

    def get(self, ID : UUID) -> Any:
        for entity_dict in vars(self.source).values():
            if ID in entity_dict:
                return entity_dict[ID]
        return None

    def update(self, ID : UUID, identificator: Identification) -> None:
        for entity_dict in vars(self.source).values():
            if ID in entity_dict:
                entity_dict[ID].identificator = identificator
                return   
    
    def delete(self, ID : UUID) -> None:
        for entity_dict in vars(self.source).values():
            if ID in entity_dict:
                del entity_dict[ID]
                return
        raise KeyError(f"Entity with ID {ID} in any source")

    # COMPLEX CRUD
    def create_from_system(self, source_system : UUID) -> None:
        system = self.get(source_system)
        for entity_dict in vars(system.source).values():
            for entity in entity_dict.values():
                self.create(entity.identificator.id, entity)

    def delete_system_from_source(self, system_id : UUID) -> None:
        system = self.get(system_id)
        if not isinstance(system, System):
            raise KeyError(f"System with ID {system_id} not found")

        for entity_dict in (system.ontologies, system.structures, system.states):
            for description_id in tuple(entity_dict):
                self.delete(description_id)
        self.delete(system_id)

# ======================================
# SYSTEMS TEST ZONE
# ======================================

if __name__ == "__main__":

    systemic_space = SystemicSpace()
    engine = System(systemic_space)
    systemic_space.create(engine.identificator.id, engine)

    crankshaft = Component(systemic_space)
    camshaft = Component(systemic_space)
    timing_chain = Component(
        systemic_space,
        is_system=True,
        label="Timing chain",
    )
    intake_module = Component(
        systemic_space,
        is_system=True,
        label="Intake",
    )
    exhaust_module = Component(
        systemic_space,
        is_system=True,
        label="Exhaust",
    )
    if timing_chain.system is None or intake_module.system is None or exhaust_module.system is None:
        raise RuntimeError("The engine subsystems must have their own System descriptions.")
    timing_chain_system = timing_chain.system
    intake_system = intake_module.system
    exhaust_system = exhaust_module.system
    bore_diameter_m = 0.086
    stroke_length_m = 0.086
    crank_pin_radius_m = stroke_length_m / 2
    connecting_rod_length_m = 0.143
    engine_displacement_l = (
        np.pi / 4 * bore_diameter_m ** 2 * stroke_length_m * 4 * 1000
    )
    crank_sprocket_radius_m = 0.03
    cam_sprocket_radius_m = 0.06
    target_air_charge_kg = 0.0006
    stoichiometric_air_fuel_ratio = 14.7
    fuel_lower_heating_value_j_kg = 44_000_000
    indicated_thermal_efficiency = 0.28
    crankshaft_inertia_kg_m2 = 0.18
    friction_torque_nm = 3.0
    viscous_friction_nm_per_rad_s = 0.025
    external_load_torque_nm = 12.0
    ambient_supply_rate_kg_s = 0.12
    cylinder_air_flow_rate_kg_s = 0.04
    exhaust_flow_rate_kg_s = 0.12
    exhaust_outlet_rate_kg_s = 0.25
    initial_ambient_air_kg = 2.0
    initial_manifold_air_kg = 0.05
    initial_fuel_kg = 0.25

    cylinder_1 = Component(systemic_space)
    piston_1 = Component(systemic_space)
    pushrod_1 = Component(systemic_space)
    rocker_1 = Component(systemic_space)
    valves_1 = Component(systemic_space)
    spark_plug_1 = Component(systemic_space)
    combustion_1 = Component(systemic_space)

    cylinder_2 = Component(systemic_space)
    piston_2 = Component(systemic_space)
    pushrod_2 = Component(systemic_space)
    rocker_2 = Component(systemic_space)
    valves_2 = Component(systemic_space)
    spark_plug_2 = Component(systemic_space)
    combustion_2 = Component(systemic_space)

    cylinder_3 = Component(systemic_space)
    piston_3 = Component(systemic_space)
    pushrod_3 = Component(systemic_space)
    rocker_3 = Component(systemic_space)
    valves_3 = Component(systemic_space)
    spark_plug_3 = Component(systemic_space)
    combustion_3 = Component(systemic_space)

    cylinder_4 = Component(systemic_space)
    piston_4 = Component(systemic_space)
    pushrod_4 = Component(systemic_space)
    rocker_4 = Component(systemic_space)
    valves_4 = Component(systemic_space)
    spark_plug_4 = Component(systemic_space)
    combustion_4 = Component(systemic_space)

    crank_angle = VariableElement(systemic_space, "crank_angle", 0.0, ValueConstraints(0))
    crankshaft.system_state.add_element(crank_angle.identificator.id)
    crank_velocity = VariableElement(
        systemic_space,
        "crank_velocity_deg_s",
        0.0,
        ValueConstraints(0, 60000),
    )
    crankshaft.system_state.add_element(crank_velocity.identificator.id)
    shaft_energy = VariableElement(
        systemic_space,
        "shaft_energy_j",
        0.0,
        ValueConstraints(0),
    )
    crankshaft.system_state.add_element(shaft_energy.identificator.id)
    engine_cycles = VariableElement(
        systemic_space,
        "engine_cycles",
        0.0,
        ValueConstraints(0),
    )
    crankshaft.system_state.add_element(engine_cycles.identificator.id)
    engine_phase = DiscreteStateElement(
        systemic_space,
        "engine_phase",
        initial_value="stopped",
        allowed_values=("stopped", "intake", "compression", "power", "exhaust"),
    )
    crankshaft.system_state.add_element(engine_phase.identificator.id)
    cam_angle = VariableElement(systemic_space, "cam_angle", 0.0, ValueConstraints(0, 360))
    camshaft.system_state.add_element(cam_angle.identificator.id)
    cam_velocity = VariableElement(
        systemic_space,
        "cam_velocity_deg_s",
        0.0,
        ValueConstraints(0, 30000),
    )
    camshaft.system_state.add_element(cam_velocity.identificator.id)
    timing_chain_travel = VariableElement(
        systemic_space,
        "timing_chain_travel_m",
        0.0,
        ValueConstraints(0, 4 * np.pi * crank_sprocket_radius_m),
    )
    timing_chain.system_state.add_element(timing_chain_travel.identificator.id)
    timing_chain_speed = VariableElement(
        systemic_space,
        "timing_chain_speed_m_s",
        0.0,
        ValueConstraints(0, 100),
    )
    timing_chain.system_state.add_element(timing_chain_speed.identificator.id)
    initial_throttle_position = 0.25
    throttle_position = VariableElement(
        systemic_space,
        "throttle_position",
        initial_throttle_position,
        ValueConstraints(0, 1),
    )
    intake_module.system_state.add_element(throttle_position.identificator.id)

    cylinder_1_phase = VariableElement(systemic_space, "cylinder_1_phase", 0.0, ValueConstraints(0, 720))
    cylinder_1.system_state.add_element(cylinder_1_phase.identificator.id)
    piston_1_position = VariableElement(
        systemic_space,
        "piston_1_position",
        0.0,
        ValueConstraints(0, stroke_length_m),
    )
    piston_1.system_state.add_element(piston_1_position.identificator.id)
    pushrod_1_intake = VariableElement(systemic_space, "pushrod_1_intake", 0.0, ValueConstraints(0, 1))
    pushrod_1_exhaust = VariableElement(systemic_space, "pushrod_1_exhaust", 0.0, ValueConstraints(0, 1))
    pushrod_1.system_state.add_element(pushrod_1_intake.identificator.id)
    pushrod_1.system_state.add_element(pushrod_1_exhaust.identificator.id)
    rocker_1_intake = VariableElement(systemic_space, "rocker_1_intake", 0.0, ValueConstraints(0, 1))
    rocker_1_exhaust = VariableElement(systemic_space, "rocker_1_exhaust", 0.0, ValueConstraints(0, 1))
    rocker_1.system_state.add_element(rocker_1_intake.identificator.id)
    rocker_1.system_state.add_element(rocker_1_exhaust.identificator.id)
    valve_1_intake = VariableElement(systemic_space, "valve_1_intake", 0.0, ValueConstraints(0, 1))
    valve_1_exhaust = VariableElement(systemic_space, "valve_1_exhaust", 0.0, ValueConstraints(0, 1))
    valves_1.system_state.add_element(valve_1_intake.identificator.id)
    valves_1.system_state.add_element(valve_1_exhaust.identificator.id)
    spark_1 = VariableElement(systemic_space, "spark_1", 0.0, ValueConstraints(0, 1))
    spark_plug_1.system_state.add_element(spark_1.identificator.id)
    pressure_1 = VariableElement(systemic_space, "pressure_1", 1.0, ValueConstraints(1, 35))
    combustion_1.system_state.add_element(pressure_1.identificator.id)

    cylinder_2_phase = VariableElement(systemic_space, "cylinder_2_phase", 540.0, ValueConstraints(0, 720))
    cylinder_2.system_state.add_element(cylinder_2_phase.identificator.id)
    piston_2_position = VariableElement(
        systemic_space,
        "piston_2_position",
        0.0,
        ValueConstraints(0, stroke_length_m),
    )
    piston_2.system_state.add_element(piston_2_position.identificator.id)
    pushrod_2_intake = VariableElement(systemic_space, "pushrod_2_intake", 0.0, ValueConstraints(0, 1))
    pushrod_2_exhaust = VariableElement(systemic_space, "pushrod_2_exhaust", 0.0, ValueConstraints(0, 1))
    pushrod_2.system_state.add_element(pushrod_2_intake.identificator.id)
    pushrod_2.system_state.add_element(pushrod_2_exhaust.identificator.id)
    rocker_2_intake = VariableElement(systemic_space, "rocker_2_intake", 0.0, ValueConstraints(0, 1))
    rocker_2_exhaust = VariableElement(systemic_space, "rocker_2_exhaust", 0.0, ValueConstraints(0, 1))
    rocker_2.system_state.add_element(rocker_2_intake.identificator.id)
    rocker_2.system_state.add_element(rocker_2_exhaust.identificator.id)
    valve_2_intake = VariableElement(systemic_space, "valve_2_intake", 0.0, ValueConstraints(0, 1))
    valve_2_exhaust = VariableElement(systemic_space, "valve_2_exhaust", 0.0, ValueConstraints(0, 1))
    valves_2.system_state.add_element(valve_2_intake.identificator.id)
    valves_2.system_state.add_element(valve_2_exhaust.identificator.id)
    spark_2 = VariableElement(systemic_space, "spark_2", 0.0, ValueConstraints(0, 1))
    spark_plug_2.system_state.add_element(spark_2.identificator.id)
    pressure_2 = VariableElement(systemic_space, "pressure_2", 1.0, ValueConstraints(1, 35))
    combustion_2.system_state.add_element(pressure_2.identificator.id)

    cylinder_3_phase = VariableElement(systemic_space, "cylinder_3_phase", 180.0, ValueConstraints(0, 720))
    cylinder_3.system_state.add_element(cylinder_3_phase.identificator.id)
    piston_3_position = VariableElement(
        systemic_space,
        "piston_3_position",
        0.0,
        ValueConstraints(0, stroke_length_m),
    )
    piston_3.system_state.add_element(piston_3_position.identificator.id)
    pushrod_3_intake = VariableElement(systemic_space, "pushrod_3_intake", 0.0, ValueConstraints(0, 1))
    pushrod_3_exhaust = VariableElement(systemic_space, "pushrod_3_exhaust", 0.0, ValueConstraints(0, 1))
    pushrod_3.system_state.add_element(pushrod_3_intake.identificator.id)
    pushrod_3.system_state.add_element(pushrod_3_exhaust.identificator.id)
    rocker_3_intake = VariableElement(systemic_space, "rocker_3_intake", 0.0, ValueConstraints(0, 1))
    rocker_3_exhaust = VariableElement(systemic_space, "rocker_3_exhaust", 0.0, ValueConstraints(0, 1))
    rocker_3.system_state.add_element(rocker_3_intake.identificator.id)
    rocker_3.system_state.add_element(rocker_3_exhaust.identificator.id)
    valve_3_intake = VariableElement(systemic_space, "valve_3_intake", 0.0, ValueConstraints(0, 1))
    valve_3_exhaust = VariableElement(systemic_space, "valve_3_exhaust", 0.0, ValueConstraints(0, 1))
    valves_3.system_state.add_element(valve_3_intake.identificator.id)
    valves_3.system_state.add_element(valve_3_exhaust.identificator.id)
    spark_3 = VariableElement(systemic_space, "spark_3", 0.0, ValueConstraints(0, 1))
    spark_plug_3.system_state.add_element(spark_3.identificator.id)
    pressure_3 = VariableElement(systemic_space, "pressure_3", 1.0, ValueConstraints(1, 35))
    combustion_3.system_state.add_element(pressure_3.identificator.id)

    cylinder_4_phase = VariableElement(systemic_space, "cylinder_4_phase", 360.0, ValueConstraints(0, 720))
    cylinder_4.system_state.add_element(cylinder_4_phase.identificator.id)
    piston_4_position = VariableElement(
        systemic_space,
        "piston_4_position",
        0.0,
        ValueConstraints(0, stroke_length_m),
    )
    piston_4.system_state.add_element(piston_4_position.identificator.id)
    pushrod_4_intake = VariableElement(systemic_space, "pushrod_4_intake", 0.0, ValueConstraints(0, 1))
    pushrod_4_exhaust = VariableElement(systemic_space, "pushrod_4_exhaust", 0.0, ValueConstraints(0, 1))
    pushrod_4.system_state.add_element(pushrod_4_intake.identificator.id)
    pushrod_4.system_state.add_element(pushrod_4_exhaust.identificator.id)
    rocker_4_intake = VariableElement(systemic_space, "rocker_4_intake", 0.0, ValueConstraints(0, 1))
    rocker_4_exhaust = VariableElement(systemic_space, "rocker_4_exhaust", 0.0, ValueConstraints(0, 1))
    rocker_4.system_state.add_element(rocker_4_intake.identificator.id)
    rocker_4.system_state.add_element(rocker_4_exhaust.identificator.id)
    valve_4_intake = VariableElement(systemic_space, "valve_4_intake", 0.0, ValueConstraints(0, 1))
    valve_4_exhaust = VariableElement(systemic_space, "valve_4_exhaust", 0.0, ValueConstraints(0, 1))
    valves_4.system_state.add_element(valve_4_intake.identificator.id)
    valves_4.system_state.add_element(valve_4_exhaust.identificator.id)
    spark_4 = VariableElement(systemic_space, "spark_4", 0.0, ValueConstraints(0, 1))
    spark_plug_4.system_state.add_element(spark_4.identificator.id)
    pressure_4 = VariableElement(systemic_space, "pressure_4", 1.0, ValueConstraints(1, 35))
    combustion_4.system_state.add_element(pressure_4.identificator.id)

    ambient_air = SystemProcessor(systemic_space)
    ambient_air.identificator.label = "Ambient air source"
    ambient_air_state = SystemState(systemic_space)
    ambient_air_mass = VariableElement(
        systemic_space,
        "ambient_air_mass_kg",
        initial_ambient_air_kg,
        ValueConstraints(0),
    )
    ambient_air_state.add_element(ambient_air_mass.identificator.id)
    ambient_air.bind_state(ambient_air_state)

    intake_manifold = SystemProcessor(systemic_space)
    intake_manifold.identificator.label = "Intake manifold"
    intake_manifold_state = SystemState(systemic_space)
    manifold_air_mass = VariableElement(
        systemic_space,
        "manifold_air_mass_kg",
        initial_manifold_air_kg,
        ValueConstraints(0),
    )
    intake_manifold_state.add_element(manifold_air_mass.identificator.id)
    intake_manifold.bind_state(intake_manifold_state)

    fuel_tank = SystemProcessor(systemic_space)
    fuel_tank.identificator.label = "Fuel source"
    fuel_tank_state = SystemState(systemic_space)
    fuel_tank_mass = VariableElement(
        systemic_space,
        "fuel_tank_mass_kg",
        initial_fuel_kg,
        ValueConstraints(0),
    )
    fuel_tank_state.add_element(fuel_tank_mass.identificator.id)
    fuel_tank.bind_state(fuel_tank_state)

    cylinder_1_flow = SystemProcessor(systemic_space)
    cylinder_1_flow.identificator.label = "Cylinder 1 charge"
    cylinder_1_flow_state = SystemState(systemic_space)
    cylinder_1_air_mass = VariableElement(
        systemic_space,
        "cylinder_1_air_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    cylinder_1_fuel_mass = VariableElement(
        systemic_space,
        "cylinder_1_fuel_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    cylinder_1_flow_state.add_element(cylinder_1_air_mass.identificator.id)
    cylinder_1_flow_state.add_element(cylinder_1_fuel_mass.identificator.id)
    cylinder_1_flow.bind_state(cylinder_1_flow_state)

    cylinder_2_flow = SystemProcessor(systemic_space)
    cylinder_2_flow.identificator.label = "Cylinder 2 charge"
    cylinder_2_flow_state = SystemState(systemic_space)
    cylinder_2_air_mass = VariableElement(
        systemic_space,
        "cylinder_2_air_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    cylinder_2_fuel_mass = VariableElement(
        systemic_space,
        "cylinder_2_fuel_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    cylinder_2_flow_state.add_element(cylinder_2_air_mass.identificator.id)
    cylinder_2_flow_state.add_element(cylinder_2_fuel_mass.identificator.id)
    cylinder_2_flow.bind_state(cylinder_2_flow_state)

    cylinder_3_flow = SystemProcessor(systemic_space)
    cylinder_3_flow.identificator.label = "Cylinder 3 charge"
    cylinder_3_flow_state = SystemState(systemic_space)
    cylinder_3_air_mass = VariableElement(
        systemic_space,
        "cylinder_3_air_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    cylinder_3_fuel_mass = VariableElement(
        systemic_space,
        "cylinder_3_fuel_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    cylinder_3_flow_state.add_element(cylinder_3_air_mass.identificator.id)
    cylinder_3_flow_state.add_element(cylinder_3_fuel_mass.identificator.id)
    cylinder_3_flow.bind_state(cylinder_3_flow_state)

    cylinder_4_flow = SystemProcessor(systemic_space)
    cylinder_4_flow.identificator.label = "Cylinder 4 charge"
    cylinder_4_flow_state = SystemState(systemic_space)
    cylinder_4_air_mass = VariableElement(
        systemic_space,
        "cylinder_4_air_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    cylinder_4_fuel_mass = VariableElement(
        systemic_space,
        "cylinder_4_fuel_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    cylinder_4_flow_state.add_element(cylinder_4_air_mass.identificator.id)
    cylinder_4_flow_state.add_element(cylinder_4_fuel_mass.identificator.id)
    cylinder_4_flow.bind_state(cylinder_4_flow_state)

    exhaust_manifold = SystemProcessor(systemic_space)
    exhaust_manifold.identificator.label = "Exhaust manifold"
    exhaust_manifold_state = SystemState(systemic_space)
    exhaust_air_mass = VariableElement(
        systemic_space,
        "exhaust_air_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    exhaust_fuel_mass = VariableElement(
        systemic_space,
        "exhaust_fuel_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    exhaust_manifold_state.add_element(exhaust_air_mass.identificator.id)
    exhaust_manifold_state.add_element(exhaust_fuel_mass.identificator.id)
    exhaust_manifold.bind_state(exhaust_manifold_state)

    tailpipe = SystemProcessor(systemic_space)
    tailpipe.identificator.label = "Exhaust outlet"
    tailpipe_state = SystemState(systemic_space)
    exhaust_outlet_air_mass = VariableElement(
        systemic_space,
        "exhaust_outlet_air_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    exhaust_outlet_fuel_mass = VariableElement(
        systemic_space,
        "exhaust_outlet_fuel_mass_kg",
        0.0,
        ValueConstraints(0),
    )
    tailpipe_state.add_element(exhaust_outlet_air_mass.identificator.id)
    tailpipe_state.add_element(exhaust_outlet_fuel_mass.identificator.id)
    tailpipe.bind_state(tailpipe_state)

    timing_drive = StateCouplingStructure(systemic_space)
    timing_drive.add_component(crankshaft.identificator.id)
    timing_drive.add_component(timing_chain.identificator.id)
    timing_drive.add_component(camshaft.identificator.id)
    timing_chain_system.suscribe_systemic_description(
        timing_chain.system_state.identificator.id,
    )
    timing_chain_system.suscribe_systemic_description(
        crankshaft.system_state.identificator.id,
    )
    timing_chain_system.suscribe_systemic_description(
        camshaft.system_state.identificator.id,
    )
    timing_chain_system.suscribe_systemic_description(timing_drive.identificator.id)

    coupling = StateCouplingStructure(systemic_space)
    coupling.add_component(crankshaft.identificator.id)
    coupling.add_component(camshaft.identificator.id)
    coupling.add_component(timing_chain.identificator.id)
    coupling.add_component(intake_module.identificator.id)
    coupling.add_component(exhaust_module.identificator.id)
    coupling.add_component(cylinder_1.identificator.id)
    coupling.add_component(piston_1.identificator.id)
    coupling.add_component(pushrod_1.identificator.id)
    coupling.add_component(rocker_1.identificator.id)
    coupling.add_component(valves_1.identificator.id)
    coupling.add_component(spark_plug_1.identificator.id)
    coupling.add_component(combustion_1.identificator.id)
    coupling.add_component(cylinder_2.identificator.id)
    coupling.add_component(piston_2.identificator.id)
    coupling.add_component(pushrod_2.identificator.id)
    coupling.add_component(rocker_2.identificator.id)
    coupling.add_component(valves_2.identificator.id)
    coupling.add_component(spark_plug_2.identificator.id)
    coupling.add_component(combustion_2.identificator.id)
    coupling.add_component(cylinder_3.identificator.id)
    coupling.add_component(piston_3.identificator.id)
    coupling.add_component(pushrod_3.identificator.id)
    coupling.add_component(rocker_3.identificator.id)
    coupling.add_component(valves_3.identificator.id)
    coupling.add_component(spark_plug_3.identificator.id)
    coupling.add_component(combustion_3.identificator.id)
    coupling.add_component(cylinder_4.identificator.id)
    coupling.add_component(piston_4.identificator.id)
    coupling.add_component(pushrod_4.identificator.id)
    coupling.add_component(rocker_4.identificator.id)
    coupling.add_component(valves_4.identificator.id)
    coupling.add_component(spark_plug_4.identificator.id)
    coupling.add_component(combustion_4.identificator.id)

    crank_to_chain_travel = StateCouplingRelation(
        systemic_space,
        crankshaft.identificator.id,
        crank_angle.identificator.id,
        timing_chain.identificator.id,
        timing_chain_travel.identificator.id,
        lambda value: np.deg2rad(value % 720) * crank_sprocket_radius_m,
        label="crankshaft-to-timing-chain travel",
    )
    crank_to_chain_speed = StateCouplingRelation(
        systemic_space,
        crankshaft.identificator.id,
        crank_velocity.identificator.id,
        timing_chain.identificator.id,
        timing_chain_speed.identificator.id,
        lambda value: np.deg2rad(value) * crank_sprocket_radius_m,
        label="crankshaft-to-timing-chain speed",
    )
    chain_to_cam = StateCouplingRelation(
        systemic_space,
        timing_chain.identificator.id,
        timing_chain_travel.identificator.id,
        camshaft.identificator.id,
        cam_angle.identificator.id,
        lambda value: np.rad2deg(value / cam_sprocket_radius_m),
        label="timing-chain-to-camshaft angle",
    )
    chain_to_cam_velocity = StateCouplingRelation(
        systemic_space,
        timing_chain.identificator.id,
        timing_chain_speed.identificator.id,
        camshaft.identificator.id,
        cam_velocity.identificator.id,
        lambda value: np.rad2deg(value / cam_sprocket_radius_m),
        label="timing-chain-to-camshaft speed",
    )
    crank_to_cylinder_1 = StateCouplingRelation(
        systemic_space, crankshaft.identificator.id, crank_angle.identificator.id,
        cylinder_1.identificator.id, cylinder_1_phase.identificator.id,
        lambda value: (value + 360) % 720, label="crankshaft-to-cylinder-1",
    )
    crank_to_cylinder_2 = StateCouplingRelation(
        systemic_space, crankshaft.identificator.id, crank_angle.identificator.id,
        cylinder_2.identificator.id, cylinder_2_phase.identificator.id,
        lambda value: (value + 540) % 720, label="crankshaft-to-cylinder-2",
    )
    crank_to_cylinder_3 = StateCouplingRelation(
        systemic_space, crankshaft.identificator.id, crank_angle.identificator.id,
        cylinder_3.identificator.id, cylinder_3_phase.identificator.id,
        lambda value: (value + 180) % 720, label="crankshaft-to-cylinder-3",
    )
    crank_to_cylinder_4 = StateCouplingRelation(
        systemic_space, crankshaft.identificator.id, crank_angle.identificator.id,
        cylinder_4.identificator.id, cylinder_4_phase.identificator.id,
        lambda value: value % 720, label="crankshaft-to-cylinder-4",
    )
    cylinder_1_to_piston = StateCouplingRelation(
        systemic_space, cylinder_1.identificator.id, cylinder_1_phase.identificator.id,
        piston_1.identificator.id, piston_1_position.identificator.id,
        lambda value: (
            crank_pin_radius_m * (1 - np.cos(np.deg2rad(value)))
            + connecting_rod_length_m
            - np.sqrt(
                connecting_rod_length_m ** 2
                - (crank_pin_radius_m * np.sin(np.deg2rad(value))) ** 2
            )
        ),
    )
    cylinder_2_to_piston = StateCouplingRelation(
        systemic_space, cylinder_2.identificator.id, cylinder_2_phase.identificator.id,
        piston_2.identificator.id, piston_2_position.identificator.id,
        lambda value: (
            crank_pin_radius_m * (1 - np.cos(np.deg2rad(value)))
            + connecting_rod_length_m
            - np.sqrt(
                connecting_rod_length_m ** 2
                - (crank_pin_radius_m * np.sin(np.deg2rad(value))) ** 2
            )
        ),
    )
    cylinder_3_to_piston = StateCouplingRelation(
        systemic_space, cylinder_3.identificator.id, cylinder_3_phase.identificator.id,
        piston_3.identificator.id, piston_3_position.identificator.id,
        lambda value: (
            crank_pin_radius_m * (1 - np.cos(np.deg2rad(value)))
            + connecting_rod_length_m
            - np.sqrt(
                connecting_rod_length_m ** 2
                - (crank_pin_radius_m * np.sin(np.deg2rad(value))) ** 2
            )
        ),
    )
    cylinder_4_to_piston = StateCouplingRelation(
        systemic_space, cylinder_4.identificator.id, cylinder_4_phase.identificator.id,
        piston_4.identificator.id, piston_4_position.identificator.id,
        lambda value: (
            crank_pin_radius_m * (1 - np.cos(np.deg2rad(value)))
            + connecting_rod_length_m
            - np.sqrt(
                connecting_rod_length_m ** 2
                - (crank_pin_radius_m * np.sin(np.deg2rad(value))) ** 2
            )
        ),
    )
    cylinder_1_to_spark = StateCouplingRelation(
        systemic_space, cylinder_1.identificator.id, cylinder_1_phase.identificator.id,
        spark_plug_1.identificator.id, spark_1.identificator.id,
        lambda value: float(abs(value - 360) <= 2),
    )
    cylinder_2_to_spark = StateCouplingRelation(
        systemic_space, cylinder_2.identificator.id, cylinder_2_phase.identificator.id,
        spark_plug_2.identificator.id, spark_2.identificator.id,
        lambda value: float(abs(value - 360) <= 2),
    )
    cylinder_3_to_spark = StateCouplingRelation(
        systemic_space, cylinder_3.identificator.id, cylinder_3_phase.identificator.id,
        spark_plug_3.identificator.id, spark_3.identificator.id,
        lambda value: float(abs(value - 360) <= 2),
    )
    cylinder_4_to_spark = StateCouplingRelation(
        systemic_space, cylinder_4.identificator.id, cylinder_4_phase.identificator.id,
        spark_plug_4.identificator.id, spark_4.identificator.id,
        lambda value: float(abs(value - 360) <= 2),
    )
    cylinder_1_to_pressure = StateCouplingRelation(
        systemic_space, cylinder_1.identificator.id, cylinder_1_phase.identificator.id,
        combustion_1.identificator.id, pressure_1.identificator.id,
        lambda value: (
            1
            + 29
            * min(
                1.0,
                cylinder_1_flow_state.read_value(cylinder_1_air_mass.identificator.id)
                / target_air_charge_kg,
                cylinder_1_flow_state.read_value(cylinder_1_fuel_mass.identificator.id)
                / (target_air_charge_kg / stoichiometric_air_fuel_ratio),
            )
            * np.exp(-0.5 * ((value - 380) / 40) ** 2)
            if 360 <= value <= 540
            else 1.0
        ),
    )
    cylinder_2_to_pressure = StateCouplingRelation(
        systemic_space, cylinder_2.identificator.id, cylinder_2_phase.identificator.id,
        combustion_2.identificator.id, pressure_2.identificator.id,
        lambda value: (
            1
            + 29
            * min(
                1.0,
                cylinder_2_flow_state.read_value(cylinder_2_air_mass.identificator.id)
                / target_air_charge_kg,
                cylinder_2_flow_state.read_value(cylinder_2_fuel_mass.identificator.id)
                / (target_air_charge_kg / stoichiometric_air_fuel_ratio),
            )
            * np.exp(-0.5 * ((value - 380) / 40) ** 2)
            if 360 <= value <= 540
            else 1.0
        ),
    )
    cylinder_3_to_pressure = StateCouplingRelation(
        systemic_space, cylinder_3.identificator.id, cylinder_3_phase.identificator.id,
        combustion_3.identificator.id, pressure_3.identificator.id,
        lambda value: (
            1
            + 29
            * min(
                1.0,
                cylinder_3_flow_state.read_value(cylinder_3_air_mass.identificator.id)
                / target_air_charge_kg,
                cylinder_3_flow_state.read_value(cylinder_3_fuel_mass.identificator.id)
                / (target_air_charge_kg / stoichiometric_air_fuel_ratio),
            )
            * np.exp(-0.5 * ((value - 380) / 40) ** 2)
            if 360 <= value <= 540
            else 1.0
        ),
    )
    cylinder_4_to_pressure = StateCouplingRelation(
        systemic_space, cylinder_4.identificator.id, cylinder_4_phase.identificator.id,
        combustion_4.identificator.id, pressure_4.identificator.id,
        lambda value: (
            1
            + 29
            * min(
                1.0,
                cylinder_4_flow_state.read_value(cylinder_4_air_mass.identificator.id)
                / target_air_charge_kg,
                cylinder_4_flow_state.read_value(cylinder_4_fuel_mass.identificator.id)
                / (target_air_charge_kg / stoichiometric_air_fuel_ratio),
            )
            * np.exp(-0.5 * ((value - 380) / 40) ** 2)
            if 360 <= value <= 540
            else 1.0
        ),
    )
    cam_to_pushrod_1_intake = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_1.identificator.id, pushrod_1_intake.identificator.id,
        lambda value: (
            np.sin(np.deg2rad(((value + 180) % 360) * 2))
            if (value + 180) % 360 <= 90
            else 0
        ),
    )
    cam_to_pushrod_1_exhaust = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_1.identificator.id, pushrod_1_exhaust.identificator.id,
        lambda value: (
            np.sin(np.deg2rad((((value + 180) % 360) - 270) * 2))
            if (value + 180) % 360 >= 270
            else 0
        ),
    )
    cam_to_pushrod_2_intake = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_2.identificator.id, pushrod_2_intake.identificator.id,
        lambda value: (
            np.sin(np.deg2rad(((value + 270) % 360) * 2))
            if (value + 270) % 360 <= 90
            else 0
        ),
    )
    cam_to_pushrod_2_exhaust = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_2.identificator.id, pushrod_2_exhaust.identificator.id,
        lambda value: (
            np.sin(np.deg2rad((((value + 270) % 360) - 270) * 2))
            if (value + 270) % 360 >= 270
            else 0
        ),
    )
    cam_to_pushrod_3_intake = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_3.identificator.id, pushrod_3_intake.identificator.id,
        lambda value: (
            np.sin(np.deg2rad(((value + 90) % 360) * 2))
            if (value + 90) % 360 <= 90
            else 0
        ),
    )
    cam_to_pushrod_3_exhaust = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_3.identificator.id, pushrod_3_exhaust.identificator.id,
        lambda value: (
            np.sin(np.deg2rad((((value + 90) % 360) - 270) * 2))
            if (value + 90) % 360 >= 270
            else 0
        ),
    )
    cam_to_pushrod_4_intake = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_4.identificator.id, pushrod_4_intake.identificator.id,
        lambda value: (
            np.sin(np.deg2rad((value % 360) * 2))
            if value % 360 <= 90
            else 0
        ),
    )
    cam_to_pushrod_4_exhaust = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_4.identificator.id, pushrod_4_exhaust.identificator.id,
        lambda value: (
            np.sin(np.deg2rad(((value % 360) - 270) * 2))
            if value % 360 >= 270
            else 0
        ),
    )

    pushrod_to_rocker_1_intake = StateCouplingRelation(
        systemic_space, pushrod_1.identificator.id, pushrod_1_intake.identificator.id,
        rocker_1.identificator.id, rocker_1_intake.identificator.id,
        lambda value: min(1, value * 1.4),
    )
    pushrod_to_rocker_1_exhaust = StateCouplingRelation(
        systemic_space, pushrod_1.identificator.id, pushrod_1_exhaust.identificator.id,
        rocker_1.identificator.id, rocker_1_exhaust.identificator.id,
        lambda value: min(1, value * 1.4),
    )
    pushrod_to_rocker_2_intake = StateCouplingRelation(
        systemic_space, pushrod_2.identificator.id, pushrod_2_intake.identificator.id,
        rocker_2.identificator.id, rocker_2_intake.identificator.id,
        lambda value: min(1, value * 1.4),
    )
    pushrod_to_rocker_2_exhaust = StateCouplingRelation(
        systemic_space, pushrod_2.identificator.id, pushrod_2_exhaust.identificator.id,
        rocker_2.identificator.id, rocker_2_exhaust.identificator.id,
        lambda value: min(1, value * 1.4),
    )
    pushrod_to_rocker_3_intake = StateCouplingRelation(
        systemic_space, pushrod_3.identificator.id, pushrod_3_intake.identificator.id,
        rocker_3.identificator.id, rocker_3_intake.identificator.id,
        lambda value: min(1, value * 1.4),
    )
    pushrod_to_rocker_3_exhaust = StateCouplingRelation(
        systemic_space, pushrod_3.identificator.id, pushrod_3_exhaust.identificator.id,
        rocker_3.identificator.id, rocker_3_exhaust.identificator.id,
        lambda value: min(1, value * 1.4),
    )
    pushrod_to_rocker_4_intake = StateCouplingRelation(
        systemic_space, pushrod_4.identificator.id, pushrod_4_intake.identificator.id,
        rocker_4.identificator.id, rocker_4_intake.identificator.id,
        lambda value: min(1, value * 1.4),
    )
    pushrod_to_rocker_4_exhaust = StateCouplingRelation(
        systemic_space, pushrod_4.identificator.id, pushrod_4_exhaust.identificator.id,
        rocker_4.identificator.id, rocker_4_exhaust.identificator.id,
        lambda value: min(1, value * 1.4),
    )

    rocker_to_valve_1_intake = StateCouplingRelation(
        systemic_space, rocker_1.identificator.id, rocker_1_intake.identificator.id,
        valves_1.identificator.id, valve_1_intake.identificator.id,
        lambda value: value,
    )
    rocker_to_valve_1_exhaust = StateCouplingRelation(
        systemic_space, rocker_1.identificator.id, rocker_1_exhaust.identificator.id,
        valves_1.identificator.id, valve_1_exhaust.identificator.id,
        lambda value: value,
    )
    rocker_to_valve_2_intake = StateCouplingRelation(
        systemic_space, rocker_2.identificator.id, rocker_2_intake.identificator.id,
        valves_2.identificator.id, valve_2_intake.identificator.id,
        lambda value: value,
    )
    rocker_to_valve_2_exhaust = StateCouplingRelation(
        systemic_space, rocker_2.identificator.id, rocker_2_exhaust.identificator.id,
        valves_2.identificator.id, valve_2_exhaust.identificator.id,
        lambda value: value,
    )
    rocker_to_valve_3_intake = StateCouplingRelation(
        systemic_space, rocker_3.identificator.id, rocker_3_intake.identificator.id,
        valves_3.identificator.id, valve_3_intake.identificator.id,
        lambda value: value,
    )
    rocker_to_valve_3_exhaust = StateCouplingRelation(
        systemic_space, rocker_3.identificator.id, rocker_3_exhaust.identificator.id,
        valves_3.identificator.id, valve_3_exhaust.identificator.id,
        lambda value: value,
    )
    rocker_to_valve_4_intake = StateCouplingRelation(
        systemic_space, rocker_4.identificator.id, rocker_4_intake.identificator.id,
        valves_4.identificator.id, valve_4_intake.identificator.id,
        lambda value: value,
    )
    rocker_to_valve_4_exhaust = StateCouplingRelation(
        systemic_space, rocker_4.identificator.id, rocker_4_exhaust.identificator.id,
        valves_4.identificator.id, valve_4_exhaust.identificator.id,
        lambda value: value,
    )

    timing_drive.add_relation(crank_to_chain_travel.identificator.id)
    timing_drive.add_relation(crank_to_chain_speed.identificator.id)
    timing_drive.add_relation(chain_to_cam.identificator.id)
    timing_drive.add_relation(chain_to_cam_velocity.identificator.id)
    coupling.add_relation(crank_to_cylinder_1.identificator.id)
    coupling.add_relation(crank_to_cylinder_2.identificator.id)
    coupling.add_relation(crank_to_cylinder_3.identificator.id)
    coupling.add_relation(crank_to_cylinder_4.identificator.id)
    coupling.add_relation(cylinder_1_to_piston.identificator.id)
    coupling.add_relation(cylinder_2_to_piston.identificator.id)
    coupling.add_relation(cylinder_3_to_piston.identificator.id)
    coupling.add_relation(cylinder_4_to_piston.identificator.id)
    coupling.add_relation(cylinder_1_to_spark.identificator.id)
    coupling.add_relation(cylinder_2_to_spark.identificator.id)
    coupling.add_relation(cylinder_3_to_spark.identificator.id)
    coupling.add_relation(cylinder_4_to_spark.identificator.id)
    coupling.add_relation(cylinder_1_to_pressure.identificator.id)
    coupling.add_relation(cylinder_2_to_pressure.identificator.id)
    coupling.add_relation(cylinder_3_to_pressure.identificator.id)
    coupling.add_relation(cylinder_4_to_pressure.identificator.id)
    coupling.add_relation(cam_to_pushrod_1_intake.identificator.id)
    coupling.add_relation(cam_to_pushrod_1_exhaust.identificator.id)
    coupling.add_relation(cam_to_pushrod_2_intake.identificator.id)
    coupling.add_relation(cam_to_pushrod_2_exhaust.identificator.id)
    coupling.add_relation(cam_to_pushrod_3_intake.identificator.id)
    coupling.add_relation(cam_to_pushrod_3_exhaust.identificator.id)
    coupling.add_relation(cam_to_pushrod_4_intake.identificator.id)
    coupling.add_relation(cam_to_pushrod_4_exhaust.identificator.id)
    coupling.add_relation(pushrod_to_rocker_1_intake.identificator.id)
    coupling.add_relation(pushrod_to_rocker_1_exhaust.identificator.id)
    coupling.add_relation(pushrod_to_rocker_2_intake.identificator.id)
    coupling.add_relation(pushrod_to_rocker_2_exhaust.identificator.id)
    coupling.add_relation(pushrod_to_rocker_3_intake.identificator.id)
    coupling.add_relation(pushrod_to_rocker_3_exhaust.identificator.id)
    coupling.add_relation(pushrod_to_rocker_4_intake.identificator.id)
    coupling.add_relation(pushrod_to_rocker_4_exhaust.identificator.id)
    coupling.add_relation(rocker_to_valve_1_intake.identificator.id)
    coupling.add_relation(rocker_to_valve_1_exhaust.identificator.id)
    coupling.add_relation(rocker_to_valve_2_intake.identificator.id)
    coupling.add_relation(rocker_to_valve_2_exhaust.identificator.id)
    coupling.add_relation(rocker_to_valve_3_intake.identificator.id)
    coupling.add_relation(rocker_to_valve_3_exhaust.identificator.id)
    coupling.add_relation(rocker_to_valve_4_intake.identificator.id)
    coupling.add_relation(rocker_to_valve_4_exhaust.identificator.id)

    engine.suscribe_systemic_description(crankshaft.system_state.identificator.id)
    engine.suscribe_systemic_description(camshaft.system_state.identificator.id)
    engine.suscribe_systemic_description(timing_chain.system_state.identificator.id)
    engine.suscribe_systemic_description(intake_module.system_state.identificator.id)
    engine.suscribe_systemic_description(ambient_air_state.identificator.id)
    engine.suscribe_systemic_description(intake_manifold_state.identificator.id)
    engine.suscribe_systemic_description(fuel_tank_state.identificator.id)
    engine.suscribe_systemic_description(cylinder_1_flow_state.identificator.id)
    engine.suscribe_systemic_description(cylinder_2_flow_state.identificator.id)
    engine.suscribe_systemic_description(cylinder_3_flow_state.identificator.id)
    engine.suscribe_systemic_description(cylinder_4_flow_state.identificator.id)
    engine.suscribe_systemic_description(exhaust_manifold_state.identificator.id)
    engine.suscribe_systemic_description(tailpipe_state.identificator.id)
    engine.suscribe_systemic_description(cylinder_1.system_state.identificator.id)
    engine.suscribe_systemic_description(piston_1.system_state.identificator.id)
    engine.suscribe_systemic_description(pushrod_1.system_state.identificator.id)
    engine.suscribe_systemic_description(rocker_1.system_state.identificator.id)
    engine.suscribe_systemic_description(valves_1.system_state.identificator.id)
    engine.suscribe_systemic_description(spark_plug_1.system_state.identificator.id)
    engine.suscribe_systemic_description(combustion_1.system_state.identificator.id)
    engine.suscribe_systemic_description(cylinder_2.system_state.identificator.id)
    engine.suscribe_systemic_description(piston_2.system_state.identificator.id)
    engine.suscribe_systemic_description(pushrod_2.system_state.identificator.id)
    engine.suscribe_systemic_description(rocker_2.system_state.identificator.id)
    engine.suscribe_systemic_description(valves_2.system_state.identificator.id)
    engine.suscribe_systemic_description(spark_plug_2.system_state.identificator.id)
    engine.suscribe_systemic_description(combustion_2.system_state.identificator.id)
    engine.suscribe_systemic_description(cylinder_3.system_state.identificator.id)
    engine.suscribe_systemic_description(piston_3.system_state.identificator.id)
    engine.suscribe_systemic_description(pushrod_3.system_state.identificator.id)
    engine.suscribe_systemic_description(rocker_3.system_state.identificator.id)
    engine.suscribe_systemic_description(valves_3.system_state.identificator.id)
    engine.suscribe_systemic_description(spark_plug_3.system_state.identificator.id)
    engine.suscribe_systemic_description(combustion_3.system_state.identificator.id)
    engine.suscribe_systemic_description(cylinder_4.system_state.identificator.id)
    engine.suscribe_systemic_description(piston_4.system_state.identificator.id)
    engine.suscribe_systemic_description(pushrod_4.system_state.identificator.id)
    engine.suscribe_systemic_description(rocker_4.system_state.identificator.id)
    engine.suscribe_systemic_description(valves_4.system_state.identificator.id)
    engine.suscribe_systemic_description(spark_plug_4.system_state.identificator.id)
    engine.suscribe_systemic_description(combustion_4.system_state.identificator.id)
    engine.suscribe_systemic_description(coupling.identificator.id)

    # Identification etiqueta cada Component existente; SystemSemantics agrega
    # arquetipos tipados y etiquetas consultables sin sustituir sus estados.
    crankshaft.identificator.label = "Crankshaft"
    camshaft.identificator.label = "Camshaft"
    cylinder_1.identificator.label = "Cylinder 1"
    piston_1.identificator.label = "Piston 1"
    pushrod_1.identificator.label = "Pushrod 1"
    rocker_1.identificator.label = "Rocker 1"
    valves_1.identificator.label = "Valves 1"
    spark_plug_1.identificator.label = "Spark plug 1"
    combustion_1.identificator.label = "Combustion 1"
    timing_chain.system.identificator.label = "Timing-chain subsystem"
    intake_module.system.identificator.label = "Intake subsystem"
    exhaust_module.system.identificator.label = "Exhaust subsystem"
    timing_chain.semantic_attributes = {
        "drive_type": "roller chain",
        "ratio": 2.0,
        "cam_to_crank_ratio": 0.5,
    }
    intake_module.semantic_attributes = {
        "role": "air and fuel intake",
        "valve_count": 4,
        "air_fuel_ratio": stoichiometric_air_fuel_ratio,
    }
    exhaust_module.semantic_attributes = {
        "role": "combustion gas outlet",
        "outlet_count": 1,
    }
    cylinder_2.identificator.label = "Cylinder 2"
    piston_2.identificator.label = "Piston 2"
    pushrod_2.identificator.label = "Pushrod 2"
    rocker_2.identificator.label = "Rocker 2"
    valves_2.identificator.label = "Valves 2"
    spark_plug_2.identificator.label = "Spark plug 2"
    combustion_2.identificator.label = "Combustion 2"
    cylinder_3.identificator.label = "Cylinder 3"
    piston_3.identificator.label = "Piston 3"
    pushrod_3.identificator.label = "Pushrod 3"
    rocker_3.identificator.label = "Rocker 3"
    valves_3.identificator.label = "Valves 3"
    spark_plug_3.identificator.label = "Spark plug 3"
    combustion_3.identificator.label = "Combustion 3"
    cylinder_4.identificator.label = "Cylinder 4"
    piston_4.identificator.label = "Piston 4"
    pushrod_4.identificator.label = "Pushrod 4"
    rocker_4.identificator.label = "Rocker 4"
    valves_4.identificator.label = "Valves 4"
    spark_plug_4.identificator.label = "Spark plug 4"
    combustion_4.identificator.label = "Combustion 4"
    engine.semantic_attributes = {
        "configuration": "inline-four",
        "valve_train": "OHV",
        "firing_order": "1-3-4-2",
        "cylinder_count": 4,
        "piston_count": 4,
        "valves_per_cylinder": 2,
        "total_valves": 8,
        "bore_mm": bore_diameter_m * 1000,
        "stroke_mm": stroke_length_m * 1000,
        "connecting_rod_mm": connecting_rod_length_m * 1000,
        "displacement_l": engine_displacement_l,
        "cycle_angle_deg": 720,
    }
    timing_chain.system.semantic_attributes = {
        "drive_type": "roller chain",
        "ratio": 2.0,
        "cam_to_crank_ratio": 0.5,
    }
    intake_module.system.semantic_attributes = {
        "role": "metered air and fuel input",
        "air_fuel_ratio": stoichiometric_air_fuel_ratio,
    }
    exhaust_module.system.semantic_attributes = {
        "role": "combustion gas output",
        "flow_mode": "continuous",
    }

    # define_archetype_from_instance registra el System actual como instancia
    # y deriva su esquema desde semantic_attributes; add_attribute fija tipos.
    engine.semantics.add_tag(engine.identificator.id, "engine")
    engine.semantics.add_tag(engine.identificator.id, "four-stroke")
    engine.semantics.add_tag(engine.identificator.id, "OHV")
    engine.semantics.add_tag(timing_chain.identificator.id, "timing-drive")
    engine.semantics.add_tag(intake_module.identificator.id, "intake")
    engine.semantics.add_tag(exhaust_module.identificator.id, "exhaust")
    timing_chain_system.semantics.add_tag(
        timing_chain_system.identificator.id,
        "timing-chain",
    )
    intake_system.semantics.add_tag(intake_system.identificator.id, "intake")
    exhaust_system.semantics.add_tag(exhaust_system.identificator.id, "exhaust")
    engine_archetype = engine.semantics.define_archetype_from_instance(
        "L4 OHV engine",
        engine.identificator.id,
    )
    engine.semantics.define_archetype_from_instance(
        "Timing chain component",
        timing_chain.identificator.id,
    )
    engine.semantics.define_archetype_from_instance(
        "Intake component",
        intake_module.identificator.id,
    )
    engine.semantics.define_archetype_from_instance(
        "Exhaust component",
        exhaust_module.identificator.id,
    )
    timing_chain_system.semantics.define_archetype_from_instance(
        "Timing-chain subsystem",
        timing_chain_system.identificator.id,
    )
    intake_system.semantics.define_archetype_from_instance(
        "Intake flow subsystem",
        intake_system.identificator.id,
    )
    exhaust_system.semantics.define_archetype_from_instance(
        "Exhaust flow subsystem",
        exhaust_system.identificator.id,
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute(
            "configuration", SemanticValueType.STRING, "inline-four", required=True,
        ),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute("valve_train", SemanticValueType.STRING, "OHV", required=True),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute("firing_order", SemanticValueType.STRING, "1-3-4-2", required=True),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute("cylinder_count", SemanticValueType.INTEGER, 4, required=True),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute("piston_count", SemanticValueType.INTEGER, 4, required=True),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute("valves_per_cylinder", SemanticValueType.INTEGER, 2, required=True),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute("total_valves", SemanticValueType.INTEGER, 8, required=True),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute(
            "bore_mm",
            SemanticValueType.NUMBER,
            bore_diameter_m * 1000,
            required=True,
        ),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute(
            "stroke_mm",
            SemanticValueType.NUMBER,
            stroke_length_m * 1000,
            required=True,
        ),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute(
            "connecting_rod_mm",
            SemanticValueType.NUMBER,
            connecting_rod_length_m * 1000,
            required=True,
        ),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute(
            "displacement_l",
            SemanticValueType.NUMBER,
            engine_displacement_l,
            required=True,
        ),
    )
    engine.semantics.add_attribute(
        engine_archetype.id,
        SemanticAttribute("cycle_angle_deg", SemanticValueType.INTEGER, 720, required=True),
    )

    # add_tag clasifica las entidades por función; los SemanticSet siguientes
    # agrupan los 33 componentes del motor en una partición explícita.
    engine.semantics.add_tag(timing_chain.identificator.id, "valvetrain")
    engine.semantics.add_tag(intake_module.identificator.id, "gas-exchange")
    engine.semantics.add_tag(exhaust_module.identificator.id, "gas-exchange")
    engine.semantics.add_tag(crankshaft.identificator.id, "rotating")
    engine.semantics.add_tag(camshaft.identificator.id, "rotating")
    engine.semantics.add_tag(cylinder_1.identificator.id, "cylinder")
    engine.semantics.add_tag(cylinder_2.identificator.id, "cylinder")
    engine.semantics.add_tag(cylinder_3.identificator.id, "cylinder")
    engine.semantics.add_tag(cylinder_4.identificator.id, "cylinder")
    engine.semantics.add_tag(piston_1.identificator.id, "reciprocating")
    engine.semantics.add_tag(piston_2.identificator.id, "reciprocating")
    engine.semantics.add_tag(piston_3.identificator.id, "reciprocating")
    engine.semantics.add_tag(piston_4.identificator.id, "reciprocating")
    engine.semantics.add_tag(pushrod_1.identificator.id, "valvetrain")
    engine.semantics.add_tag(pushrod_2.identificator.id, "valvetrain")
    engine.semantics.add_tag(pushrod_3.identificator.id, "valvetrain")
    engine.semantics.add_tag(pushrod_4.identificator.id, "valvetrain")
    engine.semantics.add_tag(rocker_1.identificator.id, "valvetrain")
    engine.semantics.add_tag(rocker_2.identificator.id, "valvetrain")
    engine.semantics.add_tag(rocker_3.identificator.id, "valvetrain")
    engine.semantics.add_tag(rocker_4.identificator.id, "valvetrain")
    engine.semantics.add_tag(valves_1.identificator.id, "valvetrain")
    engine.semantics.add_tag(valves_2.identificator.id, "valvetrain")
    engine.semantics.add_tag(valves_3.identificator.id, "valvetrain")
    engine.semantics.add_tag(valves_4.identificator.id, "valvetrain")
    engine.semantics.add_tag(spark_plug_1.identificator.id, "ignition")
    engine.semantics.add_tag(spark_plug_2.identificator.id, "ignition")
    engine.semantics.add_tag(spark_plug_3.identificator.id, "ignition")
    engine.semantics.add_tag(spark_plug_4.identificator.id, "ignition")
    engine.semantics.add_tag(combustion_1.identificator.id, "combustion")
    engine.semantics.add_tag(combustion_2.identificator.id, "combustion")
    engine.semantics.add_tag(combustion_3.identificator.id, "combustion")
    engine.semantics.add_tag(combustion_4.identificator.id, "combustion")

    engine_parts = engine.semantics.define_set("L4 OHV components")
    rotating_parts = engine.semantics.define_set("Rotating assembly")
    cylinder_parts = engine.semantics.define_set("Cylinders")
    reciprocating_parts = engine.semantics.define_set("Pistons")
    valvetrain_parts = engine.semantics.define_set("Valve train")
    ignition_parts = engine.semantics.define_set("Ignition")
    combustion_parts = engine.semantics.define_set("Combustion chambers")
    gas_exchange_parts = engine.semantics.define_set("Gas exchange")
    engine.semantics.add_member(engine_parts.id, crankshaft.identificator.id)
    engine.semantics.add_member(engine_parts.id, camshaft.identificator.id)
    engine.semantics.add_member(engine_parts.id, cylinder_1.identificator.id)
    engine.semantics.add_member(engine_parts.id, piston_1.identificator.id)
    engine.semantics.add_member(engine_parts.id, pushrod_1.identificator.id)
    engine.semantics.add_member(engine_parts.id, rocker_1.identificator.id)
    engine.semantics.add_member(engine_parts.id, valves_1.identificator.id)
    engine.semantics.add_member(engine_parts.id, spark_plug_1.identificator.id)
    engine.semantics.add_member(engine_parts.id, combustion_1.identificator.id)
    engine.semantics.add_member(engine_parts.id, cylinder_2.identificator.id)
    engine.semantics.add_member(engine_parts.id, piston_2.identificator.id)
    engine.semantics.add_member(engine_parts.id, pushrod_2.identificator.id)
    engine.semantics.add_member(engine_parts.id, rocker_2.identificator.id)
    engine.semantics.add_member(engine_parts.id, valves_2.identificator.id)
    engine.semantics.add_member(engine_parts.id, spark_plug_2.identificator.id)
    engine.semantics.add_member(engine_parts.id, combustion_2.identificator.id)
    engine.semantics.add_member(engine_parts.id, cylinder_3.identificator.id)
    engine.semantics.add_member(engine_parts.id, piston_3.identificator.id)
    engine.semantics.add_member(engine_parts.id, pushrod_3.identificator.id)
    engine.semantics.add_member(engine_parts.id, rocker_3.identificator.id)
    engine.semantics.add_member(engine_parts.id, valves_3.identificator.id)
    engine.semantics.add_member(engine_parts.id, spark_plug_3.identificator.id)
    engine.semantics.add_member(engine_parts.id, combustion_3.identificator.id)
    engine.semantics.add_member(engine_parts.id, cylinder_4.identificator.id)
    engine.semantics.add_member(engine_parts.id, piston_4.identificator.id)
    engine.semantics.add_member(engine_parts.id, pushrod_4.identificator.id)
    engine.semantics.add_member(engine_parts.id, rocker_4.identificator.id)
    engine.semantics.add_member(engine_parts.id, valves_4.identificator.id)
    engine.semantics.add_member(engine_parts.id, spark_plug_4.identificator.id)
    engine.semantics.add_member(engine_parts.id, combustion_4.identificator.id)
    engine.semantics.add_member(engine_parts.id, timing_chain.identificator.id)
    engine.semantics.add_member(engine_parts.id, intake_module.identificator.id)
    engine.semantics.add_member(engine_parts.id, exhaust_module.identificator.id)
    engine.semantics.add_member(rotating_parts.id, crankshaft.identificator.id)
    engine.semantics.add_member(rotating_parts.id, camshaft.identificator.id)
    engine.semantics.add_member(cylinder_parts.id, cylinder_1.identificator.id)
    engine.semantics.add_member(cylinder_parts.id, cylinder_2.identificator.id)
    engine.semantics.add_member(cylinder_parts.id, cylinder_3.identificator.id)
    engine.semantics.add_member(cylinder_parts.id, cylinder_4.identificator.id)
    engine.semantics.add_member(reciprocating_parts.id, piston_1.identificator.id)
    engine.semantics.add_member(reciprocating_parts.id, piston_2.identificator.id)
    engine.semantics.add_member(reciprocating_parts.id, piston_3.identificator.id)
    engine.semantics.add_member(reciprocating_parts.id, piston_4.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, pushrod_1.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, pushrod_2.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, pushrod_3.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, pushrod_4.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, rocker_1.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, rocker_2.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, rocker_3.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, rocker_4.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, valves_1.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, valves_2.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, valves_3.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, valves_4.identificator.id)
    engine.semantics.add_member(valvetrain_parts.id, timing_chain.identificator.id)
    engine.semantics.add_member(ignition_parts.id, spark_plug_1.identificator.id)
    engine.semantics.add_member(ignition_parts.id, spark_plug_2.identificator.id)
    engine.semantics.add_member(ignition_parts.id, spark_plug_3.identificator.id)
    engine.semantics.add_member(ignition_parts.id, spark_plug_4.identificator.id)
    engine.semantics.add_member(combustion_parts.id, combustion_1.identificator.id)
    engine.semantics.add_member(combustion_parts.id, combustion_2.identificator.id)
    engine.semantics.add_member(combustion_parts.id, combustion_3.identificator.id)
    engine.semantics.add_member(combustion_parts.id, combustion_4.identificator.id)
    engine.semantics.add_member(gas_exchange_parts.id, intake_module.identificator.id)
    engine.semantics.add_member(gas_exchange_parts.id, exhaust_module.identificator.id)
    engine.semantics.define_partition(
        engine_parts.id,
        (
            rotating_parts.id,
            cylinder_parts.id,
            reciprocating_parts.id,
            valvetrain_parts.id,
            ignition_parts.id,
            combustion_parts.id,
            gas_exchange_parts.id,
        ),
    )

    # add_part declara la mereología System→Component y Cylinder→subcomponent;
    # is_part_of conserva la transitividad sin duplicar conexiones indirectas.
    engine.semantics.add_part(engine.identificator.id, crankshaft.identificator.id)
    engine.semantics.add_part(engine.identificator.id, camshaft.identificator.id)
    engine.semantics.add_part(engine.identificator.id, timing_chain.identificator.id)
    engine.semantics.add_part(engine.identificator.id, intake_module.identificator.id)
    engine.semantics.add_part(engine.identificator.id, exhaust_module.identificator.id)
    engine.semantics.add_part(
        timing_chain.identificator.id,
        timing_chain_system.identificator.id,
    )
    engine.semantics.add_part(
        intake_module.identificator.id,
        intake_system.identificator.id,
    )
    engine.semantics.add_part(
        exhaust_module.identificator.id,
        exhaust_system.identificator.id,
    )
    engine.semantics.add_part(
        intake_module.identificator.id,
        ambient_air.identificator.id,
    )
    engine.semantics.add_part(
        intake_module.identificator.id,
        intake_manifold.identificator.id,
    )
    engine.semantics.add_part(intake_module.identificator.id, fuel_tank.identificator.id)
    engine.semantics.add_part(
        intake_module.identificator.id,
        cylinder_1_flow.identificator.id,
    )
    engine.semantics.add_part(
        intake_module.identificator.id,
        cylinder_2_flow.identificator.id,
    )
    engine.semantics.add_part(
        intake_module.identificator.id,
        cylinder_3_flow.identificator.id,
    )
    engine.semantics.add_part(
        intake_module.identificator.id,
        cylinder_4_flow.identificator.id,
    )
    engine.semantics.add_part(
        exhaust_module.identificator.id,
        exhaust_manifold.identificator.id,
    )
    engine.semantics.add_part(exhaust_module.identificator.id, tailpipe.identificator.id)
    engine.semantics.add_part(engine.identificator.id, cylinder_1.identificator.id)
    engine.semantics.add_part(engine.identificator.id, cylinder_2.identificator.id)
    engine.semantics.add_part(engine.identificator.id, cylinder_3.identificator.id)
    engine.semantics.add_part(engine.identificator.id, cylinder_4.identificator.id)
    engine.semantics.add_part(cylinder_1.identificator.id, piston_1.identificator.id)
    engine.semantics.add_part(cylinder_1.identificator.id, pushrod_1.identificator.id)
    engine.semantics.add_part(cylinder_1.identificator.id, rocker_1.identificator.id)
    engine.semantics.add_part(cylinder_1.identificator.id, valves_1.identificator.id)
    engine.semantics.add_part(cylinder_1.identificator.id, spark_plug_1.identificator.id)
    engine.semantics.add_part(cylinder_1.identificator.id, combustion_1.identificator.id)
    engine.semantics.add_part(cylinder_2.identificator.id, piston_2.identificator.id)
    engine.semantics.add_part(cylinder_2.identificator.id, pushrod_2.identificator.id)
    engine.semantics.add_part(cylinder_2.identificator.id, rocker_2.identificator.id)
    engine.semantics.add_part(cylinder_2.identificator.id, valves_2.identificator.id)
    engine.semantics.add_part(cylinder_2.identificator.id, spark_plug_2.identificator.id)
    engine.semantics.add_part(cylinder_2.identificator.id, combustion_2.identificator.id)
    engine.semantics.add_part(cylinder_3.identificator.id, piston_3.identificator.id)
    engine.semantics.add_part(cylinder_3.identificator.id, pushrod_3.identificator.id)
    engine.semantics.add_part(cylinder_3.identificator.id, rocker_3.identificator.id)
    engine.semantics.add_part(cylinder_3.identificator.id, valves_3.identificator.id)
    engine.semantics.add_part(cylinder_3.identificator.id, spark_plug_3.identificator.id)
    engine.semantics.add_part(cylinder_3.identificator.id, combustion_3.identificator.id)
    engine.semantics.add_part(cylinder_4.identificator.id, piston_4.identificator.id)
    engine.semantics.add_part(cylinder_4.identificator.id, pushrod_4.identificator.id)
    engine.semantics.add_part(cylinder_4.identificator.id, rocker_4.identificator.id)
    engine.semantics.add_part(cylinder_4.identificator.id, valves_4.identificator.id)
    engine.semantics.add_part(cylinder_4.identificator.id, spark_plug_4.identificator.id)
    engine.semantics.add_part(cylinder_4.identificator.id, combustion_4.identificator.id)

    # SemanticUnit da dimensión y unidad a VariableElement sin alterar su valor;
    # bind_state_variable vincula cada UUID de variable con su unidad declarada.
    engine.semantics.add_base_dimension("length")
    engine.semantics.add_base_dimension("angle")
    engine.semantics.add_base_dimension("pressure")
    engine.semantics.add_base_dimension("time")
    engine.semantics.add_base_dimension("mass")
    meter_unit = engine.semantics.define_unit("meter", "m", {"length": 1})
    meter_per_second_unit = engine.semantics.define_unit(
        "meter per second",
        "m/s",
        {"length": 1, "time": -1},
    )
    degree_unit = engine.semantics.define_unit("degree", "deg", {"angle": 1})
    degree_per_second_unit = engine.semantics.define_unit(
        "degree per second",
        "deg/s",
        {"angle": 1, "time": -1},
    )
    bar_unit = engine.semantics.define_unit("bar", "bar", {"pressure": 1})
    fraction_unit = engine.semantics.define_unit("fraction", "1", {})
    kilogram_unit = engine.semantics.define_unit("kilogram", "kg", {"mass": 1})
    joule_unit = engine.semantics.define_unit(
        "joule",
        "J",
        {"mass": 1, "length": 2, "time": -2},
    )
    engine.semantics.bind_state_variable(
        crankshaft.system_state,
        shaft_energy.identificator.id,
        joule_unit.id,
    )
    engine.semantics.bind_state_variable(
        timing_chain.system_state,
        timing_chain_travel.identificator.id,
        meter_unit.id,
    )
    engine.semantics.bind_state_variable(
        timing_chain.system_state,
        timing_chain_speed.identificator.id,
        meter_per_second_unit.id,
    )
    engine.semantics.bind_state_variable(
        intake_module.system_state,
        throttle_position.identificator.id,
        fraction_unit.id,
    )
    engine.semantics.bind_state_variable(
        ambient_air_state,
        ambient_air_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        intake_manifold_state,
        manifold_air_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        fuel_tank_state,
        fuel_tank_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        cylinder_1_flow_state,
        cylinder_1_air_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        cylinder_1_flow_state,
        cylinder_1_fuel_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        cylinder_2_flow_state,
        cylinder_2_air_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        cylinder_2_flow_state,
        cylinder_2_fuel_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        cylinder_3_flow_state,
        cylinder_3_air_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        cylinder_3_flow_state,
        cylinder_3_fuel_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        cylinder_4_flow_state,
        cylinder_4_air_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        cylinder_4_flow_state,
        cylinder_4_fuel_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        exhaust_manifold_state,
        exhaust_air_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        exhaust_manifold_state,
        exhaust_fuel_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        tailpipe_state,
        exhaust_outlet_air_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(
        tailpipe_state,
        exhaust_outlet_fuel_mass.identificator.id,
        kilogram_unit.id,
    )
    engine.semantics.bind_state_variable(crankshaft.system_state, crank_angle.identificator.id, degree_unit.id)
    engine.semantics.bind_state_variable(
        crankshaft.system_state,
        crank_velocity.identificator.id,
        degree_per_second_unit.id,
    )
    engine.semantics.bind_state_variable(
        crankshaft.system_state,
        engine_cycles.identificator.id,
        fraction_unit.id,
    )
    engine.semantics.bind_state_variable(camshaft.system_state, cam_angle.identificator.id, degree_unit.id)
    engine.semantics.bind_state_variable(
        camshaft.system_state,
        cam_velocity.identificator.id,
        degree_per_second_unit.id,
    )
    engine.semantics.bind_state_variable(cylinder_1.system_state, cylinder_1_phase.identificator.id, degree_unit.id)
    engine.semantics.bind_state_variable(piston_1.system_state, piston_1_position.identificator.id, meter_unit.id)
    engine.semantics.bind_state_variable(pushrod_1.system_state, pushrod_1_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(pushrod_1.system_state, pushrod_1_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(rocker_1.system_state, rocker_1_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(rocker_1.system_state, rocker_1_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(valves_1.system_state, valve_1_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(valves_1.system_state, valve_1_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(spark_plug_1.system_state, spark_1.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(combustion_1.system_state, pressure_1.identificator.id, bar_unit.id)
    engine.semantics.bind_state_variable(cylinder_2.system_state, cylinder_2_phase.identificator.id, degree_unit.id)
    engine.semantics.bind_state_variable(piston_2.system_state, piston_2_position.identificator.id, meter_unit.id)
    engine.semantics.bind_state_variable(pushrod_2.system_state, pushrod_2_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(pushrod_2.system_state, pushrod_2_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(rocker_2.system_state, rocker_2_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(rocker_2.system_state, rocker_2_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(valves_2.system_state, valve_2_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(valves_2.system_state, valve_2_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(spark_plug_2.system_state, spark_2.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(combustion_2.system_state, pressure_2.identificator.id, bar_unit.id)
    engine.semantics.bind_state_variable(cylinder_3.system_state, cylinder_3_phase.identificator.id, degree_unit.id)
    engine.semantics.bind_state_variable(piston_3.system_state, piston_3_position.identificator.id, meter_unit.id)
    engine.semantics.bind_state_variable(pushrod_3.system_state, pushrod_3_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(pushrod_3.system_state, pushrod_3_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(rocker_3.system_state, rocker_3_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(rocker_3.system_state, rocker_3_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(valves_3.system_state, valve_3_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(valves_3.system_state, valve_3_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(spark_plug_3.system_state, spark_3.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(combustion_3.system_state, pressure_3.identificator.id, bar_unit.id)
    engine.semantics.bind_state_variable(cylinder_4.system_state, cylinder_4_phase.identificator.id, degree_unit.id)
    engine.semantics.bind_state_variable(piston_4.system_state, piston_4_position.identificator.id, meter_unit.id)
    engine.semantics.bind_state_variable(pushrod_4.system_state, pushrod_4_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(pushrod_4.system_state, pushrod_4_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(rocker_4.system_state, rocker_4_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(rocker_4.system_state, rocker_4_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(valves_4.system_state, valve_4_intake.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(valves_4.system_state, valve_4_exhaust.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(spark_plug_4.system_state, spark_4.identificator.id, fraction_unit.id)
    engine.semantics.bind_state_variable(combustion_4.system_state, pressure_4.identificator.id, bar_unit.id)

    engine_sampling_interval = 0.0025

    ambient_to_manifold_port = Port(
        systemic_space,
        intake_manifold.identificator.id,
    )
    ambient_to_manifold_port.identificator.label = "Intake air inlet"
    ambient_air.add_port(ambient_to_manifold_port.identificator.id)
    manifold_to_cylinder_1_port = Port(
        systemic_space,
        cylinder_1_flow.identificator.id,
    )
    manifold_to_cylinder_1_port.identificator.label = "Cylinder 1 intake"
    intake_manifold.add_port(manifold_to_cylinder_1_port.identificator.id)
    manifold_to_cylinder_2_port = Port(
        systemic_space,
        cylinder_2_flow.identificator.id,
    )
    manifold_to_cylinder_2_port.identificator.label = "Cylinder 2 intake"
    intake_manifold.add_port(manifold_to_cylinder_2_port.identificator.id)
    manifold_to_cylinder_3_port = Port(
        systemic_space,
        cylinder_3_flow.identificator.id,
    )
    manifold_to_cylinder_3_port.identificator.label = "Cylinder 3 intake"
    intake_manifold.add_port(manifold_to_cylinder_3_port.identificator.id)
    manifold_to_cylinder_4_port = Port(
        systemic_space,
        cylinder_4_flow.identificator.id,
    )
    manifold_to_cylinder_4_port.identificator.label = "Cylinder 4 intake"
    intake_manifold.add_port(manifold_to_cylinder_4_port.identificator.id)
    fuel_to_cylinder_1_port = Port(systemic_space, cylinder_1_flow.identificator.id)
    fuel_to_cylinder_1_port.identificator.label = "Cylinder 1 injector"
    fuel_tank.add_port(fuel_to_cylinder_1_port.identificator.id)
    fuel_to_cylinder_2_port = Port(systemic_space, cylinder_2_flow.identificator.id)
    fuel_to_cylinder_2_port.identificator.label = "Cylinder 2 injector"
    fuel_tank.add_port(fuel_to_cylinder_2_port.identificator.id)
    fuel_to_cylinder_3_port = Port(systemic_space, cylinder_3_flow.identificator.id)
    fuel_to_cylinder_3_port.identificator.label = "Cylinder 3 injector"
    fuel_tank.add_port(fuel_to_cylinder_3_port.identificator.id)
    fuel_to_cylinder_4_port = Port(systemic_space, cylinder_4_flow.identificator.id)
    fuel_to_cylinder_4_port.identificator.label = "Cylinder 4 injector"
    fuel_tank.add_port(fuel_to_cylinder_4_port.identificator.id)

    cylinder_1_to_exhaust_port = Port(
        systemic_space,
        exhaust_manifold.identificator.id,
    )
    cylinder_1_to_exhaust_port.identificator.label = "Cylinder 1 exhaust"
    cylinder_1_flow.add_port(cylinder_1_to_exhaust_port.identificator.id)
    cylinder_2_to_exhaust_port = Port(
        systemic_space,
        exhaust_manifold.identificator.id,
    )
    cylinder_2_to_exhaust_port.identificator.label = "Cylinder 2 exhaust"
    cylinder_2_flow.add_port(cylinder_2_to_exhaust_port.identificator.id)
    cylinder_3_to_exhaust_port = Port(
        systemic_space,
        exhaust_manifold.identificator.id,
    )
    cylinder_3_to_exhaust_port.identificator.label = "Cylinder 3 exhaust"
    cylinder_3_flow.add_port(cylinder_3_to_exhaust_port.identificator.id)
    cylinder_4_to_exhaust_port = Port(
        systemic_space,
        exhaust_manifold.identificator.id,
    )
    cylinder_4_to_exhaust_port.identificator.label = "Cylinder 4 exhaust"
    cylinder_4_flow.add_port(cylinder_4_to_exhaust_port.identificator.id)
    exhaust_to_tailpipe_port = Port(systemic_space, tailpipe.identificator.id)
    exhaust_to_tailpipe_port.identificator.label = "Exhaust outlet"
    exhaust_manifold.add_port(exhaust_to_tailpipe_port.identificator.id)

    intake_flow_bus = FlowBusStructure(systemic_space)
    intake_flow_bus.add_node(ambient_air.identificator.id)
    intake_flow_bus.add_node(intake_manifold.identificator.id)
    intake_flow_bus.add_node(fuel_tank.identificator.id)
    intake_flow_bus.add_node(cylinder_1_flow.identificator.id)
    intake_flow_bus.add_node(cylinder_2_flow.identificator.id)
    intake_flow_bus.add_node(cylinder_3_flow.identificator.id)
    intake_flow_bus.add_node(cylinder_4_flow.identificator.id)
    ambient_to_manifold = ContinuousFlowRelation(
        systemic_space,
        ambient_air.identificator.id,
        ambient_air_mass.identificator.id,
        intake_manifold.identificator.id,
        manifold_air_mass.identificator.id,
        rate=lambda _time, _source, _target: (
            ambient_supply_rate_kg_s
            * intake_module.system_state.read_value(
                throttle_position.identificator.id,
            )
        ),
        label="ambient-air-to-manifold",
        port_id=ambient_to_manifold_port.identificator.id,
    )
    manifold_to_cylinder_1 = ContinuousFlowRelation(
        systemic_space,
        intake_manifold.identificator.id,
        manifold_air_mass.identificator.id,
        cylinder_1_flow.identificator.id,
        cylinder_1_air_mass.identificator.id,
        rate=lambda _time, source, _target: (
            min(
                cylinder_air_flow_rate_kg_s
                * intake_module.system_state.read_value(
                    throttle_position.identificator.id,
                ),
                0.999 * source / engine_sampling_interval,
            )
            * valves_1.system_state.read_value(valve_1_intake.identificator.id)
            / max(
                1.0,
                valves_1.system_state.read_value(valve_1_intake.identificator.id)
                + valves_2.system_state.read_value(valve_2_intake.identificator.id)
                + valves_3.system_state.read_value(valve_3_intake.identificator.id)
                + valves_4.system_state.read_value(valve_4_intake.identificator.id),
            )
        ),
        label="manifold-to-cylinder-1-air",
        port_id=manifold_to_cylinder_1_port.identificator.id,
    )
    manifold_to_cylinder_2 = ContinuousFlowRelation(
        systemic_space,
        intake_manifold.identificator.id,
        manifold_air_mass.identificator.id,
        cylinder_2_flow.identificator.id,
        cylinder_2_air_mass.identificator.id,
        rate=lambda _time, source, _target: (
            min(
                cylinder_air_flow_rate_kg_s
                * intake_module.system_state.read_value(
                    throttle_position.identificator.id,
                ),
                0.999 * source / engine_sampling_interval,
            )
            * valves_2.system_state.read_value(valve_2_intake.identificator.id)
            / max(
                1.0,
                valves_1.system_state.read_value(valve_1_intake.identificator.id)
                + valves_2.system_state.read_value(valve_2_intake.identificator.id)
                + valves_3.system_state.read_value(valve_3_intake.identificator.id)
                + valves_4.system_state.read_value(valve_4_intake.identificator.id),
            )
        ),
        label="manifold-to-cylinder-2-air",
        port_id=manifold_to_cylinder_2_port.identificator.id,
    )
    manifold_to_cylinder_3 = ContinuousFlowRelation(
        systemic_space,
        intake_manifold.identificator.id,
        manifold_air_mass.identificator.id,
        cylinder_3_flow.identificator.id,
        cylinder_3_air_mass.identificator.id,
        rate=lambda _time, source, _target: (
            min(
                cylinder_air_flow_rate_kg_s
                * intake_module.system_state.read_value(
                    throttle_position.identificator.id,
                ),
                0.999 * source / engine_sampling_interval,
            )
            * valves_3.system_state.read_value(valve_3_intake.identificator.id)
            / max(
                1.0,
                valves_1.system_state.read_value(valve_1_intake.identificator.id)
                + valves_2.system_state.read_value(valve_2_intake.identificator.id)
                + valves_3.system_state.read_value(valve_3_intake.identificator.id)
                + valves_4.system_state.read_value(valve_4_intake.identificator.id),
            )
        ),
        label="manifold-to-cylinder-3-air",
        port_id=manifold_to_cylinder_3_port.identificator.id,
    )
    manifold_to_cylinder_4 = ContinuousFlowRelation(
        systemic_space,
        intake_manifold.identificator.id,
        manifold_air_mass.identificator.id,
        cylinder_4_flow.identificator.id,
        cylinder_4_air_mass.identificator.id,
        rate=lambda _time, source, _target: (
            min(
                cylinder_air_flow_rate_kg_s
                * intake_module.system_state.read_value(
                    throttle_position.identificator.id,
                ),
                0.999 * source / engine_sampling_interval,
            )
            * valves_4.system_state.read_value(valve_4_intake.identificator.id)
            / max(
                1.0,
                valves_1.system_state.read_value(valve_1_intake.identificator.id)
                + valves_2.system_state.read_value(valve_2_intake.identificator.id)
                + valves_3.system_state.read_value(valve_3_intake.identificator.id)
                + valves_4.system_state.read_value(valve_4_intake.identificator.id),
            )
        ),
        label="manifold-to-cylinder-4-air",
        port_id=manifold_to_cylinder_4_port.identificator.id,
    )
    fuel_to_cylinder_1 = ContinuousFlowRelation(
        systemic_space,
        fuel_tank.identificator.id,
        fuel_tank_mass.identificator.id,
        cylinder_1_flow.identificator.id,
        cylinder_1_fuel_mass.identificator.id,
        rate=lambda _time, source, _target: (
            min(
                cylinder_air_flow_rate_kg_s
                / stoichiometric_air_fuel_ratio
                * intake_module.system_state.read_value(
                    throttle_position.identificator.id,
                ),
                0.999 * source / engine_sampling_interval,
            )
            * valves_1.system_state.read_value(valve_1_intake.identificator.id)
            / max(
                1.0,
                valves_1.system_state.read_value(valve_1_intake.identificator.id)
                + valves_2.system_state.read_value(valve_2_intake.identificator.id)
                + valves_3.system_state.read_value(valve_3_intake.identificator.id)
                + valves_4.system_state.read_value(valve_4_intake.identificator.id),
            )
        ),
        label="injector-to-cylinder-1-fuel",
        port_id=fuel_to_cylinder_1_port.identificator.id,
    )
    fuel_to_cylinder_2 = ContinuousFlowRelation(
        systemic_space,
        fuel_tank.identificator.id,
        fuel_tank_mass.identificator.id,
        cylinder_2_flow.identificator.id,
        cylinder_2_fuel_mass.identificator.id,
        rate=lambda _time, source, _target: (
            min(
                cylinder_air_flow_rate_kg_s
                / stoichiometric_air_fuel_ratio
                * intake_module.system_state.read_value(
                    throttle_position.identificator.id,
                ),
                0.999 * source / engine_sampling_interval,
            )
            * valves_2.system_state.read_value(valve_2_intake.identificator.id)
            / max(
                1.0,
                valves_1.system_state.read_value(valve_1_intake.identificator.id)
                + valves_2.system_state.read_value(valve_2_intake.identificator.id)
                + valves_3.system_state.read_value(valve_3_intake.identificator.id)
                + valves_4.system_state.read_value(valve_4_intake.identificator.id),
            )
        ),
        label="injector-to-cylinder-2-fuel",
        port_id=fuel_to_cylinder_2_port.identificator.id,
    )
    fuel_to_cylinder_3 = ContinuousFlowRelation(
        systemic_space,
        fuel_tank.identificator.id,
        fuel_tank_mass.identificator.id,
        cylinder_3_flow.identificator.id,
        cylinder_3_fuel_mass.identificator.id,
        rate=lambda _time, source, _target: (
            min(
                cylinder_air_flow_rate_kg_s
                / stoichiometric_air_fuel_ratio
                * intake_module.system_state.read_value(
                    throttle_position.identificator.id,
                ),
                0.999 * source / engine_sampling_interval,
            )
            * valves_3.system_state.read_value(valve_3_intake.identificator.id)
            / max(
                1.0,
                valves_1.system_state.read_value(valve_1_intake.identificator.id)
                + valves_2.system_state.read_value(valve_2_intake.identificator.id)
                + valves_3.system_state.read_value(valve_3_intake.identificator.id)
                + valves_4.system_state.read_value(valve_4_intake.identificator.id),
            )
        ),
        label="injector-to-cylinder-3-fuel",
        port_id=fuel_to_cylinder_3_port.identificator.id,
    )
    fuel_to_cylinder_4 = ContinuousFlowRelation(
        systemic_space,
        fuel_tank.identificator.id,
        fuel_tank_mass.identificator.id,
        cylinder_4_flow.identificator.id,
        cylinder_4_fuel_mass.identificator.id,
        rate=lambda _time, source, _target: (
            min(
                cylinder_air_flow_rate_kg_s
                / stoichiometric_air_fuel_ratio
                * intake_module.system_state.read_value(
                    throttle_position.identificator.id,
                ),
                0.999 * source / engine_sampling_interval,
            )
            * valves_4.system_state.read_value(valve_4_intake.identificator.id)
            / max(
                1.0,
                valves_1.system_state.read_value(valve_1_intake.identificator.id)
                + valves_2.system_state.read_value(valve_2_intake.identificator.id)
                + valves_3.system_state.read_value(valve_3_intake.identificator.id)
                + valves_4.system_state.read_value(valve_4_intake.identificator.id),
            )
        ),
        label="injector-to-cylinder-4-fuel",
        port_id=fuel_to_cylinder_4_port.identificator.id,
    )
    intake_flow_bus.add_relation(ambient_to_manifold.identificator.id)
    intake_flow_bus.add_relation(manifold_to_cylinder_1.identificator.id)
    intake_flow_bus.add_relation(manifold_to_cylinder_2.identificator.id)
    intake_flow_bus.add_relation(manifold_to_cylinder_3.identificator.id)
    intake_flow_bus.add_relation(manifold_to_cylinder_4.identificator.id)
    intake_flow_bus.add_relation(fuel_to_cylinder_1.identificator.id)
    intake_flow_bus.add_relation(fuel_to_cylinder_2.identificator.id)
    intake_flow_bus.add_relation(fuel_to_cylinder_3.identificator.id)
    intake_flow_bus.add_relation(fuel_to_cylinder_4.identificator.id)
    intake_system.suscribe_systemic_description(intake_module.system_state.identificator.id)
    intake_system.suscribe_systemic_description(ambient_air.identificator.id)
    intake_system.suscribe_systemic_description(ambient_air_state.identificator.id)
    intake_system.suscribe_systemic_description(intake_manifold.identificator.id)
    intake_system.suscribe_systemic_description(intake_manifold_state.identificator.id)
    intake_system.suscribe_systemic_description(fuel_tank.identificator.id)
    intake_system.suscribe_systemic_description(fuel_tank_state.identificator.id)
    intake_system.suscribe_systemic_description(cylinder_1_flow.identificator.id)
    intake_system.suscribe_systemic_description(cylinder_1_flow_state.identificator.id)
    intake_system.suscribe_systemic_description(cylinder_2_flow.identificator.id)
    intake_system.suscribe_systemic_description(cylinder_2_flow_state.identificator.id)
    intake_system.suscribe_systemic_description(cylinder_3_flow.identificator.id)
    intake_system.suscribe_systemic_description(cylinder_3_flow_state.identificator.id)
    intake_system.suscribe_systemic_description(cylinder_4_flow.identificator.id)
    intake_system.suscribe_systemic_description(cylinder_4_flow_state.identificator.id)
    intake_system.suscribe_systemic_description(intake_flow_bus.identificator.id)
    intake_system.semantics.add_part(intake_system.identificator.id, ambient_air.identificator.id)
    intake_system.semantics.add_part(
        intake_system.identificator.id,
        intake_manifold.identificator.id,
    )
    intake_system.semantics.add_part(intake_system.identificator.id, fuel_tank.identificator.id)
    intake_system.semantics.add_part(
        intake_system.identificator.id,
        cylinder_1_flow.identificator.id,
    )
    intake_system.semantics.add_part(
        intake_system.identificator.id,
        cylinder_2_flow.identificator.id,
    )
    intake_system.semantics.add_part(
        intake_system.identificator.id,
        cylinder_3_flow.identificator.id,
    )
    intake_system.semantics.add_part(
        intake_system.identificator.id,
        cylinder_4_flow.identificator.id,
    )
    intake_input_ports = intake_system.semantics.define_set("Intake inputs")
    intake_output_ports = intake_system.semantics.define_set("Intake outputs")
    intake_system.semantics.add_member(
        intake_input_ports.id,
        ambient_to_manifold_port.identificator.id,
    )
    intake_system.semantics.add_member(
        intake_output_ports.id,
        manifold_to_cylinder_1_port.identificator.id,
    )
    intake_system.semantics.add_member(
        intake_output_ports.id,
        manifold_to_cylinder_2_port.identificator.id,
    )
    intake_system.semantics.add_member(
        intake_output_ports.id,
        manifold_to_cylinder_3_port.identificator.id,
    )
    intake_system.semantics.add_member(
        intake_output_ports.id,
        manifold_to_cylinder_4_port.identificator.id,
    )
    intake_system.semantics.add_member(
        intake_output_ports.id,
        fuel_to_cylinder_1_port.identificator.id,
    )
    intake_system.semantics.add_member(
        intake_output_ports.id,
        fuel_to_cylinder_2_port.identificator.id,
    )
    intake_system.semantics.add_member(
        intake_output_ports.id,
        fuel_to_cylinder_3_port.identificator.id,
    )
    intake_system.semantics.add_member(
        intake_output_ports.id,
        fuel_to_cylinder_4_port.identificator.id,
    )

    cylinder_1_to_exhaust = ContinuousFlowRelation(
        systemic_space,
        cylinder_1_flow.identificator.id,
        cylinder_1_air_mass.identificator.id,
        exhaust_manifold.identificator.id,
        exhaust_air_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_flow_rate_kg_s
            * valves_1.system_state.read_value(valve_1_exhaust.identificator.id),
            0.999 * source / engine_sampling_interval,
        ),
        label="cylinder-1-to-exhaust-air",
        port_id=cylinder_1_to_exhaust_port.identificator.id,
    )
    cylinder_1_fuel_to_exhaust = ContinuousFlowRelation(
        systemic_space,
        cylinder_1_flow.identificator.id,
        cylinder_1_fuel_mass.identificator.id,
        exhaust_manifold.identificator.id,
        exhaust_fuel_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_flow_rate_kg_s
            * valves_1.system_state.read_value(valve_1_exhaust.identificator.id),
            0.999 * source / engine_sampling_interval,
        ),
        label="cylinder-1-to-exhaust-fuel",
        port_id=cylinder_1_to_exhaust_port.identificator.id,
    )
    cylinder_2_to_exhaust = ContinuousFlowRelation(
        systemic_space,
        cylinder_2_flow.identificator.id,
        cylinder_2_air_mass.identificator.id,
        exhaust_manifold.identificator.id,
        exhaust_air_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_flow_rate_kg_s
            * valves_2.system_state.read_value(valve_2_exhaust.identificator.id),
            0.999 * source / engine_sampling_interval,
        ),
        label="cylinder-2-to-exhaust-air",
        port_id=cylinder_2_to_exhaust_port.identificator.id,
    )
    cylinder_2_fuel_to_exhaust = ContinuousFlowRelation(
        systemic_space,
        cylinder_2_flow.identificator.id,
        cylinder_2_fuel_mass.identificator.id,
        exhaust_manifold.identificator.id,
        exhaust_fuel_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_flow_rate_kg_s
            * valves_2.system_state.read_value(valve_2_exhaust.identificator.id),
            0.999 * source / engine_sampling_interval,
        ),
        label="cylinder-2-to-exhaust-fuel",
        port_id=cylinder_2_to_exhaust_port.identificator.id,
    )
    cylinder_3_to_exhaust = ContinuousFlowRelation(
        systemic_space,
        cylinder_3_flow.identificator.id,
        cylinder_3_air_mass.identificator.id,
        exhaust_manifold.identificator.id,
        exhaust_air_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_flow_rate_kg_s
            * valves_3.system_state.read_value(valve_3_exhaust.identificator.id),
            0.999 * source / engine_sampling_interval,
        ),
        label="cylinder-3-to-exhaust-air",
        port_id=cylinder_3_to_exhaust_port.identificator.id,
    )
    cylinder_3_fuel_to_exhaust = ContinuousFlowRelation(
        systemic_space,
        cylinder_3_flow.identificator.id,
        cylinder_3_fuel_mass.identificator.id,
        exhaust_manifold.identificator.id,
        exhaust_fuel_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_flow_rate_kg_s
            * valves_3.system_state.read_value(valve_3_exhaust.identificator.id),
            0.999 * source / engine_sampling_interval,
        ),
        label="cylinder-3-to-exhaust-fuel",
        port_id=cylinder_3_to_exhaust_port.identificator.id,
    )
    cylinder_4_to_exhaust = ContinuousFlowRelation(
        systemic_space,
        cylinder_4_flow.identificator.id,
        cylinder_4_air_mass.identificator.id,
        exhaust_manifold.identificator.id,
        exhaust_air_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_flow_rate_kg_s
            * valves_4.system_state.read_value(valve_4_exhaust.identificator.id),
            0.999 * source / engine_sampling_interval,
        ),
        label="cylinder-4-to-exhaust-air",
        port_id=cylinder_4_to_exhaust_port.identificator.id,
    )
    cylinder_4_fuel_to_exhaust = ContinuousFlowRelation(
        systemic_space,
        cylinder_4_flow.identificator.id,
        cylinder_4_fuel_mass.identificator.id,
        exhaust_manifold.identificator.id,
        exhaust_fuel_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_flow_rate_kg_s
            * valves_4.system_state.read_value(valve_4_exhaust.identificator.id),
            0.999 * source / engine_sampling_interval,
        ),
        label="cylinder-4-to-exhaust-fuel",
        port_id=cylinder_4_to_exhaust_port.identificator.id,
    )
    exhaust_to_tailpipe_air = ContinuousFlowRelation(
        systemic_space,
        exhaust_manifold.identificator.id,
        exhaust_air_mass.identificator.id,
        tailpipe.identificator.id,
        exhaust_outlet_air_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_outlet_rate_kg_s,
            0.999 * source / engine_sampling_interval,
        ),
        label="exhaust-air-outlet",
        port_id=exhaust_to_tailpipe_port.identificator.id,
    )
    exhaust_to_tailpipe_fuel = ContinuousFlowRelation(
        systemic_space,
        exhaust_manifold.identificator.id,
        exhaust_fuel_mass.identificator.id,
        tailpipe.identificator.id,
        exhaust_outlet_fuel_mass.identificator.id,
        rate=lambda _time, source, _target: min(
            exhaust_outlet_rate_kg_s,
            0.999 * source / engine_sampling_interval,
        ),
        label="exhaust-fuel-outlet",
        port_id=exhaust_to_tailpipe_port.identificator.id,
    )
    exhaust_flow_bus = FlowBusStructure(systemic_space)
    exhaust_flow_bus.add_node(cylinder_1_flow.identificator.id)
    exhaust_flow_bus.add_node(cylinder_2_flow.identificator.id)
    exhaust_flow_bus.add_node(cylinder_3_flow.identificator.id)
    exhaust_flow_bus.add_node(cylinder_4_flow.identificator.id)
    exhaust_flow_bus.add_node(exhaust_manifold.identificator.id)
    exhaust_flow_bus.add_node(tailpipe.identificator.id)
    exhaust_flow_bus.add_relation(cylinder_1_to_exhaust.identificator.id)
    exhaust_flow_bus.add_relation(cylinder_1_fuel_to_exhaust.identificator.id)
    exhaust_flow_bus.add_relation(cylinder_2_to_exhaust.identificator.id)
    exhaust_flow_bus.add_relation(cylinder_2_fuel_to_exhaust.identificator.id)
    exhaust_flow_bus.add_relation(cylinder_3_to_exhaust.identificator.id)
    exhaust_flow_bus.add_relation(cylinder_3_fuel_to_exhaust.identificator.id)
    exhaust_flow_bus.add_relation(cylinder_4_to_exhaust.identificator.id)
    exhaust_flow_bus.add_relation(cylinder_4_fuel_to_exhaust.identificator.id)
    exhaust_flow_bus.add_relation(exhaust_to_tailpipe_air.identificator.id)
    exhaust_flow_bus.add_relation(exhaust_to_tailpipe_fuel.identificator.id)
    exhaust_system.suscribe_systemic_description(exhaust_module.system_state.identificator.id)
    exhaust_system.suscribe_systemic_description(cylinder_1_flow.identificator.id)
    exhaust_system.suscribe_systemic_description(cylinder_2_flow.identificator.id)
    exhaust_system.suscribe_systemic_description(cylinder_3_flow.identificator.id)
    exhaust_system.suscribe_systemic_description(cylinder_4_flow.identificator.id)
    exhaust_system.suscribe_systemic_description(exhaust_manifold.identificator.id)
    exhaust_system.suscribe_systemic_description(exhaust_manifold_state.identificator.id)
    exhaust_system.suscribe_systemic_description(tailpipe.identificator.id)
    exhaust_system.suscribe_systemic_description(tailpipe_state.identificator.id)
    exhaust_system.suscribe_systemic_description(exhaust_flow_bus.identificator.id)
    exhaust_system.semantics.add_part(
        exhaust_system.identificator.id,
        cylinder_1_flow.identificator.id,
    )
    exhaust_system.semantics.add_part(
        exhaust_system.identificator.id,
        cylinder_2_flow.identificator.id,
    )
    exhaust_system.semantics.add_part(
        exhaust_system.identificator.id,
        cylinder_3_flow.identificator.id,
    )
    exhaust_system.semantics.add_part(
        exhaust_system.identificator.id,
        cylinder_4_flow.identificator.id,
    )
    exhaust_system.semantics.add_part(
        exhaust_system.identificator.id,
        exhaust_manifold.identificator.id,
    )
    exhaust_system.semantics.add_part(
        exhaust_system.identificator.id,
        tailpipe.identificator.id,
    )
    exhaust_input_ports = exhaust_system.semantics.define_set("Exhaust inputs")
    exhaust_output_ports = exhaust_system.semantics.define_set("Exhaust outputs")
    exhaust_system.semantics.add_member(
        exhaust_input_ports.id,
        cylinder_1_to_exhaust_port.identificator.id,
    )
    exhaust_system.semantics.add_member(
        exhaust_input_ports.id,
        cylinder_2_to_exhaust_port.identificator.id,
    )
    exhaust_system.semantics.add_member(
        exhaust_input_ports.id,
        cylinder_3_to_exhaust_port.identificator.id,
    )
    exhaust_system.semantics.add_member(
        exhaust_input_ports.id,
        cylinder_4_to_exhaust_port.identificator.id,
    )
    exhaust_system.semantics.add_member(
        exhaust_output_ports.id,
        exhaust_to_tailpipe_port.identificator.id,
    )

    # QualitativeDomain clasifica el ángulo local en las cuatro carreras;
    # QualitativeAutomaton declara el orden cíclico entre ellas.
    four_stroke_domain = engine.semantics.define_qualitative_domain(
        "four-stroke phase",
        ("intake", "compression", "power", "exhaust"),
    )
    engine.semantics.map_qualitative_interval(four_stroke_domain.id, 0, 180, "intake")
    engine.semantics.map_qualitative_interval(four_stroke_domain.id, 180, 360, "compression")
    engine.semantics.map_qualitative_interval(four_stroke_domain.id, 360, 540, "power")
    engine.semantics.map_qualitative_interval(four_stroke_domain.id, 540, 720, "exhaust")
    four_stroke_automaton = engine.semantics.define_automaton(
        "four-stroke cycle",
        ("intake", "compression", "power", "exhaust"),
        ("advance",),
        "intake",
        {
            ("intake", "advance"): "compression",
            ("compression", "advance"): "power",
            ("power", "advance"): "exhaust",
            ("exhaust", "advance"): "intake",
        },
    )

    crank_interval = ProcessTrigger(
        systemic_space, ProcessTriggerKind.INTERVAL,
        interval=engine_sampling_interval, initial_delay=0,
    )
    engine_process = Process(
        systemic_space, engine.identificator.id, label="L4 OHV cycle",
    )
    propagate_couplings = ProcessAction(
        systemic_space, ProcessActionKind.PROPAGATE_STATE_COUPLINGS,
        structure_id=coupling.identificator.id,
    )
    engine_process.add_trigger(crank_interval.identificator.id)
    engine_process.add_action(propagate_couplings.identificator.id)
    engine.suscribe_systemic_description(engine_process.identificator.id)

    timing_chain_trigger = ProcessTrigger(
        systemic_space,
        ProcessTriggerKind.INTERVAL,
        interval=engine_sampling_interval,
        initial_delay=0,
    )
    timing_chain_propagation = ProcessAction(
        systemic_space,
        ProcessActionKind.PROPAGATE_STATE_COUPLINGS,
        structure_id=timing_drive.identificator.id,
    )
    timing_chain_process = Process(
        systemic_space,
        timing_chain_system.identificator.id,
        label="Timing-chain kinematics",
    )
    timing_chain_process.add_trigger(timing_chain_trigger.identificator.id)
    timing_chain_process.add_action(timing_chain_propagation.identificator.id)
    timing_chain_system.suscribe_systemic_description(
        timing_chain_process.identificator.id,
    )

    intake_flow_trigger = ProcessTrigger(
        systemic_space,
        ProcessTriggerKind.INTERVAL,
        interval=engine_sampling_interval,
        initial_delay=0,
    )
    advance_intake_flow = ProcessAction(
        systemic_space,
        ProcessActionKind.ADVANCE_FLOW_BUS,
        structure_id=intake_flow_bus.identificator.id,
        duration=engine_sampling_interval,
    )
    intake_flow_process = Process(
        systemic_space,
        intake_system.identificator.id,
        label="Intake continuous flows",
    )
    intake_flow_process.add_trigger(intake_flow_trigger.identificator.id)
    intake_flow_process.add_action(advance_intake_flow.identificator.id)
    intake_system.suscribe_systemic_description(intake_flow_process.identificator.id)

    exhaust_flow_trigger = ProcessTrigger(
        systemic_space,
        ProcessTriggerKind.INTERVAL,
        interval=engine_sampling_interval,
        initial_delay=0,
    )
    advance_exhaust_flow = ProcessAction(
        systemic_space,
        ProcessActionKind.ADVANCE_FLOW_BUS,
        structure_id=exhaust_flow_bus.identificator.id,
        duration=engine_sampling_interval,
    )
    exhaust_flow_process = Process(
        systemic_space,
        exhaust_system.identificator.id,
        label="Exhaust continuous flows",
    )
    exhaust_flow_process.add_trigger(exhaust_flow_trigger.identificator.id)
    exhaust_flow_process.add_action(advance_exhaust_flow.identificator.id)
    exhaust_system.suscribe_systemic_description(exhaust_flow_process.identificator.id)

    engine_start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    starter_trigger = ProcessTrigger(
        systemic_space,
        ProcessTriggerKind.CALENDAR,
        scheduled_at=engine_start,
    )
    starter_speed = ProcessExpression(
        systemic_space,
        ProcessExpressionOperator.CONSTANT,
        value=7200,
    )
    start_crankshaft = ProcessAction(
        systemic_space,
        ProcessActionKind.SET_STATE_VALUE,
        expression_id=starter_speed.identificator.id,
        state_id=crankshaft.system_state.identificator.id,
        element_id=crank_velocity.identificator.id,
    )
    starter_process = Process(
        systemic_space,
        engine.identificator.id,
        label="Calendar starter",
    )
    starter_process.add_trigger(starter_trigger.identificator.id)
    starter_process.add_action(start_crankshaft.identificator.id)
    engine.suscribe_systemic_description(starter_process.identificator.id)

    throttle_trigger = ProcessTrigger(
        systemic_space,
        ProcessTriggerKind.CALENDAR,
        scheduled_at=engine_start + timedelta(seconds=0.25),
    )
    open_throttle_value = ProcessExpression(
        systemic_space,
        ProcessExpressionOperator.CONSTANT,
        value=0.75,
    )
    open_throttle = ProcessAction(
        systemic_space,
        ProcessActionKind.SET_STATE_VALUE,
        expression_id=open_throttle_value.identificator.id,
        state_id=intake_module.system_state.identificator.id,
        element_id=throttle_position.identificator.id,
    )
    throttle_process = Process(
        systemic_space,
        engine.identificator.id,
        label="Driver throttle step",
    )
    throttle_process.add_trigger(throttle_trigger.identificator.id)
    throttle_process.add_action(open_throttle.identificator.id)
    engine.suscribe_systemic_description(throttle_process.identificator.id)

    crank_angle_ref = ProcessValueRef(
        systemic_space,
        ProcessValueScope.SYSTEM_STATE,
        element_id=crank_angle.identificator.id,
        state_id=crankshaft.system_state.identificator.id,
    )
    crank_angle_value = ProcessExpression(
        systemic_space,
        ProcessExpressionOperator.READ,
        value_ref_id=crank_angle_ref.identificator.id,
    )
    cycle_count_ref = ProcessValueRef(
        systemic_space,
        ProcessValueScope.SYSTEM_STATE,
        element_id=engine_cycles.identificator.id,
        state_id=crankshaft.system_state.identificator.id,
    )
    cycle_count_value = ProcessExpression(
        systemic_space,
        ProcessExpressionOperator.READ,
        value_ref_id=cycle_count_ref.identificator.id,
    )
    one_cycle = ProcessExpression(
        systemic_space,
        ProcessExpressionOperator.CONSTANT,
        value=1,
    )
    next_cycle_count = ProcessExpression(
        systemic_space,
        ProcessExpressionOperator.ADD,
        operands=(cycle_count_value.identificator.id, one_cycle.identificator.id),
    )
    degrees_per_cycle = ProcessExpression(
        systemic_space,
        ProcessExpressionOperator.CONSTANT,
        value=720,
    )
    cycle_limit = ProcessExpression(
        systemic_space,
        ProcessExpressionOperator.MULTIPLY,
        operands=(
            next_cycle_count.identificator.id,
            degrees_per_cycle.identificator.id,
        ),
    )
    cycle_complete = ProcessCondition(
        systemic_space,
        ProcessConditionOperator.GREATER_EQUAL,
        (crank_angle_value.identificator.id, cycle_limit.identificator.id),
    )
    cycle_trigger = ProcessTrigger(
        systemic_space,
        ProcessTriggerKind.CONDITION,
        condition_id=cycle_complete.identificator.id,
    )
    increment_cycle_count = ProcessAction(
        systemic_space,
        ProcessActionKind.SET_STATE_VALUE,
        expression_id=next_cycle_count.identificator.id,
        state_id=crankshaft.system_state.identificator.id,
        element_id=engine_cycles.identificator.id,
    )
    cycle_process = Process(
        systemic_space,
        engine.identificator.id,
        label="Four-stroke cycle boundary",
    )
    cycle_process.add_trigger(cycle_trigger.identificator.id)
    cycle_process.add_action(increment_cycle_count.identificator.id)
    engine.suscribe_systemic_description(cycle_process.identificator.id)

    timing_drive.propagate()
    coupling.propagate()
    engine_history = engine.enable_history()
    engine_simulation = engine.run_simpy(
        until=0.000001,
        environment=sim.Environment(),
        calendar_origin=engine_start,
        condition_interval=engine_sampling_interval,
    )
    timing_chain_simulation = timing_chain_system.run_simpy(
        until=0.000001,
        environment=sim.Environment(),
        calendar_origin=engine_start,
        condition_interval=engine_sampling_interval,
    )
    intake_simulation = intake_system.run_simpy(
        until=0.000001,
        environment=sim.Environment(),
        calendar_origin=engine_start,
        condition_interval=engine_sampling_interval,
    )
    exhaust_simulation = exhaust_system.run_simpy(
        until=0.000001,
        environment=sim.Environment(),
        calendar_origin=engine_start,
        condition_interval=engine_sampling_interval,
    )
    if crankshaft.system_state.read_value(crank_velocity.identificator.id) <= 0:
        raise RuntimeError("The calendar starter did not activate the crankshaft.")
    crankshaft.system_state.set_discrete_value(
        engine_phase.identificator.id,
        "intake",
    )

    def crankshaft_derivatives(_time, values):
        crank_angle_deg = values[0]
        crank_velocity_deg_s = values[1]
        cylinder_1_phase_deg = (crank_angle_deg + 360) % 720
        cylinder_2_phase_deg = (crank_angle_deg + 540) % 720
        cylinder_3_phase_deg = (crank_angle_deg + 180) % 720
        cylinder_4_phase_deg = crank_angle_deg % 720
        cylinder_1_fuel_energy = indicated_thermal_efficiency * min(
            cylinder_1_flow_state.read_value(cylinder_1_fuel_mass.identificator.id)
            * fuel_lower_heating_value_j_kg,
            cylinder_1_flow_state.read_value(cylinder_1_air_mass.identificator.id)
            / stoichiometric_air_fuel_ratio
            * fuel_lower_heating_value_j_kg,
        )
        cylinder_2_fuel_energy = indicated_thermal_efficiency * min(
            cylinder_2_flow_state.read_value(cylinder_2_fuel_mass.identificator.id)
            * fuel_lower_heating_value_j_kg,
            cylinder_2_flow_state.read_value(cylinder_2_air_mass.identificator.id)
            / stoichiometric_air_fuel_ratio
            * fuel_lower_heating_value_j_kg,
        )
        cylinder_3_fuel_energy = indicated_thermal_efficiency * min(
            cylinder_3_flow_state.read_value(cylinder_3_fuel_mass.identificator.id)
            * fuel_lower_heating_value_j_kg,
            cylinder_3_flow_state.read_value(cylinder_3_air_mass.identificator.id)
            / stoichiometric_air_fuel_ratio
            * fuel_lower_heating_value_j_kg,
        )
        cylinder_4_fuel_energy = indicated_thermal_efficiency * min(
            cylinder_4_flow_state.read_value(cylinder_4_fuel_mass.identificator.id)
            * fuel_lower_heating_value_j_kg,
            cylinder_4_flow_state.read_value(cylinder_4_air_mass.identificator.id)
            / stoichiometric_air_fuel_ratio
            * fuel_lower_heating_value_j_kg,
        )
        cylinder_1_torque = (
            0.5
            * cylinder_1_fuel_energy
            * np.sin(np.deg2rad(cylinder_1_phase_deg - 360))
            if 360 <= cylinder_1_phase_deg <= 540
            else 0.0
        )
        cylinder_2_torque = (
            0.5
            * cylinder_2_fuel_energy
            * np.sin(np.deg2rad(cylinder_2_phase_deg - 360))
            if 360 <= cylinder_2_phase_deg <= 540
            else 0.0
        )
        cylinder_3_torque = (
            0.5
            * cylinder_3_fuel_energy
            * np.sin(np.deg2rad(cylinder_3_phase_deg - 360))
            if 360 <= cylinder_3_phase_deg <= 540
            else 0.0
        )
        cylinder_4_torque = (
            0.5
            * cylinder_4_fuel_energy
            * np.sin(np.deg2rad(cylinder_4_phase_deg - 360))
            if 360 <= cylinder_4_phase_deg <= 540
            else 0.0
        )
        combustion_torque = (
            cylinder_1_torque
            + cylinder_2_torque
            + cylinder_3_torque
            + cylinder_4_torque
        )
        angular_velocity_rad_s = np.deg2rad(crank_velocity_deg_s)
        friction_torque = (
            friction_torque_nm
            + viscous_friction_nm_per_rad_s * angular_velocity_rad_s
        )
        net_torque = combustion_torque - friction_torque - external_load_torque_nm
        delivered_torque = max(0.0, net_torque)
        return (
            crank_velocity_deg_s,
            np.rad2deg(net_torque / crankshaft_inertia_kg_m2),
            delivered_torque * angular_velocity_rad_s,
        )

    crank_integrator = ODEIntegrator(
        crankshaft.system_state,
        (
            crank_angle.identificator.id,
            crank_velocity.identificator.id,
            shaft_energy.identificator.id,
        ),
        crankshaft_derivatives,
    )
    engine_step = engine_sampling_interval
    engine_duration = 1.0
    engine_steps = int(engine_duration / engine_step)
    time_samples = [0.0]
    speed_samples_rpm = [
        crankshaft.system_state.read_value(crank_velocity.identificator.id) / 6
    ]
    energy_samples_j = [
        crankshaft.system_state.read_value(shaft_energy.identificator.id)
    ]
    power_samples_w = [0.0]
    pressure_1_samples_bar = [
        combustion_1.system_state.read_value(pressure_1.identificator.id)
    ]
    manifold_air_samples_kg = [
        intake_manifold_state.read_value(manifold_air_mass.identificator.id)
    ]
    exhaust_outlet_samples_kg = [
        tailpipe_state.read_value(exhaust_outlet_air_mass.identificator.id)
        + tailpipe_state.read_value(exhaust_outlet_fuel_mass.identificator.id)
    ]
    print("\nL4 OHV: throttle-dependent four-stroke simulation")
    print("time | cycle | phase | rpm | crank | cam | cylinder 1 pressure | shaft power")
    for step in range(1, engine_steps + 1):
        step_start = (step - 1) * engine_step
        step_end = step * engine_step
        crank_trajectory = crank_integrator.integrate(
            (step_start, step_end),
            t_eval=(step_start, step_end),
        )
        timing_chain_simulation = timing_chain_system.run_simpy(
            until=step_end + 0.000001,
            environment=timing_chain_simulation,
            condition_interval=engine_sampling_interval,
        )
        engine_simulation = engine.run_simpy(
            until=step_end + 0.000001,
            environment=engine_simulation,
            condition_interval=engine_sampling_interval,
        )
        intake_simulation = intake_system.run_simpy(
            until=step_end + 0.000001,
            environment=intake_simulation,
            condition_interval=engine_sampling_interval,
        )
        exhaust_simulation = exhaust_system.run_simpy(
            until=step_end + 0.000001,
            environment=exhaust_simulation,
            condition_interval=engine_sampling_interval,
        )

        energy_now = crankshaft.system_state.read_value(shaft_energy.identificator.id)
        time_samples.append(step_end)
        speed_samples_rpm.append(
            crankshaft.system_state.read_value(crank_velocity.identificator.id) / 6
        )
        energy_samples_j.append(energy_now)
        power_samples_w.append(
            (energy_now - energy_samples_j[-2]) / engine_step
        )
        pressure_1_samples_bar.append(
            combustion_1.system_state.read_value(pressure_1.identificator.id)
        )
        manifold_air_samples_kg.append(
            intake_manifold_state.read_value(manifold_air_mass.identificator.id)
        )
        exhaust_outlet_samples_kg.append(
            tailpipe_state.read_value(exhaust_outlet_air_mass.identificator.id)
            + tailpipe_state.read_value(exhaust_outlet_fuel_mass.identificator.id)
        )

        phase_angle = crankshaft.system_state.read_value(crank_angle.identificator.id) % 720
        if phase_angle < 180:
            crankshaft.system_state.set_discrete_value(
                engine_phase.identificator.id,
                "intake",
            )
        elif phase_angle < 360:
            crankshaft.system_state.set_discrete_value(
                engine_phase.identificator.id,
                "compression",
            )
        elif phase_angle < 540:
            crankshaft.system_state.set_discrete_value(
                engine_phase.identificator.id,
                "power",
            )
        else:
            crankshaft.system_state.set_discrete_value(
                engine_phase.identificator.id,
                "exhaust",
            )

        simulated_datetime = engine.simulation_datetime
        if simulated_datetime is None:
            raise RuntimeError("The calendar clock was not configured.")
        if step % 40 == 0:
            print(
                f"{step_end:0.3f}s | "
                f"cycle {crankshaft.system_state.read_value(engine_cycles.identificator.id):0.0f} | "
                f"{crankshaft.system_state.read_discrete_value(engine_phase.identificator.id)} | "
                f"{speed_samples_rpm[-1]:0.0f} rpm | "
                f"crank {phase_angle:06.2f} deg | "
                f"cam {camshaft.system_state.read_value(cam_angle.identificator.id):06.2f} deg | "
                f"C1 {pressure_1_samples_bar[-1]:04.2f} bar | "
                f"{power_samples_w[-1]:0.0f} W | "
                f"ODE {len(crank_trajectory.times)} samples"
            )

    final_simulated_datetime = engine.simulation_datetime
    if final_simulated_datetime is None:
        raise RuntimeError("The calendar clock was not configured.")
    print("\nTemporal summary")
    print(
        "Completed cycles:",
        crankshaft.system_state.read_value(engine_cycles.identificator.id),
    )
    print("Final simulated date:", final_simulated_datetime.isoformat())
    initial_speed_rpm = speed_samples_rpm[0]
    final_speed_rpm = speed_samples_rpm[-1]
    produced_energy_j = energy_samples_j[-1]
    mean_shaft_power_w = produced_energy_j / engine_duration
    peak_shaft_power_w = max(power_samples_w)
    consumed_air_kg = 2.0 - ambient_air_state.read_value(ambient_air_mass.identificator.id)
    injected_fuel_kg = 0.25 - fuel_tank_state.read_value(fuel_tank_mass.identificator.id)
    discharged_exhaust_kg = (
        tailpipe_state.read_value(exhaust_outlet_air_mass.identificator.id)
        + tailpipe_state.read_value(exhaust_outlet_fuel_mass.identificator.id)
    )
    expected_cam_angle = (
        crankshaft.system_state.read_value(crank_angle.identificator.id) % 720
    ) / 2
    actual_cam_angle = camshaft.system_state.read_value(cam_angle.identificator.id)
    if not np.isclose(actual_cam_angle, expected_cam_angle, atol=1e-6):
        raise RuntimeError("The timing chain lost the 2:1 crank-to-cam synchronization.")
    if final_speed_rpm <= initial_speed_rpm:
        raise RuntimeError("The throttle step did not accelerate the crankshaft.")
    if produced_energy_j <= 0 or mean_shaft_power_w <= 0:
        raise RuntimeError("The engine did not deliver positive shaft energy.")
    if consumed_air_kg <= 0 or injected_fuel_kg <= 0 or discharged_exhaust_kg <= 0:
        raise RuntimeError("The intake and exhaust flow paths did not transfer mass.")
    print(
        f"Engine speed: {initial_speed_rpm:.0f} -> {final_speed_rpm:.0f} rpm "
        f"(throttle {initial_throttle_position:.2f} -> "
        f"{intake_module.system_state.read_value(throttle_position.identificator.id):.2f})"
    )
    print(f"Valve estimate: 4 cylinders x 2 valves = {engine.semantic_attributes['total_valves']}")
    print(f"Intake air delivered: {consumed_air_kg:.6f} kg")
    print(f"Fuel injected: {injected_fuel_kg:.6f} kg")
    print(f"Exhaust discharged: {discharged_exhaust_kg:.6f} kg")
    print(f"Mechanical shaft energy: {produced_energy_j:.2f} J")
    print(f"Mean / peak shaft power: {mean_shaft_power_w:.2f} / {peak_shaft_power_w:.2f} W")
    print(f"Mechanical energy: {produced_energy_j / 3_600_000:.8f} kWh")
    print("Trajectory samples:", len(engine_history.samples))
    print("Discrete phase transitions:", len(engine_history.discrete_events))

    piston_1_trajectory = engine_history.trajectory(
        piston_1.system_state.identificator.id,
        piston_1_position.identificator.id,
    )
    piston_2_trajectory = engine_history.trajectory(
        piston_2.system_state.identificator.id,
        piston_2_position.identificator.id,
    )
    piston_3_trajectory = engine_history.trajectory(
        piston_3.system_state.identificator.id,
        piston_3_position.identificator.id,
    )
    piston_4_trajectory = engine_history.trajectory(
        piston_4.system_state.identificator.id,
        piston_4_position.identificator.id,
    )

    piston_figure, (
        piston_1_axis,
        piston_2_axis,
        piston_3_axis,
        piston_4_axis,
    ) = plt.subplots(
        4,
        1,
        sharex=True,
        num="L4 OHV piston trajectories",
    )
    piston_1_axis.plot(
        piston_1_trajectory.times,
        piston_1_trajectory.values[piston_1_position.identificator.id],
    )
    piston_1_axis.set_ylabel("Position (m)")
    piston_1_axis.set_title("Piston 1")
    piston_1_axis.grid(True)
    piston_2_axis.plot(
        piston_2_trajectory.times,
        piston_2_trajectory.values[piston_2_position.identificator.id],
    )
    piston_2_axis.set_ylabel("Position (m)")
    piston_2_axis.set_title("Piston 2")
    piston_2_axis.grid(True)
    piston_3_axis.plot(
        piston_3_trajectory.times,
        piston_3_trajectory.values[piston_3_position.identificator.id],
    )
    piston_3_axis.set_ylabel("Position (m)")
    piston_3_axis.set_title("Piston 3")
    piston_3_axis.grid(True)
    piston_4_axis.plot(
        piston_4_trajectory.times,
        piston_4_trajectory.values[piston_4_position.identificator.id],
    )
    piston_4_axis.set_ylabel("Position (m)")
    piston_4_axis.set_title("Piston 4")
    piston_4_axis.set_xlabel("Simulation time (s)")
    piston_4_axis.grid(True)
    piston_figure.suptitle("Piston motion")
    piston_figure.tight_layout()
    piston_figure.savefig("piston_trajectories.png", dpi=150)

    engine_figure, (
        speed_axis,
        power_axis,
        energy_axis,
        pressure_axis,
    ) = plt.subplots(
        4,
        1,
        sharex=True,
        num="L4 OHV engine response",
    )
    speed_axis.plot(time_samples, speed_samples_rpm)
    speed_axis.set_ylabel("Speed (rpm)")
    speed_axis.set_title("Throttle response")
    speed_axis.grid(True)
    power_axis.plot(time_samples, power_samples_w)
    power_axis.set_ylabel("Shaft power (W)")
    power_axis.grid(True)
    energy_axis.plot(time_samples, energy_samples_j)
    energy_axis.set_ylabel("Delivered energy (J)")
    energy_axis.grid(True)
    pressure_axis.plot(time_samples, pressure_1_samples_bar)
    pressure_axis.set_ylabel("C1 pressure (bar)")
    pressure_axis.set_xlabel("Simulation time (s)")
    pressure_axis.grid(True)
    engine_figure.suptitle("Crankshaft dynamics and output")
    engine_figure.tight_layout()
    engine_figure.savefig("engine_performance.png", dpi=150)

    flow_figure, (manifold_axis, exhaust_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        num="L4 OHV intake and exhaust flows",
    )
    manifold_axis.plot(time_samples, manifold_air_samples_kg)
    manifold_axis.set_ylabel("Manifold air (kg)")
    manifold_axis.set_title("Continuous flow inventories")
    manifold_axis.grid(True)
    exhaust_axis.plot(time_samples, exhaust_outlet_samples_kg)
    exhaust_axis.set_ylabel("Exhaust discharged (kg)")
    exhaust_axis.set_xlabel("Simulation time (s)")
    exhaust_axis.grid(True)
    flow_figure.tight_layout()
    flow_figure.savefig("engine_flows.png", dpi=150)
    plt.show()
