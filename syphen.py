# ==================================
# SYSTEMS PHYSICS ENGINE (SYPHEN)
# ==================================

""" Syphen, stands for Systemic Physics Engine, it´s functionality is to model systems through Topologies, 
States, Processes, Inputs and Outputs Processing and Semantics Domain powered by a set of specialized python libraries
 """

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from math import isfinite
from typing import ClassVar, Any, Optional, Callable, Optional, Sequence, Union
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
        self._expressions_by_symbol : dict[sym.Symbol, sym.Expr] = {}

        self.expressions : list[DependentExpressionElement] = []
        self.symbols : list[sym.Symbol] = []
        self._active_kernel: Optional[CompiledStateKernel] = None

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

        entity_element._owner_state_id = self.identificator.id
        self.elements[element] = entity_element
        if isinstance(entity_element, VariableElement):
            self._variables_by_name[entity_element.name] = entity_element
            self._active_kernel = None

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

    def read_element(
        self,
        element: UUID,
    ) -> Union[float, np.ndarray, sym.Lambda]:
        entity_element = self.elements.get(element)
        if isinstance(entity_element, VariableElement):
            return self.read_value(element)
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

    def set_values(self, updates: dict[UUID, float]) -> None:
        validated_updates = self.validate_values(updates)
        self._commit_validated_values(validated_updates)

    def _commit_validated_values(self, updates: dict[UUID, float]) -> None:
        for element, value in updates.items():
            variable = self.elements[element]
            if not isinstance(variable, VariableElement):
                raise TypeError("La variable validada dejó de pertenecer al estado.")
            variable._value = value

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

class ProcessValueScope(Enum):
    SYSTEM_STATE = auto()
    TOKEN_STATE = auto()
    SIMULATION_TIME = auto()

class ProcessExpressionOperator(Enum):
    CONSTANT = auto()
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
        value: Optional[float] = None,
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
    ) -> float:
        if self.operator is ProcessExpressionOperator.CONSTANT:
            if self.value is None or self.operands or self.value_ref_id is not None:
                raise ValueError("La expresión CONSTANT requiere solo un valor.")
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
            raw_value = state.read_element(reference.element_id)
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
        elif initial_delay is not None:
            raise ValueError("initial_delay solo aplica al trigger INTERVAL.")
        self.identificator = Identification(entity=Ontology.EntityType.PROCESS_TRIGGER)
        self.kind = kind
        self.processor_id = processor_id
        self.condition_id = condition_id
        self.interval = interval
        self.initial_delay = initial_delay
        self.space_auto_registry()

class ProcessAction(Entity):
    def __init__(
        self, systemic_space: SystemicSpace, kind: ProcessActionKind, expression_id: Optional[UUID] = None, state_id: Optional[UUID] = None,
        element_id: Optional[UUID] = None, processor_id: Optional[UUID] = None, port_id: Optional[UUID] = None, token_id: Optional[UUID] = None,
        structure_id: Optional[UUID] = None, relation_id: Optional[UUID] = None,
        component_id: Optional[UUID] = None,) -> None:
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

        updates: dict[SystemState, dict[UUID, float]] = {}
        emissions: list[tuple[SystemProcessor, UUID, UUID]] = []
        coupling_actions: list[tuple[StateCouplingStructure, UUID]] = []
        propagation_actions: list[StateCouplingStructure] = []
        
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

        validated_updates = {state: state.validate_values(state_updates) for state, state_updates in updates.items()}

        for state, state_updates in validated_updates.items():
            state._commit_validated_values(state_updates)
        for structure, relation_id in coupling_actions:
            structure.apply_relation(relation_id)
        for structure in propagation_actions:
            structure.propagate()
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

class StateCouplingRelation(SystemRelation):

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
        if not isinstance(source_element, VariableElement):
            raise TypeError("El origen debe ser una VariableElement del estado origen.")
        if not isinstance(target_element, VariableElement):
            raise TypeError("El destino debe ser una VariableElement del estado destino.")
        if not callable(transform):
            raise TypeError("transform debe ser una función escalar invocable.")
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
        self.transform = transform
        self.input_constraints = input_constraints or source_element.constraints
        self.output_constraints = output_constraints or target_element.constraints

    def evaluate(self) -> float:
        source_component = self.systemic_space.get(self.source_id)
        target_component = self.systemic_space.get(self.target_id)
        if not isinstance(source_component, Component) or not isinstance(
            target_component, Component
        ):
            raise TypeError("Los extremos del acoplamiento deben seguir siendo Components.")

        source_state = source_component.system_state
        target_state = target_component.system_state
        source_value = source_state.read_value(self.source_element_id)
        source_value = self.input_constraints.validate(
            source_value,
            "Valor de entrada del acoplamiento",
        )
        if not isinstance(source_value, float):
            raise TypeError("El acoplamiento requiere un valor fuente escalar.")

        transformed_value = self.transform(source_value)
        transformed_value = self.output_constraints.validate(
            transformed_value,
            "Resultado del acoplamiento",
        )
        if not isinstance(transformed_value, float):
            raise TypeError("El acoplamiento debe producir un valor escalar.")
        target_element = target_state.elements.get(self.target_element_id)
        if not isinstance(target_element, VariableElement):
            raise TypeError("El destino debe seguir siendo una VariableElement.")
        return float(
            target_element.constraints.validate(
                transformed_value,
                f"Variable '{target_element.name}'",
            )
        )

    def apply(self) -> float:
        value = self.evaluate()
        source_component = self.systemic_space.get(self.source_id)
        target_component = self.systemic_space.get(self.target_id)
        if not isinstance(source_component, Component) or not isinstance(
            target_component, Component
        ):
            raise TypeError("Los extremos del acoplamiento deben ser Components.")
        write_bridge = StateTransducer(
            consumer=source_component.system_state,
            provider=target_component.system_state,
            writable_elements={self.target_element_id},
        )
        write_bridge.write(self.target_element_id, value)
        return value

class StateCouplingStructure(SystemStructure):

    def __init__(self, systemic_space: SystemicSpace) -> None:
        super().__init__(systemic_space)
        self.identificator.entity = Ontology.EntityType.STATE_COUPLING_STRUCTURE

    def add_relation(self, relation_id: UUID) -> None:
        relation = self.systemic_space.get(relation_id)
        if not isinstance(relation, StateCouplingRelation):
            raise TypeError(
                "StateCouplingStructure solo admite StateCouplingRelation."
            )
        super().add_relation(relation_id)

    def apply_relation(self, relation_id: UUID) -> float:
        relation = self.relations.get(relation_id)
        if not isinstance(relation, StateCouplingRelation):
            raise KeyError("El acoplamiento no pertenece a esta estructura.")
        return relation.apply()

    def propagate(self) -> None:
        try:
            ordered_nodes = tuple(nx.topological_sort(self.topology))
        except nx.NetworkXUnfeasible as error:
            raise ValueError(
                "La propagación de acoplamientos requiere un grafo acíclico."
            ) from error

        for source_id in ordered_nodes:
            for target_id in self.topology.successors(source_id):
                for edge in self.topology[source_id][target_id].values():
                    self.apply_relation(edge["relation_id"])

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

# System Processes Control related Functions

    def run_process(self, process_id: UUID, trigger_id: UUID, token_id: Optional[UUID] = None, now: float = 0.0,) -> bool:
        process = self.processes.get(process_id)
        if process is None:
            raise KeyError("El Process no pertenece a este System.")
        return process.execute(trigger_id, token_id=token_id, now=now)

    def trigger_processes(self, kind: ProcessTriggerKind, processor_id: Optional[UUID] = None, 
    token_id: Optional[UUID] = None, now: float = 0.0,) -> int:
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
                if process.execute(trigger_id, token_id=token_id, now=now):
                    executed += 1
        return executed

    def run_condition_processes(self, now: float = 0.0) -> int:
        return self.trigger_processes(
            ProcessTriggerKind.CONDITION,
            now=now,
        )

    def run_simpy(self, until: float, environment: Optional[sim.Environment] = None,) -> sim.Environment:
        if not np.isfinite(until) or until <= 0:
            raise ValueError("until debe ser un tiempo finito mayor que cero.")
        if environment is None:
            environment = sim.Environment()
        if environment is not self._simpy_environment:
            self._simpy_environment = environment
            self._simpy_scheduled.clear()
        if until <= environment.now:
            raise ValueError("until debe ser mayor que el tiempo actual de SimPy.")

        scheduled: list[tuple[Process, UUID, float, float]] = []
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
                    scheduled.append(
                        (process, trigger_id, trigger.interval, initial_delay)
                    )

        def repeat(process: Process, trigger_id: UUID, interval: float, initial_delay: float,) -> Any:
            yield environment.timeout(initial_delay)
            while True:
                process.execute(trigger_id, now=environment.now)
                yield environment.timeout(interval)

        for process, trigger_id, interval, initial_delay in scheduled:
            environment.process(
                repeat(process, trigger_id, interval, initial_delay)
            )
            self._simpy_scheduled.add((process.identificator.id, trigger_id))
        environment.run(until=until)
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

    crank_angle = VariableElement(systemic_space, "crank_angle", 0.0, ValueConstraints(0, 720))
    crankshaft.system_state.add_element(crank_angle.identificator.id)
    cam_angle = VariableElement(systemic_space, "cam_angle", 0.0, ValueConstraints(0, 360))
    camshaft.system_state.add_element(cam_angle.identificator.id)

    cylinder_1_phase = VariableElement(systemic_space, "cylinder_1_phase", 0.0, ValueConstraints(0, 720))
    cylinder_1.system_state.add_element(cylinder_1_phase.identificator.id)
    piston_1_position = VariableElement(systemic_space, "piston_1_position", 0.0, ValueConstraints(0, 0.086))
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
    pressure_1 = VariableElement(systemic_space, "pressure_1", 1.0, ValueConstraints(1, 10))
    combustion_1.system_state.add_element(pressure_1.identificator.id)

    cylinder_2_phase = VariableElement(systemic_space, "cylinder_2_phase", 540.0, ValueConstraints(0, 720))
    cylinder_2.system_state.add_element(cylinder_2_phase.identificator.id)
    piston_2_position = VariableElement(systemic_space, "piston_2_position", 0.0, ValueConstraints(0, 0.086))
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
    pressure_2 = VariableElement(systemic_space, "pressure_2", 1.0, ValueConstraints(1, 10))
    combustion_2.system_state.add_element(pressure_2.identificator.id)

    cylinder_3_phase = VariableElement(systemic_space, "cylinder_3_phase", 180.0, ValueConstraints(0, 720))
    cylinder_3.system_state.add_element(cylinder_3_phase.identificator.id)
    piston_3_position = VariableElement(systemic_space, "piston_3_position", 0.0, ValueConstraints(0, 0.086))
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
    pressure_3 = VariableElement(systemic_space, "pressure_3", 1.0, ValueConstraints(1, 10))
    combustion_3.system_state.add_element(pressure_3.identificator.id)

    cylinder_4_phase = VariableElement(systemic_space, "cylinder_4_phase", 360.0, ValueConstraints(0, 720))
    cylinder_4.system_state.add_element(cylinder_4_phase.identificator.id)
    piston_4_position = VariableElement(systemic_space, "piston_4_position", 0.0, ValueConstraints(0, 0.086))
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
    pressure_4 = VariableElement(systemic_space, "pressure_4", 1.0, ValueConstraints(1, 10))
    combustion_4.system_state.add_element(pressure_4.identificator.id)

    coupling = StateCouplingStructure(systemic_space)
    coupling.add_component(crankshaft.identificator.id)
    coupling.add_component(camshaft.identificator.id)
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

    crank_to_cam = StateCouplingRelation(
        systemic_space, crankshaft.identificator.id, crank_angle.identificator.id,
        camshaft.identificator.id, cam_angle.identificator.id,
        lambda value: value / 2, label="crankshaft-to-camshaft",
    )
    crank_to_cylinder_1 = StateCouplingRelation(
        systemic_space, crankshaft.identificator.id, crank_angle.identificator.id,
        cylinder_1.identificator.id, cylinder_1_phase.identificator.id,
        lambda value: value % 720, label="crankshaft-to-cylinder-1",
    )
    crank_to_cylinder_2 = StateCouplingRelation(
        systemic_space, crankshaft.identificator.id, crank_angle.identificator.id,
        cylinder_2.identificator.id, cylinder_2_phase.identificator.id,
        lambda value: (value + 180) % 720, label="crankshaft-to-cylinder-2",
    )
    crank_to_cylinder_3 = StateCouplingRelation(
        systemic_space, crankshaft.identificator.id, crank_angle.identificator.id,
        cylinder_3.identificator.id, cylinder_3_phase.identificator.id,
        lambda value: (value + 540) % 720, label="crankshaft-to-cylinder-3",
    )
    crank_to_cylinder_4 = StateCouplingRelation(
        systemic_space, crankshaft.identificator.id, crank_angle.identificator.id,
        cylinder_4.identificator.id, cylinder_4_phase.identificator.id,
        lambda value: (value + 360) % 720, label="crankshaft-to-cylinder-4",
    )
    cylinder_1_to_piston = StateCouplingRelation(
        systemic_space, cylinder_1.identificator.id, cylinder_1_phase.identificator.id,
        piston_1.identificator.id, piston_1_position.identificator.id,
        lambda value: 0.043 * (1 - np.cos(np.deg2rad(value))),
    )
    cylinder_2_to_piston = StateCouplingRelation(
        systemic_space, cylinder_2.identificator.id, cylinder_2_phase.identificator.id,
        piston_2.identificator.id, piston_2_position.identificator.id,
        lambda value: 0.043 * (1 - np.cos(np.deg2rad(value))),
    )
    cylinder_3_to_piston = StateCouplingRelation(
        systemic_space, cylinder_3.identificator.id, cylinder_3_phase.identificator.id,
        piston_3.identificator.id, piston_3_position.identificator.id,
        lambda value: 0.043 * (1 - np.cos(np.deg2rad(value))),
    )
    cylinder_4_to_piston = StateCouplingRelation(
        systemic_space, cylinder_4.identificator.id, cylinder_4_phase.identificator.id,
        piston_4.identificator.id, piston_4_position.identificator.id,
        lambda value: 0.043 * (1 - np.cos(np.deg2rad(value))),
    )
    cylinder_1_to_spark = StateCouplingRelation(
        systemic_space, cylinder_1.identificator.id, cylinder_1_phase.identificator.id,
        spark_plug_1.identificator.id, spark_1.identificator.id,
        lambda value: float(value < 2 or value > 718),
    )
    cylinder_2_to_spark = StateCouplingRelation(
        systemic_space, cylinder_2.identificator.id, cylinder_2_phase.identificator.id,
        spark_plug_2.identificator.id, spark_2.identificator.id,
        lambda value: float(value < 2 or value > 718),
    )
    cylinder_3_to_spark = StateCouplingRelation(
        systemic_space, cylinder_3.identificator.id, cylinder_3_phase.identificator.id,
        spark_plug_3.identificator.id, spark_3.identificator.id,
        lambda value: float(value < 2 or value > 718),
    )
    cylinder_4_to_spark = StateCouplingRelation(
        systemic_space, cylinder_4.identificator.id, cylinder_4_phase.identificator.id,
        spark_plug_4.identificator.id, spark_4.identificator.id,
        lambda value: float(value < 2 or value > 718),
    )
    cylinder_1_to_pressure = StateCouplingRelation(
        systemic_space, cylinder_1.identificator.id, cylinder_1_phase.identificator.id,
        combustion_1.identificator.id, pressure_1.identificator.id,
        lambda value: 1 + 9 * np.exp(-0.5 * (min(value, 720 - value) / 35) ** 2),
    )
    cylinder_2_to_pressure = StateCouplingRelation(
        systemic_space, cylinder_2.identificator.id, cylinder_2_phase.identificator.id,
        combustion_2.identificator.id, pressure_2.identificator.id,
        lambda value: 1 + 9 * np.exp(-0.5 * (min(value, 720 - value) / 35) ** 2),
    )
    cylinder_3_to_pressure = StateCouplingRelation(
        systemic_space, cylinder_3.identificator.id, cylinder_3_phase.identificator.id,
        combustion_3.identificator.id, pressure_3.identificator.id,
        lambda value: 1 + 9 * np.exp(-0.5 * (min(value, 720 - value) / 35) ** 2),
    )
    cylinder_4_to_pressure = StateCouplingRelation(
        systemic_space, cylinder_4.identificator.id, cylinder_4_phase.identificator.id,
        combustion_4.identificator.id, pressure_4.identificator.id,
        lambda value: 1 + 9 * np.exp(-0.5 * (min(value, 720 - value) / 35) ** 2),
    )
    cam_to_pushrod_1_intake = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_1.identificator.id, pushrod_1_intake.identificator.id,
        lambda value: max(0, np.sin(np.deg2rad(value))),
    )
    cam_to_pushrod_1_exhaust = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_1.identificator.id, pushrod_1_exhaust.identificator.id,
        lambda value: max(0, -np.sin(np.deg2rad(value))),
    )
    cam_to_pushrod_2_intake = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_2.identificator.id, pushrod_2_intake.identificator.id,
        lambda value: max(0, np.sin(np.deg2rad(value + 90))),
    )
    cam_to_pushrod_2_exhaust = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_2.identificator.id, pushrod_2_exhaust.identificator.id,
        lambda value: max(0, -np.sin(np.deg2rad(value + 90))),
    )
    cam_to_pushrod_3_intake = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_3.identificator.id, pushrod_3_intake.identificator.id,
        lambda value: max(0, np.sin(np.deg2rad(value + 270))),
    )
    cam_to_pushrod_3_exhaust = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_3.identificator.id, pushrod_3_exhaust.identificator.id,
        lambda value: max(0, -np.sin(np.deg2rad(value + 270))),
    )
    cam_to_pushrod_4_intake = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_4.identificator.id, pushrod_4_intake.identificator.id,
        lambda value: max(0, np.sin(np.deg2rad(value + 180))),
    )
    cam_to_pushrod_4_exhaust = StateCouplingRelation(
        systemic_space, camshaft.identificator.id, cam_angle.identificator.id,
        pushrod_4.identificator.id, pushrod_4_exhaust.identificator.id,
        lambda value: max(0, -np.sin(np.deg2rad(value + 180))),
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

    coupling.add_relation(crank_to_cam.identificator.id)
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
        "cycle_angle_deg": 720,
    }

    # define_archetype_from_instance registra el System actual como instancia
    # y deriva su esquema desde semantic_attributes; add_attribute fija tipos.
    engine.semantics.add_tag(engine.identificator.id, "engine")
    engine.semantics.add_tag(engine.identificator.id, "four-stroke")
    engine.semantics.add_tag(engine.identificator.id, "OHV")
    engine_archetype = engine.semantics.define_archetype_from_instance(
        "L4 OHV engine",
        engine.identificator.id,
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
        SemanticAttribute("cycle_angle_deg", SemanticValueType.INTEGER, 720, required=True),
    )

    # add_tag clasifica las entidades por función; los SemanticSet siguientes
    # son agrupaciones explícitas y su partición cubre los 30 Components.
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
    engine.semantics.add_member(ignition_parts.id, spark_plug_1.identificator.id)
    engine.semantics.add_member(ignition_parts.id, spark_plug_2.identificator.id)
    engine.semantics.add_member(ignition_parts.id, spark_plug_3.identificator.id)
    engine.semantics.add_member(ignition_parts.id, spark_plug_4.identificator.id)
    engine.semantics.add_member(combustion_parts.id, combustion_1.identificator.id)
    engine.semantics.add_member(combustion_parts.id, combustion_2.identificator.id)
    engine.semantics.add_member(combustion_parts.id, combustion_3.identificator.id)
    engine.semantics.add_member(combustion_parts.id, combustion_4.identificator.id)
    engine.semantics.define_partition(
        engine_parts.id,
        (
            rotating_parts.id,
            cylinder_parts.id,
            reciprocating_parts.id,
            valvetrain_parts.id,
            ignition_parts.id,
            combustion_parts.id,
        ),
    )

    # add_part declara la mereología System→Component y Cylinder→subcomponent;
    # is_part_of conserva la transitividad sin duplicar conexiones indirectas.
    engine.semantics.add_part(engine.identificator.id, crankshaft.identificator.id)
    engine.semantics.add_part(engine.identificator.id, camshaft.identificator.id)
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
    meter_unit = engine.semantics.define_unit("meter", "m", {"length": 1})
    degree_unit = engine.semantics.define_unit("degree", "deg", {"angle": 1})
    bar_unit = engine.semantics.define_unit("bar", "bar", {"pressure": 1})
    fraction_unit = engine.semantics.define_unit("fraction", "1", {})
    engine.semantics.bind_state_variable(crankshaft.system_state, crank_angle.identificator.id, degree_unit.id)
    engine.semantics.bind_state_variable(camshaft.system_state, cam_angle.identificator.id, degree_unit.id)
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

    simulation_time = ProcessValueRef(
        systemic_space, ProcessValueScope.SIMULATION_TIME,
    )
    time_value = ProcessExpression(
        systemic_space, ProcessExpressionOperator.READ,
        value_ref_id=simulation_time.identificator.id,
    )
    crank_velocity = ProcessExpression(
        systemic_space, ProcessExpressionOperator.CONSTANT, value=7200,
    )
    current_crank_angle = ProcessExpression(
        systemic_space, ProcessExpressionOperator.MULTIPLY,
        operands=(time_value.identificator.id, crank_velocity.identificator.id),
    )
    crank_interval = ProcessTrigger(
        systemic_space, ProcessTriggerKind.INTERVAL,
        interval=0.025, initial_delay=0,
    )
    engine_process = Process(
        systemic_space, engine.identificator.id, label="L4 OHV cycle",
    )
    set_crank_angle = ProcessAction(
        systemic_space, ProcessActionKind.SET_COMPONENT_VALUE,
        expression_id=current_crank_angle.identificator.id,
        element_id=crank_angle.identificator.id,
        component_id=crankshaft.identificator.id,
    )
    propagate_couplings = ProcessAction(
        systemic_space, ProcessActionKind.PROPAGATE_STATE_COUPLINGS,
        structure_id=coupling.identificator.id,
    )
    engine_process.add_trigger(crank_interval.identificator.id)
    engine_process.add_action(set_crank_angle.identificator.id)
    engine_process.add_action(propagate_couplings.identificator.id)
    engine.suscribe_systemic_description(engine_process.identificator.id)

    simulation = sim.Environment()
    simulation = engine.run_simpy(until=0.0001, environment=simulation)
    print("advance 1/5")
    print("crank_angle", crankshaft.system_state.read_value(crank_angle.identificator.id))
    print("cam_angle", camshaft.system_state.read_value(cam_angle.identificator.id))
    print("cylinder_1_phase", cylinder_1.system_state.read_value(cylinder_1_phase.identificator.id))
    print("piston_1_position", piston_1.system_state.read_value(piston_1_position.identificator.id))
    print("pushrod_1_intake", pushrod_1.system_state.read_value(pushrod_1_intake.identificator.id))
    print("pushrod_1_exhaust", pushrod_1.system_state.read_value(pushrod_1_exhaust.identificator.id))
    print("rocker_1_intake", rocker_1.system_state.read_value(rocker_1_intake.identificator.id))
    print("rocker_1_exhaust", rocker_1.system_state.read_value(rocker_1_exhaust.identificator.id))
    print("valve_1_intake", valves_1.system_state.read_value(valve_1_intake.identificator.id))
    print("valve_1_exhaust", valves_1.system_state.read_value(valve_1_exhaust.identificator.id))
    print("spark_1", spark_plug_1.system_state.read_value(spark_1.identificator.id))
    print("pressure_1", combustion_1.system_state.read_value(pressure_1.identificator.id))
    print("cylinder_2_phase", cylinder_2.system_state.read_value(cylinder_2_phase.identificator.id))
    print("piston_2_position", piston_2.system_state.read_value(piston_2_position.identificator.id))
    print("pushrod_2_intake", pushrod_2.system_state.read_value(pushrod_2_intake.identificator.id))
    print("pushrod_2_exhaust", pushrod_2.system_state.read_value(pushrod_2_exhaust.identificator.id))
    print("rocker_2_intake", rocker_2.system_state.read_value(rocker_2_intake.identificator.id))
    print("rocker_2_exhaust", rocker_2.system_state.read_value(rocker_2_exhaust.identificator.id))
    print("valve_2_intake", valves_2.system_state.read_value(valve_2_intake.identificator.id))
    print("valve_2_exhaust", valves_2.system_state.read_value(valve_2_exhaust.identificator.id))
    print("spark_2", spark_plug_2.system_state.read_value(spark_2.identificator.id))
    print("pressure_2", combustion_2.system_state.read_value(pressure_2.identificator.id))
    print("cylinder_3_phase", cylinder_3.system_state.read_value(cylinder_3_phase.identificator.id))
    print("piston_3_position", piston_3.system_state.read_value(piston_3_position.identificator.id))
    print("pushrod_3_intake", pushrod_3.system_state.read_value(pushrod_3_intake.identificator.id))
    print("pushrod_3_exhaust", pushrod_3.system_state.read_value(pushrod_3_exhaust.identificator.id))
    print("rocker_3_intake", rocker_3.system_state.read_value(rocker_3_intake.identificator.id))
    print("rocker_3_exhaust", rocker_3.system_state.read_value(rocker_3_exhaust.identificator.id))
    print("valve_3_intake", valves_3.system_state.read_value(valve_3_intake.identificator.id))
    print("valve_3_exhaust", valves_3.system_state.read_value(valve_3_exhaust.identificator.id))
    print("spark_3", spark_plug_3.system_state.read_value(spark_3.identificator.id))
    print("pressure_3", combustion_3.system_state.read_value(pressure_3.identificator.id))
    print("cylinder_4_phase", cylinder_4.system_state.read_value(cylinder_4_phase.identificator.id))
    print("piston_4_position", piston_4.system_state.read_value(piston_4_position.identificator.id))
    print("pushrod_4_intake", pushrod_4.system_state.read_value(pushrod_4_intake.identificator.id))
    print("pushrod_4_exhaust", pushrod_4.system_state.read_value(pushrod_4_exhaust.identificator.id))
    print("rocker_4_intake", rocker_4.system_state.read_value(rocker_4_intake.identificator.id))
    print("rocker_4_exhaust", rocker_4.system_state.read_value(rocker_4_exhaust.identificator.id))
    print("valve_4_intake", valves_4.system_state.read_value(valve_4_intake.identificator.id))
    print("valve_4_exhaust", valves_4.system_state.read_value(valve_4_exhaust.identificator.id))
    print("spark_4", spark_plug_4.system_state.read_value(spark_4.identificator.id))
    print("pressure_4", combustion_4.system_state.read_value(pressure_4.identificator.id))
    simulation = engine.run_simpy(until=0.0251, environment=simulation)
    print("advance 2/5")
    print("crank_angle", crankshaft.system_state.read_value(crank_angle.identificator.id))
    print("cam_angle", camshaft.system_state.read_value(cam_angle.identificator.id))
    print("cylinder_1_phase", cylinder_1.system_state.read_value(cylinder_1_phase.identificator.id))
    print("piston_1_position", piston_1.system_state.read_value(piston_1_position.identificator.id))
    print("pushrod_1_intake", pushrod_1.system_state.read_value(pushrod_1_intake.identificator.id))
    print("pushrod_1_exhaust", pushrod_1.system_state.read_value(pushrod_1_exhaust.identificator.id))
    print("rocker_1_intake", rocker_1.system_state.read_value(rocker_1_intake.identificator.id))
    print("rocker_1_exhaust", rocker_1.system_state.read_value(rocker_1_exhaust.identificator.id))
    print("valve_1_intake", valves_1.system_state.read_value(valve_1_intake.identificator.id))
    print("valve_1_exhaust", valves_1.system_state.read_value(valve_1_exhaust.identificator.id))
    print("spark_1", spark_plug_1.system_state.read_value(spark_1.identificator.id))
    print("pressure_1", combustion_1.system_state.read_value(pressure_1.identificator.id))
    print("cylinder_2_phase", cylinder_2.system_state.read_value(cylinder_2_phase.identificator.id))
    print("piston_2_position", piston_2.system_state.read_value(piston_2_position.identificator.id))
    print("pushrod_2_intake", pushrod_2.system_state.read_value(pushrod_2_intake.identificator.id))
    print("pushrod_2_exhaust", pushrod_2.system_state.read_value(pushrod_2_exhaust.identificator.id))
    print("rocker_2_intake", rocker_2.system_state.read_value(rocker_2_intake.identificator.id))
    print("rocker_2_exhaust", rocker_2.system_state.read_value(rocker_2_exhaust.identificator.id))
    print("valve_2_intake", valves_2.system_state.read_value(valve_2_intake.identificator.id))
    print("valve_2_exhaust", valves_2.system_state.read_value(valve_2_exhaust.identificator.id))
    print("spark_2", spark_plug_2.system_state.read_value(spark_2.identificator.id))
    print("pressure_2", combustion_2.system_state.read_value(pressure_2.identificator.id))
    print("cylinder_3_phase", cylinder_3.system_state.read_value(cylinder_3_phase.identificator.id))
    print("piston_3_position", piston_3.system_state.read_value(piston_3_position.identificator.id))
    print("pushrod_3_intake", pushrod_3.system_state.read_value(pushrod_3_intake.identificator.id))
    print("pushrod_3_exhaust", pushrod_3.system_state.read_value(pushrod_3_exhaust.identificator.id))
    print("rocker_3_intake", rocker_3.system_state.read_value(rocker_3_intake.identificator.id))
    print("rocker_3_exhaust", rocker_3.system_state.read_value(rocker_3_exhaust.identificator.id))
    print("valve_3_intake", valves_3.system_state.read_value(valve_3_intake.identificator.id))
    print("valve_3_exhaust", valves_3.system_state.read_value(valve_3_exhaust.identificator.id))
    print("spark_3", spark_plug_3.system_state.read_value(spark_3.identificator.id))
    print("pressure_3", combustion_3.system_state.read_value(pressure_3.identificator.id))
    print("cylinder_4_phase", cylinder_4.system_state.read_value(cylinder_4_phase.identificator.id))
    print("piston_4_position", piston_4.system_state.read_value(piston_4_position.identificator.id))
    print("pushrod_4_intake", pushrod_4.system_state.read_value(pushrod_4_intake.identificator.id))
    print("pushrod_4_exhaust", pushrod_4.system_state.read_value(pushrod_4_exhaust.identificator.id))
    print("rocker_4_intake", rocker_4.system_state.read_value(rocker_4_intake.identificator.id))
    print("rocker_4_exhaust", rocker_4.system_state.read_value(rocker_4_exhaust.identificator.id))
    print("valve_4_intake", valves_4.system_state.read_value(valve_4_intake.identificator.id))
    print("valve_4_exhaust", valves_4.system_state.read_value(valve_4_exhaust.identificator.id))
    print("spark_4", spark_plug_4.system_state.read_value(spark_4.identificator.id))
    print("pressure_4", combustion_4.system_state.read_value(pressure_4.identificator.id))
    simulation = engine.run_simpy(until=0.0501, environment=simulation)
    print("advance 3/5")
    print("crank_angle", crankshaft.system_state.read_value(crank_angle.identificator.id))
    print("cam_angle", camshaft.system_state.read_value(cam_angle.identificator.id))
    print("cylinder_1_phase", cylinder_1.system_state.read_value(cylinder_1_phase.identificator.id))
    print("piston_1_position", piston_1.system_state.read_value(piston_1_position.identificator.id))
    print("pushrod_1_intake", pushrod_1.system_state.read_value(pushrod_1_intake.identificator.id))
    print("pushrod_1_exhaust", pushrod_1.system_state.read_value(pushrod_1_exhaust.identificator.id))
    print("rocker_1_intake", rocker_1.system_state.read_value(rocker_1_intake.identificator.id))
    print("rocker_1_exhaust", rocker_1.system_state.read_value(rocker_1_exhaust.identificator.id))
    print("valve_1_intake", valves_1.system_state.read_value(valve_1_intake.identificator.id))
    print("valve_1_exhaust", valves_1.system_state.read_value(valve_1_exhaust.identificator.id))
    print("spark_1", spark_plug_1.system_state.read_value(spark_1.identificator.id))
    print("pressure_1", combustion_1.system_state.read_value(pressure_1.identificator.id))
    print("cylinder_2_phase", cylinder_2.system_state.read_value(cylinder_2_phase.identificator.id))
    print("piston_2_position", piston_2.system_state.read_value(piston_2_position.identificator.id))
    print("pushrod_2_intake", pushrod_2.system_state.read_value(pushrod_2_intake.identificator.id))
    print("pushrod_2_exhaust", pushrod_2.system_state.read_value(pushrod_2_exhaust.identificator.id))
    print("rocker_2_intake", rocker_2.system_state.read_value(rocker_2_intake.identificator.id))
    print("rocker_2_exhaust", rocker_2.system_state.read_value(rocker_2_exhaust.identificator.id))
    print("valve_2_intake", valves_2.system_state.read_value(valve_2_intake.identificator.id))
    print("valve_2_exhaust", valves_2.system_state.read_value(valve_2_exhaust.identificator.id))
    print("spark_2", spark_plug_2.system_state.read_value(spark_2.identificator.id))
    print("pressure_2", combustion_2.system_state.read_value(pressure_2.identificator.id))
    print("cylinder_3_phase", cylinder_3.system_state.read_value(cylinder_3_phase.identificator.id))
    print("piston_3_position", piston_3.system_state.read_value(piston_3_position.identificator.id))
    print("pushrod_3_intake", pushrod_3.system_state.read_value(pushrod_3_intake.identificator.id))
    print("pushrod_3_exhaust", pushrod_3.system_state.read_value(pushrod_3_exhaust.identificator.id))
    print("rocker_3_intake", rocker_3.system_state.read_value(rocker_3_intake.identificator.id))
    print("rocker_3_exhaust", rocker_3.system_state.read_value(rocker_3_exhaust.identificator.id))
    print("valve_3_intake", valves_3.system_state.read_value(valve_3_intake.identificator.id))
    print("valve_3_exhaust", valves_3.system_state.read_value(valve_3_exhaust.identificator.id))
    print("spark_3", spark_plug_3.system_state.read_value(spark_3.identificator.id))
    print("pressure_3", combustion_3.system_state.read_value(pressure_3.identificator.id))
    print("cylinder_4_phase", cylinder_4.system_state.read_value(cylinder_4_phase.identificator.id))
    print("piston_4_position", piston_4.system_state.read_value(piston_4_position.identificator.id))
    print("pushrod_4_intake", pushrod_4.system_state.read_value(pushrod_4_intake.identificator.id))
    print("pushrod_4_exhaust", pushrod_4.system_state.read_value(pushrod_4_exhaust.identificator.id))
    print("rocker_4_intake", rocker_4.system_state.read_value(rocker_4_intake.identificator.id))
    print("rocker_4_exhaust", rocker_4.system_state.read_value(rocker_4_exhaust.identificator.id))
    print("valve_4_intake", valves_4.system_state.read_value(valve_4_intake.identificator.id))
    print("valve_4_exhaust", valves_4.system_state.read_value(valve_4_exhaust.identificator.id))
    print("spark_4", spark_plug_4.system_state.read_value(spark_4.identificator.id))
    print("pressure_4", combustion_4.system_state.read_value(pressure_4.identificator.id))
    simulation = engine.run_simpy(until=0.0751, environment=simulation)
    print("advance 4/5")
    print("crank_angle", crankshaft.system_state.read_value(crank_angle.identificator.id))
    print("cam_angle", camshaft.system_state.read_value(cam_angle.identificator.id))
    print("cylinder_1_phase", cylinder_1.system_state.read_value(cylinder_1_phase.identificator.id))
    print("piston_1_position", piston_1.system_state.read_value(piston_1_position.identificator.id))
    print("pushrod_1_intake", pushrod_1.system_state.read_value(pushrod_1_intake.identificator.id))
    print("pushrod_1_exhaust", pushrod_1.system_state.read_value(pushrod_1_exhaust.identificator.id))
    print("rocker_1_intake", rocker_1.system_state.read_value(rocker_1_intake.identificator.id))
    print("rocker_1_exhaust", rocker_1.system_state.read_value(rocker_1_exhaust.identificator.id))
    print("valve_1_intake", valves_1.system_state.read_value(valve_1_intake.identificator.id))
    print("valve_1_exhaust", valves_1.system_state.read_value(valve_1_exhaust.identificator.id))
    print("spark_1", spark_plug_1.system_state.read_value(spark_1.identificator.id))
    print("pressure_1", combustion_1.system_state.read_value(pressure_1.identificator.id))
    print("cylinder_2_phase", cylinder_2.system_state.read_value(cylinder_2_phase.identificator.id))
    print("piston_2_position", piston_2.system_state.read_value(piston_2_position.identificator.id))
    print("pushrod_2_intake", pushrod_2.system_state.read_value(pushrod_2_intake.identificator.id))
    print("pushrod_2_exhaust", pushrod_2.system_state.read_value(pushrod_2_exhaust.identificator.id))
    print("rocker_2_intake", rocker_2.system_state.read_value(rocker_2_intake.identificator.id))
    print("rocker_2_exhaust", rocker_2.system_state.read_value(rocker_2_exhaust.identificator.id))
    print("valve_2_intake", valves_2.system_state.read_value(valve_2_intake.identificator.id))
    print("valve_2_exhaust", valves_2.system_state.read_value(valve_2_exhaust.identificator.id))
    print("spark_2", spark_plug_2.system_state.read_value(spark_2.identificator.id))
    print("pressure_2", combustion_2.system_state.read_value(pressure_2.identificator.id))
    print("cylinder_3_phase", cylinder_3.system_state.read_value(cylinder_3_phase.identificator.id))
    print("piston_3_position", piston_3.system_state.read_value(piston_3_position.identificator.id))
    print("pushrod_3_intake", pushrod_3.system_state.read_value(pushrod_3_intake.identificator.id))
    print("pushrod_3_exhaust", pushrod_3.system_state.read_value(pushrod_3_exhaust.identificator.id))
    print("rocker_3_intake", rocker_3.system_state.read_value(rocker_3_intake.identificator.id))
    print("rocker_3_exhaust", rocker_3.system_state.read_value(rocker_3_exhaust.identificator.id))
    print("valve_3_intake", valves_3.system_state.read_value(valve_3_intake.identificator.id))
    print("valve_3_exhaust", valves_3.system_state.read_value(valve_3_exhaust.identificator.id))
    print("spark_3", spark_plug_3.system_state.read_value(spark_3.identificator.id))
    print("pressure_3", combustion_3.system_state.read_value(pressure_3.identificator.id))
    print("cylinder_4_phase", cylinder_4.system_state.read_value(cylinder_4_phase.identificator.id))
    print("piston_4_position", piston_4.system_state.read_value(piston_4_position.identificator.id))
    print("pushrod_4_intake", pushrod_4.system_state.read_value(pushrod_4_intake.identificator.id))
    print("pushrod_4_exhaust", pushrod_4.system_state.read_value(pushrod_4_exhaust.identificator.id))
    print("rocker_4_intake", rocker_4.system_state.read_value(rocker_4_intake.identificator.id))
    print("rocker_4_exhaust", rocker_4.system_state.read_value(rocker_4_exhaust.identificator.id))
    print("valve_4_intake", valves_4.system_state.read_value(valve_4_intake.identificator.id))
    print("valve_4_exhaust", valves_4.system_state.read_value(valve_4_exhaust.identificator.id))
    print("spark_4", spark_plug_4.system_state.read_value(spark_4.identificator.id))
    print("pressure_4", combustion_4.system_state.read_value(pressure_4.identificator.id))
    simulation = engine.run_simpy(until=0.1001, environment=simulation)
    print("advance 5/5")
    print("crank_angle", crankshaft.system_state.read_value(crank_angle.identificator.id))
    print("cam_angle", camshaft.system_state.read_value(cam_angle.identificator.id))
    print("cylinder_1_phase", cylinder_1.system_state.read_value(cylinder_1_phase.identificator.id))
    print("piston_1_position", piston_1.system_state.read_value(piston_1_position.identificator.id))
    print("pushrod_1_intake", pushrod_1.system_state.read_value(pushrod_1_intake.identificator.id))
    print("pushrod_1_exhaust", pushrod_1.system_state.read_value(pushrod_1_exhaust.identificator.id))
    print("rocker_1_intake", rocker_1.system_state.read_value(rocker_1_intake.identificator.id))
    print("rocker_1_exhaust", rocker_1.system_state.read_value(rocker_1_exhaust.identificator.id))
    print("valve_1_intake", valves_1.system_state.read_value(valve_1_intake.identificator.id))
    print("valve_1_exhaust", valves_1.system_state.read_value(valve_1_exhaust.identificator.id))
    print("spark_1", spark_plug_1.system_state.read_value(spark_1.identificator.id))
    print("pressure_1", combustion_1.system_state.read_value(pressure_1.identificator.id))
    print("cylinder_2_phase", cylinder_2.system_state.read_value(cylinder_2_phase.identificator.id))
    print("piston_2_position", piston_2.system_state.read_value(piston_2_position.identificator.id))
    print("pushrod_2_intake", pushrod_2.system_state.read_value(pushrod_2_intake.identificator.id))
    print("pushrod_2_exhaust", pushrod_2.system_state.read_value(pushrod_2_exhaust.identificator.id))
    print("rocker_2_intake", rocker_2.system_state.read_value(rocker_2_intake.identificator.id))
    print("rocker_2_exhaust", rocker_2.system_state.read_value(rocker_2_exhaust.identificator.id))
    print("valve_2_intake", valves_2.system_state.read_value(valve_2_intake.identificator.id))
    print("valve_2_exhaust", valves_2.system_state.read_value(valve_2_exhaust.identificator.id))
    print("spark_2", spark_plug_2.system_state.read_value(spark_2.identificator.id))
    print("pressure_2", combustion_2.system_state.read_value(pressure_2.identificator.id))
    print("cylinder_3_phase", cylinder_3.system_state.read_value(cylinder_3_phase.identificator.id))
    print("piston_3_position", piston_3.system_state.read_value(piston_3_position.identificator.id))
    print("pushrod_3_intake", pushrod_3.system_state.read_value(pushrod_3_intake.identificator.id))
    print("pushrod_3_exhaust", pushrod_3.system_state.read_value(pushrod_3_exhaust.identificator.id))
    print("rocker_3_intake", rocker_3.system_state.read_value(rocker_3_intake.identificator.id))
    print("rocker_3_exhaust", rocker_3.system_state.read_value(rocker_3_exhaust.identificator.id))
    print("valve_3_intake", valves_3.system_state.read_value(valve_3_intake.identificator.id))
    print("valve_3_exhaust", valves_3.system_state.read_value(valve_3_exhaust.identificator.id))
    print("spark_3", spark_plug_3.system_state.read_value(spark_3.identificator.id))
    print("pressure_3", combustion_3.system_state.read_value(pressure_3.identificator.id))
    print("cylinder_4_phase", cylinder_4.system_state.read_value(cylinder_4_phase.identificator.id))
    print("piston_4_position", piston_4.system_state.read_value(piston_4_position.identificator.id))
    print("pushrod_4_intake", pushrod_4.system_state.read_value(pushrod_4_intake.identificator.id))
    print("pushrod_4_exhaust", pushrod_4.system_state.read_value(pushrod_4_exhaust.identificator.id))
    print("rocker_4_intake", rocker_4.system_state.read_value(rocker_4_intake.identificator.id))
    print("rocker_4_exhaust", rocker_4.system_state.read_value(rocker_4_exhaust.identificator.id))
    print("valve_4_intake", valves_4.system_state.read_value(valve_4_intake.identificator.id))
    print("valve_4_exhaust", valves_4.system_state.read_value(valve_4_exhaust.identificator.id))
    print("spark_4", spark_plug_4.system_state.read_value(spark_4.identificator.id))
    print("pressure_4", combustion_4.system_state.read_value(pressure_4.identificator.id))
    
    plt.figure(figsize = (6,4))
    nx.draw(G = coupling.topology, 
    node_color = "black",  
    with_labels=False,
    node_size=800,
    font_size=12,
    font_weight='bold',
    edge_color='gray')
    plt.show()
