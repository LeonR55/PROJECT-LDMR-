from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum, auto
from uuid import uuid4, UUID

# IMPORTS DE LIBRERIAS DE CIENCIA DE DATOS Y SIMULACION

import networkx as nx
import numpy, scipy, simpy, sympy


# SYPHEN (SYSTEMS PHYSICS ENGINE)

@dataclass
class Ontology:

    class EntityType(Enum):
        NOT_DEFINED = auto()
        VARIABLE_TYPE = auto()
        STATE_VARIABLE = auto()
        STATE = auto()
        SYSTEM_EDGE = auto()
        SYSTEM_TOPOLOGY = auto()
        COMPONENT = auto()
        TOKEN = auto()
        SYSTEM = auto()
        SPACE = auto()
    
@dataclass
class Identification:

    id : UUID = field(init = False, default_factory = uuid4)
    label : Optional[str] = None
    entity : Optional[Ontology.EntityType] = None

    def labeler(self):
        pass

@dataclass
class Source:

    variable_type : dict[UUID, VariableType] = field(default_factory = dict)
    state_variable : dict[UUID, VariableDescriptor] = field(default_factory = dict)
    state : dict[UUID, State] = field(default_factory = dict)
    system_edge : dict[UUID, SystemEdge] = field(default_factory = dict)
    system_topology : dict[UUID, SystemTopology] = field(default_factory = dict)
    component : dict[UUID, Component] = field(default_factory = dict)
    token : dict[UUID, Token] = field(default_factory = dict)
    system : dict[UUID, System] = field(default_factory = dict)

    def funnel(self, entity: Entity) -> dict[UUID, Any]:

        destinations: dict[type, dict[UUID, Any]] = {

            VariableDescriptor: self.variable_type,
            StateVariable: self.state_variable,
            State: self.state,
            TopologyEdge: self.system_edge,
            SystemTopology: self.system_topology,
            Component: self.component,
            Token: self.token,
            System: self.system,
        }
    
        entity_type = type(entity)
    
        try:
            return destinations[entity_type]
        except KeyError:
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

    def suscribe_system_components(self, system : UUID) -> None:
        source_system : System = self.systemic_space.get(system)
        source_system.system_topology



class SpaceMainTopology:

    def __init__(self, systemic_space : SystemicSpace) -> None:
        self.graph : nx.MultiDiGraph = nx.MultiDiGraph()
        self.systemic_space : SystemicSpace = systemic_space


class ValidationEngine:

    def __init__(self):
        pass

    # Validación de Existencia en Registros 
    def validate_existence(self, entity : UUID, source : Source) -> bool:
        for entity_dict in vars(source).values():
            if entity in entity_dict:
                return True
        return False

    # Validación de Requerimentos de Tipos de Entidad 
    def validate_entity_entity(self, entity : Optional[Entity], expected_entity_type : Any) -> bool:
        return isinstance(entity, expected_entity_type)

    # Validación de Unico
    def validate_uniqueness(self, entity : UUID, entity_Type : Ontology.EntityType) -> bool:
        pass


class SystemGateKeeper:
    
    def __init__(self, systemic_space : SystemicSpace):
        self.space = systemic_space.source

    def intercept_and_authorize(self, entity : UUID):
        pass 

class SystemicSpace:

    identificator : Identification

    validator : ValidationEngine

    source : Source
    space_main_topology : SpaceMainTopology
    flat_space : FlatSpaceTopology 

    def __init__(self):

        self.identificator = Identification(entity = Ontology.EntityType.SPACE)
        self.validator = ValidationEngine()

        self.source = Source()

        self.space_main_topology = SpaceMainTopology(systemic_space = self)
        self.flat_space = FlatSpaceTopology(systemic_space = self)

    # CRUD SOURCE MANAGING
    def create(self, ID : UUID , entity : Entity) -> None:
        destination = self.source.funnel(entity)
        destination[ID] = entity

    def get(self, ID : UUID) -> Optional[Entity]:
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
        for entity_dict in vars(self.source).values():


    # 
    def suscribe_component(self, entity_id: UUID) -> None:        
        work_entity = self.get(entity_id)
        if not self.validator.validate_entity_entity(work_entity, Component):
            raise TypeError(f"Entity with ID {entity_id} is not a Component")
        self.graph_space.add_node(entity_id)

    def unsuscribe_component(self, entity_id: UUID) -> None:
        self.validator.validate_existence(entity_id, self.source)
        self.graph_space.remove_node(entity_id)

    def create_relation(self, source_id: UUID, target_id: UUID, edge_type: Ontology.EntityType) -> None:
        components = [self.get(source_id), self.get(target_id)]
        for component in components:
            if not self.validator.validate_entity_entity(component, Component or System):
                raise TypeError(f"Entity with ID is not a Component.")
        self.graph_space.add_edge(source_id, target_id, type=edge_type)
        
    def delete_relation(self, source_id: UUID, target_id: UUID) -> None:
        self.validator.validate_existence(source_id, self.source)
        self.validator.validate_existence(target_id, self.source)
        if not self.graph_space.has_edge(source_id, target_id):
            raise KeyError(f"The Relation Described does not exist{source_id} y {target_id}.")
        self.graph_space.remove_edge(source_id, target_id)

    # SYSTEMS GRAPH MANAGEMENT

    def suscribe_system_topology(self):
        for system_topology in self.source.system_topology.values():
            self.validator.validate_entity_entity(system_topology, SystemTopology)
            if system_topology.identificator == False:
                raise ValueError ("The ")
        

    def unsuscribe_system_topology(self):
        pass

class TemporalEngine:
     
    space : SystemicSpace

    def __init__(self, space : SystemicSpace):
        self.space = space
        self.env = simpy.Environment()

    def run_time (self, until : float) -> None:
        self.env.run(until = until)

    # ========= #
    # ESTADO 
    # ========= #
    class StateRule(ABC):

        @property
        @abstractmethod
        def allowed_types(self) -> set[type]:
            pass
            
        @abstractmethod
        def validate_value(self, value_to_validate) -> bool:
            pass

    class DecimalStateRule(StateRule):

        @property
        def allowed_types (self) -> set[type] :
            return {float}
    
        def __init__(self, max_value : float  ,  min_value : float) -> None:
            self.max_value : Optional[float] = max_value
            self.min_value : Optional[float] = min_value    

        def validate_value(self, value_to_validate : Any) -> bool:
            if type(value_to_validate) not in self.allowed_types:
                return False
            elif self.min_value is not None and value_to_validate < self.min_value:
                return False
            elif self.max_value is not None and value_to_validate > self.max_value:
                return False
            else:
                return True

    class IntegerValidationRule(StateRule):

        @property
        def allowed_types (self) -> set[type] :
            return {int}

        def __init__(self, max_value : int, min_value : int) -> None:
            self.max_value : Optional[int] = max_value
            self.min_value : Optional[int] = min_value    

        def validate_value(self, value_to_validate : Any) -> bool:
            if type(value_to_validate) not in self.allowed_types:
                return False
            elif self.min_value is not None and value_to_validate < self.min_value:
                return False
            elif self.max_value is not None and value_to_validate > self.max_value:
                return False
            else:
                return True

    class BinaryStateRule(StateRule):

        @property
        def allowed_types (self) -> set[type] :
            return {bool}
        
        def validate_value(self, value_to_validate : Any) -> bool:
            return type(value_to_validate) is bool

    class DescriptiveStateRule(StateRule):
        
        @property
        def allowed_types (self) -> set[type] :
            return {str}

        def __init__(self, possible_values : list[str]):
            self.possible_values : list[str] = possible_values

        def validate_value(self, value_to_validate : Any) -> bool:
                if type(value_to_validate) not in self.allowed_types:
                    return False
                elif value_to_validate not in self.possible_values:
                    return False
                else:
                    return True

    class CompoundValidationRule:
        pass     

    

    class Entity(ABC):
        pass        

    @dataclass
    class VariableDescriptor(Entity):

        identicator : Identification = field(init = False, default_factory = lambda : Identification(entity = Ontology.EntityType.VARIABLE_TYPE))
        name: Optional[str] = None
        validation_rule: StateRule = field(default_factory = lambda : DecimalStateRule(max_value=1000, min_value=0))

    @dataclass 
    class StateVariable(Entity):

        identificator : Identification = field(init = False, default_factory = lambda : Identification(entity=Ontology.EntityType.STATE_VARIABLE))
        name : Optional[str] = None
        variable_descriptor : VariableDescriptor = field(default_factory = lambda : VariableDescriptor())
        current_value : Any = None

        def update_value(self, new_value: Any) -> None:
            if self.variable_descriptor.validation_rule.validate_value(new_value) == False:
                raise ValueError("The Current Value is not a valid value for the variable created")
            else:
                self.current_value = new_value 

    @dataclass 
    class State(Entity):

        identificator : Identification = field(default_factory = lambda : Identification(entity=Ontology.EntityType.STATE))
        variables : dict[UUID, StateVariable] = field(default_factory = dict)
        values: dict[UUID, Any] = field(init = False, default_factory = dict)

        # Metodos de Acceso y Modificacion de las Variables de Estado
        @property
        def readable_values(self) -> dict[str, Any]:
            return {(self.variables[uuid].name or f"Unknown_{uuid}"): value for uuid, value in self.values.items()}


    # ======================== #
    # TOPOLOGIA INTERACCIONAL
    # ======================== #
    @dataclass
    class TopologyEdge(Entity):
        
        identificator : Identification = field(init = False, default_factory = lambda : Identification(entity=Ontology.EntityType.SYSTEM_EDGE))
        origin_node : Optional[UUID] = None
        destination_node : Optional[UUID] = None

        
    class BusEdge(TopologyEdge):

        token_queue : dict[UUID, Token] = field(default_factory = dict)

        def enqueue_token(self, token : UUID) -> None:
            pass

    class StateRelationVariableEdge(TopologyEdge):

        pass

        



    @dataclass
    class SystemTopology(Entity):

        identificator : Identification = field(default_factory = lambda : Identification(label = "System Topology", entity = Ontology.EntityType.SYSTEM_TOPOLOGY))
        


    @dataclass
    class FunctionalStateTopology(SystemTopology):
        pass

    @dataclass
    class BusTopology(SystemTopology):
        pass

    class MereologicalTopology(SystemTopology):
        pass

    # ============== #
    #  PROCESO
    # ============== #

    @dataclass
    class Token:

        identificator : Identification = field(default_factory = lambda : Identification(entity = Ontology.EntityType.TOKEN))
        state : State = field(default_factory = State)

    @dataclass
    class Signal:
        pass    

    @dataclass
    class Component(Entity):

        identificator : Identification = field(init = False, default_factory = lambda : Identification(entity=Ontology.EntityType.COMPONENT))



    class System(Entity):

        def __init__(self):

            self.identificator = Identification(label = "System", entity = Ontology.EntityType.SYSTEM)
            self.system_topology : 




    
"""
FUNCION

Tenemos una Topologia de Estado con Variables 

Construir una Topologia de Componentes a partir de las relaciones de las variables

una variable pertenece al estado de un componente o sistema, si esa variable tiene una relacion con la variable de otro estado e otros componente o sistema
el grafo debe tener una relacion de el componente respecto al otro componente



"""