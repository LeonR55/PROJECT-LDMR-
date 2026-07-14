from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Union, Any
from enum import Enum
from uuid import uuid4, UUID
import networkx as nx
import copy

# ==========================================
# 1. Declaración de Taxonomias y Tipos
# ==========================================
class Entity(Enum):
    ERGON = "Ergon"
    SUPER_ERGON = "Super_Ergon"
    DYNAMIC_PARTICLE = "Dynamic_Particle"
    PHORON = "Phoron"
    TELION = "Telion"
    PHORON_BUS = "Phoron_Bus"
    INTER_SYSTEMIC_PHORON_BUS = "Inter_Systemic_Phoron_Bus"
    VARIABLE_DESCRIPTOR = "Variable_Descriptor"

class ErgonicDescriptiveLevels(Enum):
    PASS = 0

class StateVariableTypes(Enum):
    PASS = 0

# ==========================================
# 2. Servicios Base y Metadata
# ==========================================
@dataclass
class Metadata:
    id: UUID = field(default_factory=uuid4)
    label: str | None = None
    entity_type: Entity | None = None

@dataclass
class MasterRegistry:
    global_entity_registry: ClassVar[dict[UUID, Any]] = {}
    global_ergon_registry: ClassVar[dict[UUID, 'Ergon']] = {}
    global_dynamic_particle_registry: ClassVar[dict[UUID, 'DynamicParticle']] = {}
    global_bus_registry: ClassVar[dict[UUID, Any]] = {}
    global_descriptor_registry: ClassVar[dict[UUID, 'VariableDescriptor']] = {}

    @classmethod
    def register_entity(cls, entity: Any, entity_type: Entity | None = None) -> None:
        if not hasattr(entity, "metadata"):
            return
        entity_id = entity.metadata.id
        if entity_id in cls.global_entity_registry:
            return

        cls.global_entity_registry[entity_id] = entity
        if entity_type is not None:
            entity.metadata.entity_type = entity_type

        if isinstance(entity, Ergon):
            cls.global_ergon_registry[entity_id] = entity
        elif isinstance(entity, DynamicParticle):
            cls.global_dynamic_particle_registry[entity_id] = entity
        elif isinstance(entity, (PhoronBus, InterSystemicPhoronBus)):
            cls.global_bus_registry[entity_id] = entity
        elif isinstance(entity, VariableDescriptor):
            cls.global_descriptor_registry[entity_id] = entity

    @classmethod
    def register_ergon(cls, ergon: 'Ergon') -> None:
        cls.register_entity(ergon, Entity.ERGON)

    @classmethod
    def register_dynamic_particle(cls, particle: 'DynamicParticle') -> None:
        cls.register_entity(particle, Entity.DYNAMIC_PARTICLE)

    @classmethod
    def register_bus(cls, bus: Any) -> None:
        if isinstance(bus, InterSystemicPhoronBus):
            cls.register_entity(bus, Entity.INTER_SYSTEMIC_PHORON_BUS)
        else:
            cls.register_entity(bus, Entity.PHORON_BUS)

    @classmethod
    def register_descriptor(cls, descriptor: 'VariableDescriptor') -> None:
        cls.register_entity(descriptor, Entity.VARIABLE_DESCRIPTOR)

    @classmethod
    def get_entity(cls, entity_id: UUID) -> Any:
        if entity_id in cls.global_entity_registry:
            return cls.global_entity_registry[entity_id]
        raise ValueError(f"The {entity_id} ID does not correspond to any registered entity")

    @classmethod
    def get_ergon(cls, ergon_id: UUID) -> 'Ergon':
        if ergon_id in cls.global_ergon_registry:
            return cls.global_ergon_registry[ergon_id]
        raise ValueError(f"The {ergon_id} ID does not correspond to any registered Ergon")

    @classmethod
    def get_superergon(cls, ergon_id: UUID) -> 'SuperErgon':
        ergon = cls.get_ergon(ergon_id)
        if isinstance(ergon, SuperErgon):
            return ergon
        raise ValueError(f"The {ergon_id} ID does not correspond to a registered SuperErgon")

    @classmethod
    def get_dynamic_particle(cls, particle_id: UUID) -> 'DynamicParticle':
        if particle_id in cls.global_dynamic_particle_registry:
            return cls.global_dynamic_particle_registry[particle_id]
        raise ValueError(f"The {particle_id} ID does not correspond to any registered Dynamic Particle")

    @classmethod
    def get_bus(cls, bus_id: UUID) -> Any:
        if bus_id in cls.global_bus_registry:
            return cls.global_bus_registry[bus_id]
        raise ValueError(f"The {bus_id} ID does not correspond to any registered Bus")

    @classmethod
    def get_descriptor(cls, descriptor_id: UUID) -> 'VariableDescriptor':
        if descriptor_id in cls.global_descriptor_registry:
            return cls.global_descriptor_registry[descriptor_id]
        raise ValueError(f"The {descriptor_id} ID does not correspond to any registered Variable Descriptor")

    @classmethod
    def get_type(cls, entity_to_consult: UUID) -> Any:
        entity = cls.get_entity(entity_to_consult)
        return type(entity)

# ==========================================
# 3. Descripción de Estados
# ==========================================
@dataclass
class VariableDescriptor:
    name: str
    domain_type: str  # "numeric", "boolean", "symbolic", "categorical"
    valid_values: list[Any] | None = None
    metadata: Metadata = field(default_factory=lambda: Metadata(entity_type=Entity.VARIABLE_DESCRIPTOR))

    def __post_init__(self):
        MasterRegistry.register_descriptor(self)

@dataclass
class StateVariable:
    variable_descriptor: VariableDescriptor
    value: Any

    def validate_variable(self) -> None:
        descriptor = self.variable_descriptor

        if descriptor.valid_values is not None and self.value not in descriptor.valid_values:
            raise ValueError(
                f"The {self.value} value is not valid for the declared State Variable '{descriptor.name}'"
            )

        dtype = descriptor.domain_type.lower()
        if dtype == "numeric":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Type Error: '{descriptor.name}' is expected to be numeric, received {type(self.value).__name__}"
                )
        elif dtype == "boolean":
            if not isinstance(self.value, bool):
                raise TypeError(
                    f"Type Error: '{descriptor.name}' is expected to be boolean, received {type(self.value).__name__}"
                )
        elif dtype in {"symbolic", "categorical"}:
            if not isinstance(self.value, (str, int)):
                raise TypeError(
                    f"Type Error: The symbol '{descriptor.name}' must be a string or integer."
                )

@dataclass
class State:
    variables: dict[str, StateVariable] = field(default_factory=dict)

    def set_variable(self, name: str, variable_descriptor: VariableDescriptor, value: Any) -> None:
        variable = StateVariable(variable_descriptor=variable_descriptor, value=value)
        variable.validate_variable()
        self.variables[name] = variable

    def get_value(self, name: str) -> Any:
        if name in self.variables:
            return self.variables[name].value
        return None

    def satisfies_requirements(self, required_names: list[str]) -> bool:
        return all(req in self.variables for req in required_names)

# ==========================================
# 4. Entidades Fundamentales
# ==========================================
@dataclass
class Ergon:
    metadata: Metadata = field(default_factory=lambda: Metadata(entity_type=Entity.ERGON))
    ergonic_state: State = field(default_factory=State)
    ergonic_processes: list['ErgonicProcess'] = field(default_factory=list)

    def __post_init__(self):
        MasterRegistry.register_ergon(self)

    def add_process(self, process: 'ErgonicProcess') -> None:
        self.ergonic_processes.append(process)

    def has_processes(self) -> bool:
        return len(self.ergonic_processes) > 0

@dataclass
class SuperErgon(Ergon):
    ergonic_structure: 'ErgonicStructure' = field(default_factory=lambda: ErgonicStructure())

    def __post_init__(self):
        super().__post_init__()
        if self.ergonic_structure is None:
            self.ergonic_structure = ErgonicStructure()

@dataclass
class DynamicParticle:
    metadata: Metadata = field(default_factory=lambda: Metadata(entity_type=Entity.DYNAMIC_PARTICLE))
    entangled_ergon: Ergon | None = field(init=False, default=None)

    def __post_init__(self):
        MasterRegistry.register_dynamic_particle(self)

    def entangle_ergon(self, ergon: Ergon) -> None:
        self.entangled_ergon = ergon

@dataclass
class Telion(DynamicParticle):
    state_charge: State = field(default_factory=State)

    def __post_init__(self):
        super().__post_init__()

@dataclass
class Phoron(DynamicParticle):
    state_charge: State = field(default_factory=State)

    def __post_init__(self):
        super().__post_init__()

# ==========================================
# 5. Elementos Composicionales de Estructura y Procesos
# ==========================================
@dataclass
class ErgonicStructure:
    structural_description: nx.DiGraph = field(default_factory=nx.DiGraph)
    dynamical_description: nx.DiGraph = field(default_factory=nx.DiGraph)
    ergonic_references: list[UUID] = field(default_factory=list)

    def set_references(self, nodes: list[UUID]) -> None:
        self.ergonic_references = list(nodes)

    def build_structural_graph(self, edges: list[tuple[UUID, UUID]]) -> None:
        self.structural_description.add_nodes_from(self.ergonic_references)
        self.structural_description.add_edges_from(edges)

    def build_dynamical_graph(self, edges: list[tuple[UUID, UUID]]) -> None:
        self.dynamical_description.add_nodes_from(self.ergonic_references)
        self.dynamical_description.add_edges_from(edges)

    def add_node(self, node: UUID) -> None:
        self.ergonic_references.append(node)
        self.structural_description.add_node(node)
        self.dynamical_description.add_node(node)

    def add_edge(self, source: UUID, target: UUID, dynamic: bool = False) -> None:
        if dynamic:
            self.dynamical_description.add_edge(source, target)
        else:
            self.structural_description.add_edge(source, target)

    def remove_node(self, node: UUID) -> None:
        self.ergonic_references = [ref for ref in self.ergonic_references if ref != node]
        self.structural_description.remove_node(node)
        self.dynamical_description.remove_node(node)

    def remove_edge(self, source: UUID, target: UUID, dynamic: bool = False) -> None:
        if dynamic:
            self.dynamical_description.remove_edge(source, target)
        else:
            self.structural_description.remove_edge(source, target)

    def get_node_entities(self) -> list[Any]:
        return [MasterRegistry.get_entity(node) for node in self.ergonic_references]

    def get_successors(self, node: UUID, dynamic: bool = False) -> list[UUID]:
        graph = self.dynamical_description if dynamic else self.structural_description
        return list(graph.successors(node))

    def get_predecessors(self, node: UUID, dynamic: bool = False) -> list[UUID]:
        graph = self.dynamical_description if dynamic else self.structural_description
        return list(graph.predecessors(node))

    def describe_graph(self, dynamic: bool = False) -> str:
        graph = self.dynamical_description if dynamic else self.structural_description
        return f"Nodes: {list(graph.nodes())}, Edges: {list(graph.edges())}"

@dataclass
class ErgonicProcess:
    identity_output: Telion
    telionic_valence: list[Telion] = field(default_factory=list)

    @property
    def telioncity(self) -> int:
        return len(self.telionic_valence)

    def instance_identity_telion(self) -> Telion:
        instanced_telion = copy.deepcopy(self.identity_output)
        if self.identity_output.entangled_ergon is not None:
            instanced_telion.entangle_ergon(self.identity_output.entangled_ergon)
        self.telionic_valence.append(instanced_telion)
        return instanced_telion

    def transform_telion_into_phoron(self, telion: Telion, exit_phoron_bus: 'PhoronBus') -> 'Phoron':
        phoron = Phoron(state_charge=copy.deepcopy(telion.state_charge))
        if telion.entangled_ergon is not None:
            phoron.entangle_ergon(telion.entangled_ergon)
        exit_phoron_bus.receive_phoron(phoron)
        return phoron

# ==========================================
# 6. Interfaces y Servicios de Buses
# ==========================================
@dataclass
class PhoronBus:
    metadata: Metadata = field(default_factory=lambda: Metadata(entity_type=Entity.PHORON_BUS))
    transmitter_ergon: UUID | None = None
    receiver_ergon: UUID | None = None
    transmission_queue: list[UUID] = field(default_factory=list)

    def __post_init__(self):
        MasterRegistry.register_bus(self)

    @classmethod
    def create_phoron_bus_from_topology(cls, from_superergon_id: UUID) -> list['PhoronBus']:
        superergon = MasterRegistry.get_superergon(from_superergon_id)
        outgoing = list(superergon.ergonic_structure.dynamical_description.successors(from_superergon_id))
        if not outgoing:
            raise ValueError("No dynamic topology edges found for this SuperErgon")

        buses = []
        for target in outgoing:
            bus = cls(transmitter_ergon=from_superergon_id, receiver_ergon=target)
            buses.append(bus)
        return buses

    def receive_phoron(self, phoron: UUID | Phoron) -> None:
        phoron_id = phoron.metadata.id if not isinstance(phoron, UUID) else phoron
        MasterRegistry.get_entity(phoron_id)
        self.transmission_queue.append(phoron_id)

    def transmit_phoron(self) -> UUID | None:
        if not self.transmission_queue:
            return None
        return self.transmission_queue.pop(0)

    def peek_next_phoron(self) -> UUID | None:
        return self.transmission_queue[0] if self.transmission_queue else None

    def is_empty(self) -> bool:
        return len(self.transmission_queue) == 0

@dataclass
class InterSystemicPhoronBus:
    metadata: Metadata = field(default_factory=lambda: Metadata(entity_type=Entity.INTER_SYSTEMIC_PHORON_BUS))
    ergon_A: UUID | None = None
    ergon_B: UUID | None = None

    def __post_init__(self):
        MasterRegistry.register_bus(self)

    @classmethod
    def create(cls, ergon_A: UUID, ergon_B: UUID) -> 'InterSystemicPhoronBus':
        if ergon_A == ergon_B:
            raise ValueError("InterSystemicPhoronBus requires two different ergon endpoints")
        MasterRegistry.get_ergon(ergon_A)
        MasterRegistry.get_ergon(ergon_B)
        return cls(ergon_A=ergon_A, ergon_B=ergon_B)

# ==========================================
# 7. Nota sobre objetivos del sistema
# ==========================================
# Este módulo modela un espacio ergónico en el que:
# - Los Ergones pueden poseer estado y procesos.
# - Solo los SuperErgones pueden poseer ErgonicStructure para describir sistemas internos.
# - Teliones y Phorones son partículas dinámicas con carga de estado.
# - El MasterRegistry mantiene registro global de ergones, partículas, buses y descriptores.
# - Las estructuras ergónicas describen topologías estáticas y dinámicas con NetworkX.
# - Los Buses Phorónicos pueden crearse a partir de las aristas dinámicas de un SuperErgon.

if __name__ == "__main__":
    print("=== Sistema ergónico de ejemplo ===")

    # Descriptor y estado base
    energy_descriptor = VariableDescriptor(name="energy", domain_type="numeric")
    status_descriptor = VariableDescriptor(name="status", domain_type="symbolic", valid_values=["idle", "active", "done"])

    ergon_a = Ergon()
    ergon_a.ergonic_state.set_variable("energy", energy_descriptor, 42)
    ergon_a.ergonic_state.set_variable("status", status_descriptor, "idle")

    ergon_b = Ergon()
    ergon_b.ergonic_state.set_variable("energy", energy_descriptor, 17)
    ergon_b.ergonic_state.set_variable("status", status_descriptor, "active")

    system = SuperErgon()
    system.ergonic_state.set_variable("energy", energy_descriptor, 100)
    system.ergonic_state.set_variable("status", status_descriptor, "active")

    structure = system.ergonic_structure
    structure.set_references([system.metadata.id, ergon_a.metadata.id, ergon_b.metadata.id])
    structure.build_structural_graph([
        (system.metadata.id, ergon_a.metadata.id),
        (system.metadata.id, ergon_b.metadata.id),
    ])
    structure.build_dynamical_graph([
        (system.metadata.id, ergon_a.metadata.id),
        (ergon_a.metadata.id, ergon_b.metadata.id),
    ])

    # Proceso ergónico simple
    identity_telion = Telion()
    identity_telion.entangle_ergon(system)
    identity_telion.state_charge.set_variable("energy", energy_descriptor, 5)
    identity_telion.state_charge.set_variable("status", status_descriptor, "idle")

    ergonic_process = ErgonicProcess(identity_output=identity_telion)
    system.add_process(ergonic_process)

    print("-- Master Registry --")
    print("Total entidades:", len(MasterRegistry.global_entity_registry))
    print("Total ergones:", len(MasterRegistry.global_ergon_registry))
    print("Total partículas dinámicas:", len(MasterRegistry.global_dynamic_particle_registry))
    print("Total buses:", len(MasterRegistry.global_bus_registry))
    print("Total descriptores:", len(MasterRegistry.global_descriptor_registry))

    print("\n-- Ergones registrados --")
    for ergon_id, ergon in MasterRegistry.global_ergon_registry.items():
        print(f"Ergon {ergon_id} tipo={type(ergon).__name__} estado={list(ergon.ergonic_state.variables.keys())}")

    print("\n-- SuperErgon estructura --")
    print("Nodos estructurales:", structure.describe_graph(dynamic=False))
    print("Nodos dinámicos:", structure.describe_graph(dynamic=True))

    phoron_buses = PhoronBus.create_phoron_bus_from_topology(system.metadata.id)
    for idx, bus in enumerate(phoron_buses, start=1):
        print(f"Bus {idx}: {bus.metadata.id} from {bus.transmitter_ergon} to {bus.receiver_ergon}")

    print("\n-- Proceso ergónico --")
    telion_instance = ergonic_process.instance_identity_telion()
    print("Telion instanciado:", telion_instance.metadata.id)
    print("Telion entangled ergon:", telion_instance.entangled_ergon.metadata.id if telion_instance.entangled_ergon else None)
    print("Telion estado:", {name: telion_instance.state_charge.get_value(name) for name in telion_instance.state_charge.variables})

    print("\n-- Transformación a Phoron --")
    first_bus = phoron_buses[0]
    phoron = ergonic_process.transform_telion_into_phoron(telion_instance, first_bus)
    print("Phoron creado:", phoron.metadata.id)
    print("Phoron entangled ergon:", phoron.entangled_ergon.metadata.id if phoron.entangled_ergon else None)
    print("Phoron en cola del bus:", first_bus.transmission_queue)

    print("\n-- Transmisión de Phoron --")
    transmitted = first_bus.transmit_phoron()
    print("Phoron transmitido id:", transmitted)
    print("Bus vacío?:", first_bus.is_empty())

    inter_bus = InterSystemicPhoronBus.create(ergon_a.metadata.id, ergon_b.metadata.id)
    print("\n-- Bus inter-sistémico --")
    print("InterBus id:", inter_bus.metadata.id)
    print("InterBus endpoints:", inter_bus.ergon_A, inter_bus.ergon_B)

    print("\n=== Fin de la simulación ===")
