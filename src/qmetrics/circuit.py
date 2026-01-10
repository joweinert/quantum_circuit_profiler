from __future__ import annotations
from functools import cached_property
from itertools import permutations
import math
import numpy as np
import networkx as nx
import rustworkx as rx
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Qubit, CommutationChecker
from qiskit.circuit.random import random_circuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager, CouplingMap, Target
from qiskit.transpiler.passes import FilterOpNodes
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.visualization import dag_drawer
from qiskit import transpile
import pickle
from itertools import combinations
import hashlib
from pathlib import Path

from .utils import *


class CircuitWrapper:
    """Wrapper class for a qiskit.QuantumCircuit
    with the purpose of exposing an API for a wide
    range of features characterizing a QuantumCircuit.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        name: str = "wrapped",
        desc: str = "No description provided",
        filter_gates: tuple[str, ...] | None = (
            "measure",
            "reset",
            "barrier",
            "delay",
        ),
        basis_gates: list[str] | None = None,
        opt_level: int = 0,
        random_seed: int = 0,
    ):
        """Initializes the CircuitWrapper object by wrapping a Qiskit QuantumCircuit instance.

        Attributes:
        circuit (QuantumCircuit): The wrapped Qiskit QuantumCircuit instance.
        name (str): The name of the circuit. Defaults to "wrapped".
        desc (str): The description of the circuit. Defaults to "No description provided".
        filter_gates (tuple[str, ...] | None): A tuple of gate names to filter out from the circuit. Defaults to ("measure", "reset", "barrier", "delay").
        basis_gates (list[str] | None): A list of basis gate names to transpile the circuit to. If None, no transpilation is performed. Defaults to None.
        opt_level (int): Optimization level for transpilation to basis gates. Defaults to 0. Only used if basis_gates is not None.
        random_seed (int): Seed for random number generation. Defaults to 0.
        """
        if filter_gates and len(filter_gates) > 0:
            circuit = CircuitWrapper._filter_gates_by_name(circuit, set(filter_gates))

        if basis_gates is not None:
            circuit = transpile(
                circuit,
                basis_gates=basis_gates,
                coupling_map=None,
                optimization_level=opt_level,
            )

        self.circuit = circuit

        self.name = name
        self.desc = desc

        self._filter_gates = filter_gates
        self._basis_gates = basis_gates
        self._opt_level = opt_level
        self._rng = np.random.default_rng(random_seed)

    @staticmethod
    def _predict_mqt_size(name: str, n_param: int) -> int:
        """
        Predicts the ACTUAL physical qubit count MQTBench needs based on the input parameter.
        Most are 1:1, until now only multiplier is different.
        """
        if name == "multiplier":
            return (2 * n_param) - 4
        # if i ever find more weird cases ill add them here
        return n_param

    @classmethod
    def from_mqt_bench(
        cls,
        name: str,
        n_qubits: int,
        *,
        desc: str | None = None,
        filter_gates: tuple[str, ...] | None = ("measure", "reset", "barrier", "delay"),
        basis_gates: list[str] | None = None,
        opt_level: int = 0,
        mqt_opt_level: int | None = None,
        mqt_level=None,
        mqt_target=None,
        **mqt_kwargs,
    ) -> CircuitWrapper:
        """
        Wraps an MQTBench benchmark circuit into a CircuitWrapper object.
        See MQTBench documentation to find the full list of available benchmarks.
        Alternatively, use get_benchmark_catalog().keys() to list available benchmark names (from mqt.bench.benchmarks import get_benchmark_catalog).

        Args:
            name (str): the name of the benchmark circuit to load from MQTBench.
            n_qubits (int): the number of qubits in the circuit.
            desc (str | None): a description of the circuit. Defaults to None.
            filter_gates (tuple[str, ...] | None): a tuple of gate names to filter out. Defaults to ("measure", "reset", "barrier", "delay").
            basis_gates (list[str] | None): A list of basis gate names to transpile the circuit to. If None, no transpilation is performed. Defaults to None.
            opt_level (int): Optimization level for transpilation to basis gates. Defaults to 0.
            mqt_opt_level (int | None): Optimization level for MQTBench circuit generation. Defaults to opt_level.
            mqt_level (BenchmarkLevel | None): Benchmark level for MQTBench circuit generation.
                If None, defaults to BenchmarkLevel.INDEP. If BenchmarkLevel.NATIVEGATES or MAPPED, a target is needed.
            mqt_target (str | Target | None): Target for MQTBench circuit generation. Required for NATIVEGATES and MAPPED levels.
                If str, it is interpreted as the name of the target or device. If None, defaults to "ibm_falcon" for NATIVEGATES
                and "ibm_falcon_27" for MAPPED. For MAPPED level, the target's number of qubits must match n_qubits.
            **mqt_kwargs: Additional keyword arguments to pass to MQTBench's get_benchmark function.
        Returns:
            CircuitWrapper: the wrapped CircuitWrapper object.
        """
        try:
            from mqt.bench import BenchmarkLevel, get_benchmark
            from mqt.bench.benchmarks import get_benchmark_catalog
            from mqt.bench.targets import get_target_for_gateset, get_device
        except ImportError as e:
            raise ImportError(
                """Optional dependency 'mqt.bench' is not installed. 
                Please install it using 'pip install qiskit-circuit-profiler[benchmarks]' 
                or install it manually to use this feature."""
            ) from e

        if name not in get_benchmark_catalog().keys():

            raise ValueError(
                f"Circuit {name} not recognized. Available: {list(get_benchmark_catalog().keys())}"
            )

        mqt_kwargs["opt_level"] = opt_level if mqt_opt_level is None else mqt_opt_level
        mqt_kwargs["level"] = mqt_level or BenchmarkLevel.INDEP
        if not isinstance(mqt_kwargs["level"], BenchmarkLevel):
            raise ValueError(
                f"Invalid BenchmarkLevel: {mqt_kwargs['level']}. Must be one of {list(BenchmarkLevel)} or None."
            )
        # we need a target for NATIVEGATES and MAPPED levels
        target_obj: Target | None = None
        needed_qubits = cls._predict_mqt_size(name, n_qubits)
        if isinstance(mqt_target, Target):
            target_obj = mqt_target
            if mqt_level not in (BenchmarkLevel.NATIVEGATES, BenchmarkLevel.MAPPED):
                warning(
                    f"Target provided but BenchmarkLevel is {mqt_level}. "
                    "Target can only be used with NATIVEGATES or MAPPED levels."
                )

        elif mqt_level == BenchmarkLevel.NATIVEGATES:
            if mqt_target is None:
                mqt_target = "ibm_falcon"
            if isinstance(mqt_target, str):
                target_obj = get_target_for_gateset(
                    mqt_target, num_qubits=needed_qubits
                )

        elif mqt_level == BenchmarkLevel.MAPPED:
            if mqt_target is None:
                mqt_target = "ibm_falcon_27"
            if isinstance(mqt_target, str):
                target_obj = get_device(mqt_target)

        if target_obj is not None:
            if target_obj.num_qubits < needed_qubits:
                raise ValueError(
                    f"Target device has {target_obj.num_qubits} qubits, but circuit needs {needed_qubits}. "
                    "Impossible to map."
                )
            else:
                mqt_kwargs["target"] = target_obj

        # caching.....to not map again and again the same circuits
        # stable identifier for target, level...for hash naming
        def _stable_repr(obj):
            if hasattr(obj, "name"):
                return obj.name
            if hasattr(obj, "description"):
                return obj.description
            return str(obj)

        hash_str = f"{name}|{n_qubits}|"
        for k in sorted(mqt_kwargs.keys()):
            hash_str += f"{k}:{_stable_repr(mqt_kwargs[k])};"

        # hash as file name -> identifier
        file_hash = hashlib.md5(hash_str.encode("utf-8")).hexdigest()

        cache_dir = Path("./.circ_cache")
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"{name}_{n_qubits}_{file_hash}.pickle"

        qc = None
        # try to load
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    qc = pickle.load(f)
                    print_colored(
                        f"  -> Loaded {name} ({n_qubits}q) from cache.", bcolors.OKBLUE
                    )
            except Exception:
                warning(f"Cache file corrupted for {name}, regenerating...")
        if qc is None:
            # get it from mqtbench and cache it
            qc = get_benchmark(
                benchmark=name,
                circuit_size=n_qubits,
                **mqt_kwargs,
            )
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(qc, f)
            except Exception as e:
                warning(f"Could not write cache file: {e}")
        return cls(
            qc,
            name=name,
            desc=desc
            or f"MQTBench benchmark '{name}' with {n_qubits} qubits: {get_benchmark_catalog()[name]}",
            filter_gates=filter_gates,
            basis_gates=basis_gates,
            opt_level=opt_level,
        )

    @classmethod
    def from_qasm_file(
        cls,
        file_path: str,
        *,
        qasm3_file: bool = False,
        name: str | None = None,
        desc: str | None = None,
        filter_gates: tuple[str, ...] | None = ("measure", "reset", "barrier", "delay"),
        basis_gates: list[str] | None = None,
        opt_level: int = 0,
    ) -> CircuitWrapper:
        """Creates a CircuitWrapper object by loading a QuantumCircuit from a QASM file.

        Args:
            file_path (str): The path to the QASM file.
            qasm3_file (bool, optional): Whether the file is in QASM3 format. Defaults to False, meaning QASM2 format.
            name (str | None): The name of the circuit. Defaults to None.
            desc (str | None): A description of the circuit. Defaults to None.
            filter_gates (tuple[str, ...] | None): A tuple of gate names to filter out. Defaults to ("measure", "reset", "barrier", "delay").
            basis_gates (list[str] | None): A list of basis gate names to transpile the circuit to. If None, no transpilation is performed. Defaults to None.
            opt_level (int): Optimization level for transpilation to basis gates. Defaults to 0.

        Returns:
            CircuitWrapper: The wrapped CircuitWrapper object.
        """
        if qasm3_file:
            qc: QuantumCircuit = qasm3.load(file_path)
        else:
            qc = QuantumCircuit.from_qasm_file(file_path)
        return cls(
            qc,
            name=name or f"circuit_from_{file_path}",
            desc=desc or f"Circuit loaded from QASM file: {file_path}",
            filter_gates=filter_gates,
            basis_gates=basis_gates,
            opt_level=opt_level,
        )

    @classmethod
    def from_random_circuit(
        cls,
        n_qubits: int,
        *,
        name: str | None = "random circuit",
        desc: str | None = None,
        filter_gates: tuple[str, ...] | None = ("measure", "reset", "barrier", "delay"),
        basis_gates: list[str] | None = None,
        opt_level: int = 0,
        **random_kwargs,
    ) -> CircuitWrapper:
        """Creates a random qiskit.QuantumCircuit using Qiskit's random_circuit function and wraps it into a CircuitWrapper object.

        Args:
            n_qubits (int): The number of qubits in the circuit.
            name (str | None): The name of the circuit. Defaults to "random circuit".
            desc (str | None): A description of the circuit. Defaults to None.
            filter_gates (tuple[str, ...] | None): A tuple of gate names to filter out. Defaults to ("measure", "reset", "barrier", "delay").
            basis_gates (list[str] | None): A list of basis gate names to transpile the circuit to. If None, no transpilation is performed. Defaults to None.
            opt_level (int): Optimization level for transpilation to basis gates. Defaults to 0.
            **random_kwargs: Additional keyword arguments to pass to Qiskit's random_circuit function.

        Returns:
            CircuitWrapper: The wrapped CircuitWrapper object.
        """
        qc = random_circuit(n_qubits, **random_kwargs)

        return cls(
            qc,
            name=name,
            desc=desc
            or f"Randomly generated circuit using Qiskit's random_circuit function. n_qubits={n_qubits}, "
            + ", ".join(f"{k}={v}" for k, v in random_kwargs.items()),
            filter_gates=filter_gates,
            basis_gates=basis_gates,
            opt_level=opt_level,
        )

    @classmethod
    def from_qiskit(
        cls,
        qc: QuantumCircuit,
        *,
        name: str | None = "wrapped",
        desc: str | None = "No description provided",
        filter_gates: tuple[str, ...] | None = ("measure", "reset", "barrier", "delay"),
        basis_gates: list[str] | None = None,
        opt_level: int = 0,
    ) -> CircuitWrapper:
        """Wraps a Qiskit QuantumCircuit instance into a CircuitWrapper object.
        Args:
            qc (QuantumCircuit): The Qiskit QuantumCircuit instance to wrap.
            name (str | None): The name of the circuit. Defaults to "wrapped".
            desc (str | None): A description of the circuit. Defaults to "No description provided".
            filter_gates (tuple[str, ...] | None): A tuple of gate names to filter out. Defaults to ("measure", "reset", "barrier", "delay").
            basis_gates (list[str] | None): A list of basis gate names to transpile the circuit to. If None, no transpilation is performed. Defaults to None.
            opt_level (int): Optimization level for transpilation to basis gates. Defaults to 0.

        Returns:
            CircuitWrapper: The wrapped CircuitWrapper object.
        """
        return cls(
            qc,
            name=name,
            desc=desc,
            filter_gates=filter_gates,
            basis_gates=basis_gates,
            opt_level=opt_level,
        )

    def apply_connectivity(
        self,
        connectivity: list[list[int]] | CouplingMap,
        basis_gates: list[str] | None = None,
        optimization_level: int = 0,
    ) -> CircuitWrapper:
        """
        Transpiles the current circuit to a specific coupling map.
        Returns a NEW CircuitWrapper instance (does not modify self).
        """
        # Creates a CouplingMap from the list of edges
        if isinstance(connectivity, list):
            cmap = CouplingMap(connectivity)
        else:
            cmap = connectivity

        # Transpile
        mapped_qc = transpile(
            self.circuit,
            coupling_map=cmap,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )

        return CircuitWrapper(
            mapped_qc,
            name=f"{self.name}_mapped",
            desc=f"{self.desc} | Mapped to specific connectivity",
        )

    ### size metrics

    @property
    def num_qubits(self) -> int:
        """
        Returns:
            The number of qubits in the circuit: $n_q$.
        """
        return self.circuit.num_qubits

    @property
    def num_active_qubits(self) -> int:
        """
        Returns the number of qubits that are actually touched by at least one gate.
        """
        active = set()
        for instruction in self.circuit.data:
            for q in instruction.qubits:
                active.add(q)
        return len(active)

    @property
    def num_clbits(self) -> int:
        """
        Returns:
            The number of classical bits ($n_c$).
        """
        return self.circuit.num_clbits

    @property
    def num_gates(self) -> int:
        """
        Returns:
            The size of the circuit: $n_g$ (number of operations/gates).
        """
        return self.circuit.size()

    @property
    def num_parameters(self) -> int:
        """
        Returns:
            The number of free parameters in the circuit (e.g., for VQE/QAOA).
        """
        return self.circuit.num_parameters

    @cached_property
    def num_1q_gates(self) -> int:
        """
        Returns:
            The number of one-qubit gates in the circuit: $n_{1qg}$.
        """
        return len(self.one_qubit_gates)

    @cached_property
    def num_2q_gates(self) -> int:
        """
        Returns:
            The number of two-qubit gates in the circuit: $n_{2qg}$.
        """
        return len(self.two_qubit_gates)

    @property
    def pct_2q_gates(self) -> float:
        """
        Returns:
            The ratio of two-qubit gates to total gates: $n_{2qg} / n_g$.
        """
        ng = self.num_gates
        if ng == 0:
            return 0.0
        return self.num_2q_gates / ng

    @property
    def num_3q_plus_gates(self) -> int:
        """
        Returns:
            The number of gates with > 2 qubits (e.g., Toffoli, Fredkin, MCX).
            These are the 'other' bucket.
        """
        return len(self.gt2_gates)

    @property
    def pct_3q_plus_gates(self) -> float:
        """
        Returns:
            The ratio of gates with > 2 qubits to total gates: n_{3q+g} / n_g.
        """
        ng = self.num_gates
        if ng == 0:
            return 0.0
        return self.num_3q_plus_gates / ng

    @property
    def num_multi_qubit_gates(self) -> int:
        """
        Returns:
            The total number of entangling gates (2q + 3q + ...).
            This is the TRUE measure of non-local operations.
        """
        return self.num_2q_gates + self.num_3q_plus_gates

    @property
    def pct_multi_qubit_gates(self) -> float:
        """
        Returns:
            Ratio of entangling gates to total gates.
            Replaces 'pct_2q_gates' as the complexity proxy.
        """
        ng = self.num_gates
        if ng == 0:
            return 0.0
        return self.num_multi_qubit_gates / ng

    @cached_property
    def depth(self) -> int:
        """
        Returns:
            The depth of the circuit: $d$.
        """
        return self.circuit.depth()

    @property
    def gate_density(self) -> float:
        """
        Average number of gates per qubit-layer.
        Formula: $n_g / (n_q \\times d)$
        ~ High density = functional parallelism. Low density = idle qubits.
        """
        area = self.num_qubits * self.depth
        if area == 0:
            return 0.0
        return self.num_gates / area

    ### IG metrics

    @cached_property
    def ig_adj_mat(self) -> np.ndarray:
        """
        Returns:
            The adjacency matrix of the circuit including one-qubit gates in the diagonal.
        """
        adj_mat = self._compute_adj_mat()
        return adj_mat

    @property
    def ig_adj_mat_2q(self) -> np.ndarray:
        """
        Returns:
            The adjacency matrix of the circuit considering only two-qubit gates -> 0s in diagonal.
        """
        adj_mat = self.ig_adj_mat.copy()
        np.fill_diagonal(adj_mat, 0)
        return adj_mat

    @cached_property
    def ig(self) -> rx.PyGraph:
        """Weighted Interaction Graph (Weights = Gate Counts)"""
        matrix_float = self.ig_adj_mat_2q.astype(np.float64)
        G = rx.PyGraph.from_adjacency_matrix(matrix_float)

        for i, count in enumerate(self.node_1q_counts):
            G[i] = int(count)

        return G

    @cached_property
    def unweighted_ig(self) -> rx.PyGraph:
        """Unweighted Interaction Graph (Topology only)"""
        binary_adj = (self.ig_adj_mat_2q > 0).astype(np.float64)

        G = rx.PyGraph.from_adjacency_matrix(binary_adj)
        return G

    @cached_property
    def nx_ig(self) -> nx.Graph:
        """
        Returns:
            The interaction graph as a NetworkX graph with weights.
            (Edges have gate_count as weight attribute, node attributes include one-qubit gate counts).
        """
        G = nx.from_numpy_array(self.ig_adj_mat_2q)

        counts = self.node_1q_counts
        attrs = {i: {"n1q": int(c)} for i, c in enumerate(counts)}
        nx.set_node_attributes(G, attrs)

        return G

    @cached_property
    def nx_unweighted_ig(self) -> nx.Graph:
        """
        Returns:
            The unweighted interaction graph as a NetworkX graph.
            (Edges exist where gate_count > 0, but weights = 1).
        """
        binary_adj = (self.ig_adj_mat_2q > 0).astype(np.int64)

        G = nx.from_numpy_array(binary_adj)
        return G

    @property
    def ig_aspl(self) -> float:
        r"""Average hop count between all pairs of nodes in the IG. Computed using rustworkx's unweighted_average_shortest_path_length(G) function. It is defined by rustworkx as:
        \\[aspl = \sum_{s,t \in V, s \neq t}\frac{d(s,t)}{n(n-1)}\\]

        where $V$ is the set of nodes in $G$, $d(s,t)$ is the shortest path length from $s$ to $t$, and $n$ is the number of nodes in $G$.

        Returns:
            The average shortest path length in the interaction graph.
        """
        if self.num_qubits < 2:
            return 0.0
        aspl = rx.unweighted_average_shortest_path_length(
            self.unweighted_ig, disconnected=True
        )
        return float(aspl) if not np.isnan(aspl) else 0.0

    @property
    def ig_std_adj_mat(self) -> float:
        """
        Returns:
            Std. dev. of the interaction graphs edge-weight distribution. Undirected: takes upper triangle, including zeros for non-edges: $\\sigma(A)$
        """
        A = self.ig_adj_mat_2q
        iu = np.triu_indices(A.shape[0], k=1)
        return float(np.std(A[iu]))

    @property
    def ig_diameter(self) -> int:
        r"""The diameter of the IG, defined as the longest shortest path between any pair of nodes in the IG:
            \\[dm = \\max_{n_i \\in N}(\\epsilon_i),\\]
            where $\epsilon_i$ is the longest hop count from node $n_i$ to any other node in the IG.

        Returns:
            The diameter of the interaction graph.
        """
        if self.num_qubits < 2:
            return 0
        D = rx.graph_distance_matrix(self.unweighted_ig, null_value=float("inf"))
        mask = np.isfinite(D)

        if not np.any(mask):
            return 0
        return int(np.max(D[mask]))

    @cached_property
    def ig_betweenness_centrality(self) -> rx.CentralityMapping:
        """
        Returns:
            The maximum betweenness centrality among all nodes in the interaction graph. Normalized (between 0 and 1).
        """
        if self.num_qubits < 3:
            return {i: 0.0 for i in range(self.num_qubits)}

        return rx.betweenness_centrality(self.unweighted_ig, normalized=True)

    @property
    def ig_max_betweenness(self) -> float:
        """Equals Central Point Dominance as defined in Bandic et al. 2025.

        Returns:
            The maximal betweenness centrality of any node in the IG. Normalized between 0 and 1.
        """
        vals = self.ig_betweenness_centrality.values()
        return float(max(vals)) if vals else 0.0

    @cached_property
    def ig_degrees(self) -> np.ndarray:
        """
        Returns:
            Unweighted degree: #distinct neighbors per qubit. (how many distionct qubits each qubit interacts with via 2q gates)
        """
        A = self.ig_adj_mat_2q
        return np.asarray((A > 0).sum(axis=1), dtype=int).ravel()

    @property
    def ig_avg_degree(self) -> float:
        """
        Returns:
            The average degree of the interaction graph. (How many distinct qubits each qubit interacts with via 2q gates on average)
        """
        return float(np.mean(self.ig_degrees))

    @property
    def ig_max_degree(self) -> int:
        """
        Returns:
            The maximum degree of the interaction graph. (How many distinct qubits the most connected qubit interacts with via 2q gates)
        """
        return int(max(self.ig_degrees))

    @property
    def ig_std_degree(self) -> float:
        """
        Returns:
            The standard deviation of the degree distribution of the interaction graph. (How much the number of distinct qubits each qubit interacts with via 2q gates varies)
        """
        return float(np.std(self.ig_degrees))

    @cached_property
    def ig_strengths(self) -> np.ndarray:
        """
        Returns:
            Weighted degree (node strength): sum of 2q edge weights per qubit. (total number of 2q gates each qubit participates in)
        """
        A = self.ig_adj_mat_2q
        return np.asarray(A.sum(axis=1), dtype=float).ravel()

    @property
    def ig_avg_strength(self) -> float:
        """
        Returns:
            The average strength of the interaction graph. (How many 2q gates per qubit on average)
        """
        return np.mean(self.ig_strengths)

    @property
    def ig_max_strength(self) -> int:
        """
        Returns:
            The maximum strength of the interaction graph.
        """
        return int(max(self.ig_strengths))

    @property
    def ig_std_strength(self) -> float:
        """
        Returns:
            The standard deviation of the strength distribution of the interaction graph. (How much the number of 2q gates each qubit participates in varies)
        """
        return float(np.std(self.ig_strengths))

    @property
    def ig_max_cliques(self) -> tuple[int, int]:
        """Finds all maximal cliques in the unweighted IG.

        Returns:
            A tuple (num_max_cliques, size_max_clique).
        """
        if self.num_qubits < 2:
            return 0, 0

        max_size = 0
        num_max = 0

        # nx.find_cliques is a generator. We consume it one by one.
        for clique in nx.find_cliques(self.nx_unweighted_ig):
            size = len(clique)

            # larger? -> reset counter
            if size > max_size:
                max_size = size
                num_max = 1
            # same size? -> increment counter
            elif size == max_size:
                num_max += 1

        return num_max, max_size

    @property
    def ig_transitivity(self) -> float:
        """
        Returns:
            The transitivity of the interaction graph. (global clustering coefficient)
        """
        if self.num_qubits < 3:
            return 0.0
        return float(rx.transitivity(self.unweighted_ig))

    @property
    def ig_avg_clustering_coef(self) -> float:
        """Uses a NetworkX representation to compute the average local clustering coefficient of the IG.

        Returns:
            The average local clustering coefficient of the interaction graph.
        """
        if self.num_qubits < 3:
            return 0.0
        return float(nx.average_clustering(self.nx_unweighted_ig))

    @property
    def ig_vertex_connectivity(self) -> int:
        """The vertex connectivity of the IG, defined as the minimum number of nodes that need to be removed to disconnect the graph. Uses a NetworkX representation.

        Returns:
            The vertex connectivity of the interaction graph.
        """
        G = self.nx_unweighted_ig
        return 0 if G.number_of_nodes() < 2 else int(nx.node_connectivity(G))

    @property
    def ig_edge_connectivity(self) -> int:
        """The edge connectivity of the IG, defined as the minimum number of edges that need to be removed to disconnect the graph.

        Returns:
            The edge connectivity of the interaction graph.
        """
        cut_value, _ = rx.stoer_wagner_min_cut(self.unweighted_ig)
        edge_reliability = int(cut_value)
        return edge_reliability

    @cached_property
    def ig_coreness(self) -> dict[int, int]:
        """maximal $k$ for specific node $i$ such that $i$ is present in $k$-core graph but removed from $(k + 1)$-core (k-core is a subgraph of some graph made by removing all the nodes of degree <= k).

        Returns:
            A dictionary mapping node indices to their core numbers.
        """
        return rx.core_number(self.unweighted_ig)

    @property
    def ig_avg_coreness(self) -> float:
        """A k-core is a maximal subgraph in which each node has at least degree k. The core number of a node is the largest k for which the node is in the k-core.
        Returns:
            Average core number across all nodes in the IG.
        """
        coreness_values = np.fromiter(self.ig_coreness.values(), dtype=float)
        return float(coreness_values.mean())

    @property
    def ig_std_coreness(self) -> float:
        """
        Returns:
            Standard deviation of core numbers across all nodes in the IG.
        """
        coreness_values = np.fromiter(self.ig_coreness.values(), dtype=float)
        return float(coreness_values.std())

    @property
    def ig_max_coreness(self) -> int:
        """
        Returns:
            Maximum core number across all nodes in the IG.
        """
        return int(max(self.ig_coreness.values()))

    @property
    def ig_min_coreness(self) -> int:
        """
        Returns:
            Minimum core number across all nodes in the IG.
        """
        return int(min(self.ig_coreness.values()))

    @cached_property
    def ig_pageranks(self) -> np.ndarray:
        """
        Returns:
            The node-wise PageRank vector of the interaction graph. (Uses alpha=0.85)
        """
        Gdir = self.ig.to_directed()
        pr = rx.pagerank(Gdir, alpha=0.85)
        vec = np.zeros(Gdir.num_nodes(), dtype=float)
        for idx, score in pr.items():
            vec[idx] = score
        return vec

    @property
    def ig_max_pagerank(self) -> float:
        """
        Returns:
            The maximum PageRank score among all nodes in the interaction graph.
        """
        return float(max(self.ig_pageranks))

    @property
    def ig_min_pagerank(self) -> float:
        """
        Returns:
            The minimum PageRank score among all nodes in the interaction graph.
        """
        return float(min(self.ig_pageranks))

    @property
    def ig_std_pagerank(self) -> float:
        """
        Returns:
            The standard deviation of the PageRank scores among all nodes in the interaction graph.
        """
        return float(np.std(self.ig_pageranks))

    @property
    def ig_normalized_hhi_pagerank(self) -> float:
        """Normalized Herfindahl-Hirschman Index (HHI) of the PageRank vector.

        Returns:
            A float value between 0 and 1 indicating the concentration of PageRank scores.
        """
        p = self.ig_pageranks
        N = p.size
        if N <= 1:
            return 0.0
        hhi = float(np.sum(p * p))
        return (hhi - 1.0 / N) / (1.0 - 1.0 / N)

    ### gate dependency graph metrics

    @property
    def dag_qiskit(self):
        return circuit_to_dag(self.circuit)

    @cached_property
    def dag_graph(self) -> rx.PyDiGraph:
        return self._to_rustworkx(self.dag_qiskit)

    @cached_property
    def _gdg_metrics(self) -> dict:
        """cached helper to compute all GDG metrics in one pass as we need to traverse the GDG only once."""
        return self._compute_gdg_metrics()

    @property
    def gdg_critical_path_length(self) -> int:
        """
        Returns:
            The length of the critical path in the gate dependency graph (DAG).
            Defined as the number of GATES (Nodes) on the longest path.
            (Equivalent to Qiskit's `circuit.depth()`, NOT depth-1).
        """
        return self._gdg_metrics.get("critical_path_length", 0)

    @property
    def gdg_log_num_critical_paths(self) -> int:
        """
        Returns:
            The log number of critical paths in the gate dependency graph (DAG).
        """
        return self._gdg_metrics.get("log_num_critical_paths", 0)

    @property
    def gdg_log_total_paths(self) -> int:
        """
        Returns:
            The log number of total paths in the gate dependency graph (DAG).
        """
        return self._gdg_metrics.get("total_log_paths", 0)

    @property
    def gdg_mean_path_length(self) -> float:
        """
        Returns:
            The mean of all path lengths in the GDG.
        """
        mean = self._gdg_metrics.get("path_length_mean", 0.0)
        return mean

    @property
    def gdg_std_path_length(self) -> float:
        """
        Returns:
            The standard deviation of all path lengths in the GDG.
        """
        return self._gdg_metrics.get("path_length_std", 0.0)

    @property
    def gdg_percentage_gates_in_critical_path(self) -> float:
        """
        Returns:
            The percentage of gates that are on the critical path in the GDG.
            Formula: Critical Path Length (Nodes) / Total Gates
        """
        cp_len_nodes = self.gdg_critical_path_length
        total_gates = self.num_gates

        if total_gates == 0:
            return 0.0

        ratio = cp_len_nodes / total_gates

        return min(1.0, ratio)

    ### circuit density metrics

    @cached_property
    def op_qubit_volume(self) -> int:
        """
        Sum of the number of qubits involved in every gate.
        - 1q gate -> adds 1
        - 2q gate -> adds 2
        - Toffoli -> adds 3
        - MCX(n)  -> adds n
        """
        # Fast sum for the standard buckets
        vol_1q = self.num_1q_gates
        vol_2q = 2 * self.num_2q_gates

        # Iterate only over the complex gates (usually few)
        vol_gt2 = sum(len(inst.qubits) for inst in self.gt2_gates)

        return vol_1q + vol_2q + vol_gt2

    @property
    def density_score(self) -> float:
        r"""Generalized Density Score.
        Adapted from Bandic et al. 2025 for high-level gates.

        Formula:
        \[ \mathcal{D} = \frac{\frac{V_{total}}{d} - 1}{n_{qubits} - 1} \]
        """
        if self.num_qubits < 2 or self.depth == 0:
            return 0.0

        # Average "Width" (Active qubits per layer)
        avg_width = self.op_qubit_volume / self.depth

        numerator = avg_width - 1
        denominator = self.num_qubits - 1

        return numerator / denominator

    @property
    def idling_score(self) -> float:
        r"""Idling score as defined in Bandic et al. 2025:
        \[ \mathcal{I} = \frac{n_{qubits} \otimes d - \sum_{i=1}^{d} q_i}{n_{qubits} \otimes d}, \]
        where $q_i$ is the number of active qubits in moment $i$.

        Returns:
            Idling score as defined in Bandic et al. 2025.
        """
        area = self.num_qubits * self.depth
        if area == 0:
            return 0.0
        active_qubit_moments = self._compute_active_qubit_layers()
        # sum_qi is the total number of active qubit-moments
        idle_area = area - active_qubit_moments

        return idle_area / area

    @cached_property
    def commutation_score(self) -> float:
        """
        Fraction of overlapping operation pairs that commute.

        Overlapping = two operations share at least one qubit.
        Uses CommutationChecker pairwise (does NOT rely on CommutationAnalysis grouping,
        which is known to be non-uniformly typed and can over-group due to intransitivity).
        """
        if self.num_gates < 2:
            return 1.0

        checker = CommutationChecker()

        # Tune these for speed vs strictness.
        max_num_qubits = 3          # CommutationChecker defaults to 3-qubit checks 
        approximation_degree = 1.0  # treat as "commute" only if extremely close 
        window = None               # set to an int (e.g., 8) to only test near-neighbors on each wire

        # Normalize circuit.data across Qiskit versions
        ops = []  # list[(op, qargs, cargs)]
        ops_on_qubit = {q: [] for q in self.circuit.qubits}

        for idx, inst in enumerate(self.circuit.data):
            if hasattr(inst, "operation"):  # CircuitInstruction-style
                op = inst.operation
                qargs = list(inst.qubits)
                cargs = list(inst.clbits)
            else:  # legacy tuple style: (op, qargs, cargs)
                op, qargs, cargs = inst
                qargs = list(qargs)
                cargs = list(cargs)

            ops.append((op, qargs, cargs))
            for q in qargs:
                ops_on_qubit[q].append(idx)

        # Candidate pairs: all op-index pairs that share at least one qubit
        candidate_pairs = set()
        for idxs in ops_on_qubit.values():
            L = len(idxs)
            for i_pos, a in enumerate(idxs):
                j_stop = L if window is None else min(L, i_pos + window + 1)
                for j_pos in range(i_pos + 1, j_stop):
                    b = idxs[j_pos]
                    candidate_pairs.add((a, b) if a < b else (b, a))

        if not candidate_pairs:
            return 1.0

        commuting = 0
        tested = 0

        for a, b in candidate_pairs:
            op1, q1, c1 = ops[a]
            op2, q2, c2 = ops[b]

            # Avoid counting "skipped checks" as non-commuting:
            # CommutationChecker returns False both for "doesn't commute" and "skipped". 
            if len(q1) > max_num_qubits or len(q2) > max_num_qubits:
                continue
            if c1 or c2:
                continue
            if getattr(op1, "condition", None) is not None or getattr(op2, "condition", None) is not None:
                continue

            tested += 1
            if checker.commute(
                op1, q1, c1,
                op2, q2, c2,
                max_num_qubits=max_num_qubits,
                approximation_degree=approximation_degree,
            ):
                commuting += 1

        return 1.0 if tested == 0 else commuting / tested

    # other -> e.g. gate sampling
    @property
    def node_1q_counts(self) -> np.ndarray:
        return np.diag(self.ig_adj_mat)

    @property
    def edge_interaction_counts(self) -> dict[tuple[int, int], int]:
        return self._compute_edge_interaction_counts()

    @property
    def one_qubit_gates(self) -> list[tuple]:
        return self._gate_buckets["1q"]

    @property
    def two_qubit_gates(self) -> list[tuple]:
        return self._gate_buckets["2q"]

    @property
    def gt2_gates(self) -> list[tuple]:
        return self._gate_buckets["gt2"]

    # private helpers

    @cached_property
    def _gate_buckets(self) -> dict[str, list[tuple]]:
        """
        Scans the circuit and sorts gates into buckets of 1-qubit, 2-qubit, and >2-qubit gates.

        Returns:
          dict with keys: "1q", "2q", "gt2".
        """
        buckets = {"1q": [], "2q": [], "gt2": []}

        for instruction in self.circuit.data:
            # instruction is a CircuitInstruction(operation, qubits, clbits)
            n_q = len(instruction.qubits)

            # We store the instruction (or a tuple if you prefer lightweight)
            if n_q == 0:
                warning(f"instruction with no qubits assigned, ignoring: {instruction}")
                continue
            if n_q == 1:
                buckets["1q"].append(instruction)
            elif n_q == 2:
                buckets["2q"].append(instruction)
            elif n_q > 2:
                buckets["gt2"].append(instruction)

        return buckets

    @cached_property
    def _qubit_map(self) -> dict[Qubit, int]:
        return {q: i for i, q in enumerate(self.circuit.qubits)}

    @staticmethod
    def _filter_gates_by_name(qc: QuantumCircuit, drop_names: set[str]):
        pm = PassManager([FilterOpNodes(lambda node: node.op.name not in drop_names)])
        return pm.run(qc.copy())

    def _compute_adj_mat(self) -> np.ndarray:
        q_map = self._qubit_map
        adj_matrix = np.zeros((self.num_qubits, self.num_qubits), dtype=np.int64)

        for instruction in self.circuit.data:
            qubits = instruction.qubits
            n_q = len(qubits)

            # Global Phase, Barrier with no arguments, ...
            if n_q == 0:
                warning(f"instruction with no qubits assigned, ignoring: {instruction}")
                continue
            # Single Qubit Gate -> Diagonal
            if n_q == 1:
                idx = q_map[qubits[0]]
                adj_matrix[idx, idx] += 1
            else:
                if n_q > 2:
                    warning(
                        f"Instruction with more than 2 qubits found, counting all pairwise interactions: {instruction}"
                    )
                # all indices involved in the gate
                indices = [q_map[q] for q in qubits]

                # all pairs (u, v) where u != v
                # permutations returns (idx1, idx2), (idx2, idx1), etc.
                for u, v in permutations(indices, 2):
                    adj_matrix[u, v] += 1

        return adj_matrix

    @staticmethod
    def _to_rustworkx(dag: DAGCircuit) -> rx.PyDiGraph:
        """
        copys a Qiskit DAGCircuit into a rustworkx.PyDiGraph.

        - Node payloads: the original DAG nodes (DAGOpNode/DAGInNode/DAGOutNode)
        - Edge payloads: the wire object (Qubit/Clbit) for that dependency edge
        - Parallel edges: preserved (one per wire)
        """
        G = rx.PyDiGraph()
        index_of = {}

        # payload is the DAG node itself
        for node in dag.nodes():
            index_of[node] = G.add_node(node)

        # payload is the wire, multiedges OK
        for src, dst, wire in dag.edges():
            G.add_edge(index_of[src], index_of[dst], wire)
        return G

    def _compute_edge_interaction_counts(self) -> dict[tuple[int, int], int]:
        A = self.ig_adj_mat
        iu, ju = np.triu_indices_from(A, k=1)
        w = A[iu, ju]
        nonzero = w > 0
        return {
            (int(i), int(j)): int(w_ij)
            for i, j, w_ij in zip(iu[nonzero], ju[nonzero], w[nonzero])
        }

    def _compute_active_qubit_layers(self) -> int:
        """Helper: Counts total active qubit-moments using Qiskit's canonical layering."""
        total_active_moments = 0

        # dag.layers() yields a generator of layers
        for layer in self.dag_qiskit.layers():
            active_qubits = set()
            for node in layer["graph"].op_nodes():
                # node.qargs is a list of Qubit objects
                for q in node.qargs:
                    active_qubits.add(q)
            total_active_moments += len(active_qubits)

        return total_active_moments

    @staticmethod
    def _logsumexp(vals):
        if not vals:
            return float("-inf")
        a = max(vals)
        if a == float("-inf"):
            return float("-inf")
        return a + math.log(sum(math.exp(v - a) for v in vals))

    def _compute_gdg_per_node_metrics(self, H: rx.PyDiGraph):
        """Dynamic Programming to compute path statistics for every node.
        UPDATED: Calculates log_N (log of number of critical paths) to prevent overflow.
        """
        nodes_in_rev_topo_order = list(rx.topological_sort(H))[::-1]

        # L: Longest path (Depth) from node to sink
        # log_N: LOG of Number of critical paths starting at this node
        # log_n: LOG of Total number of paths starting at this node
        # m: Mean path length
        # v: Variance of path length
        L, log_N, log_n, m, v = {}, {}, {}, {}, {}

        for w in nodes_in_rev_topo_order:
            succ = H.successor_indices(w)

            # Base Case: Sink Node
            if not succ:
                L[w] = 1  # Depth 1
                log_N[w] = 0.0  # log(1) = 0 -> 1 critical path (self)
                log_n[w] = 0.0  # log(1) = 0 -> 1 total path
                m[w] = 1.0
                v[w] = 0.0
                continue

            # Recursive Step

            # Critical Path Length (L)
            # L[w] = 1 + max(L[successors])
            max_succ_L = max(L[u] for u in succ)
            L[w] = 1 + max_succ_L

            # Critical Path Count (log_N)
            # We only sum paths from successors that lie on the critical path (L[u] == max_succ_L)
            critical_succ_logs = [log_N[u] for u in succ if L[u] == max_succ_L]

            # log_N[w] = log( sum( exp(log_N[u]) ) )
            log_N[w] = self._logsumexp(critical_succ_logs)

            # Total Paths (log_n)
            # Total paths = sum(paths_through_u) -> logsumexp of all successors
            logZ = self._logsumexp([log_n[u] for u in succ])
            log_n[w] = logZ

            # Mean + Variance (Mixture of Successors)
            # probs[u] = exp(log_n[u] - logZ)
            probs = [math.exp(log_n[u] - logZ) for u in succ]

            # Mean: 1 + weighted average of successor means
            mean_succ = sum(p * m[u] for p, u in zip(probs, succ))
            m[w] = 1.0 + mean_succ

            # Variance: E[X^2] - E[X]^2
            sec_moment_succ = sum(p * (v[u] + m[u] ** 2) for p, u in zip(probs, succ))
            sec_moment_succ = max(
                0.0, sec_moment_succ
            )  # for potential float precision issues
            var_succ = max(0.0, sec_moment_succ - mean_succ**2)

            v[w] = var_succ

        return L, log_N, log_n, m, v

    def _compute_gdg_metrics(self) -> dict:
        """
        Calculates all GDG metrics.
        """
        # Extract Subgraph of Operations
        op_node_indices = [
            n
            for n in self.dag_graph.node_indices()
            if isinstance(self.dag_graph[n], DAGOpNode)
        ]
        H = self.dag_graph.subgraph(op_node_indices)

        if H.num_nodes() == 0:
            return {
                "critical_path_length": 0,
                "num_critical_paths": 0.0,  # Log(0) is -inf, but for features 0.0 is safer
                "path_length_mean": 0.0,
                "path_length_std": 0.0,
            }

        # Run Dynamic Programming
        L, log_N, log_n, m, v = self._compute_gdg_per_node_metrics(H)

        # Identify Sources
        sources = [n for n in H.node_indices() if H.in_degree(n) == 0]
        if not sources:
            return {
                "critical_path_length": 0,
                "num_critical_paths": 0.0,
                "path_length_mean": 0.0,
                "path_length_std": 0.0,
            }

        # Aggregation: Critical Path
        critical_path_length = max(L[s] for s in sources)

        # Sum logs of critical paths from valid sources
        # We need log( sum( exp(log_N[s]) ) ) for all s where L[s] == max
        critical_source_logs = [
            log_N[s] for s in sources if L[s] == critical_path_length
        ]

        # ln of the count
        log_num_critical_paths = self._logsumexp(critical_source_logs)

        # Aggregation: Mean + Var
        logZ = self._logsumexp([log_n[s] for s in sources])

        if logZ == float("-inf"):
            mean_len, std_len = 0.0, 0.0
            log_total = 0.0
        else:
            probs = [math.exp(log_n[s] - logZ) for s in sources]
            mean_len = sum(p * m[s] for p, s in zip(probs, sources))

            second_moment = sum(p * (v[s] + m[s] ** 2) for p, s in zip(probs, sources))
            var_len = max(0.0, second_moment - mean_len**2)
            std_len = math.sqrt(var_len)
            log_total = logZ

        return {
            "critical_path_length": critical_path_length,
            "log_num_critical_paths": log_num_critical_paths,
            "path_length_mean": mean_len,
            "path_length_std": std_len,
            "total_log_paths": log_total,
        }

    def plot_gdg(self, filename: str = None, **kwargs):
        """Plots the GDG / DAG of the circuit using Qiskit's built-in dag_drawer function.

        Args:
            filename (str, optional): The path to save the DAG plot image. If None, the plot is not saved to a file instead the PIL figure is returned. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the dag_drawer function.

        Returns:
            If filename is None, returns a PIL Image object of the DAG plot. Otherwise, saves the plot to the specified file and returns None.
        """

        fig = dag_drawer(self.dag_qiskit, filename=filename, **kwargs)

        if filename is None:
            return fig

    def plot_ig(
        self,
        filename: str = None,
        weighted: bool = True,
        **kwargs,
    ):
        """Plots the interaction graph (IG) using NetworkX and Pydot.
        Visuals matched to Qiskits dag_drawer default. Nodes are qubits, edges represent 2-qubit gates.
        Node labels include one-qubit gate counts if weighted=True.
        Edge labels show 2-qubit gate counts if weighted=True.
        """
        try:
            import networkx as nx
            from networkx.drawing.nx_pydot import to_pydot
        except ImportError as e:
            raise ImportError(
                "Optional dependency 'pydot' is not installed. Please install it using 'pip install qiskit-circuit-profiler[plotting]' or install it manually to use this feature."
            ) from e

        source_G = self.nx_ig if weighted else self.nx_unweighted_ig
        G = source_G.copy()

        layout_method = kwargs.pop("layout", "neato")

        G.graph["graph"] = {
            "layout": layout_method,
            "splines": "true",
            "overlap": "false",
            "nodesep": "0.8",
            "ranksep": "0.8",
            "forcelabels": "true",
        }

        G.graph["node"] = {
            "fontname": "Helvetica",
            "fontsize": "12",
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "green",
            "color": "black",
            "margin": "0.05",
            "fixedsize": "true",
            "width": "1.0",
            "height": "0.6",
        }

        G.graph["edge"] = {
            "fontname": "Helvetica",
            "fontsize": "10",
            "penwidth": "1.5",
            "fontcolor": "black",
            "len": "2.5",
        }

        for n, data in G.nodes(data=True):
            label = f"q[{n}]"

            n1q = data.get("n1q", 0)
            if weighted and n1q > 0:
                label += f"\n(1q: {n1q})"

            G.nodes[n]["label"] = label

        if weighted:
            for u, v, data in G.edges(data=True):
                weight = data.get("weight", 0)
                if weight > 0:
                    G.edges[u, v]["xlabel"] = str(int(weight))

        pydot_g = to_pydot(G)

        if filename:
            ext = filename.split(".")[-1].lower()
            if ext not in ["pdf", "png", "svg", "jpg", "jpeg"]:
                ext = "png"
                filename += ".png"

            pydot_g.write(filename, format=ext)
        else:
            try:
                from PIL import Image
                import io

                png_data = pydot_g.create_png()
                return Image.open(io.BytesIO(png_data))
            except ImportError:
                print(
                    "PIL is not installed. Please install it using 'pip install qiskit-circuit-profiler[plotting]' or install Pillow manually to return an image object."
                )
                return None

    def __str__(self):
        return f"Circuit(name={self.name}, n_qubits={self.num_qubits}, depth={self.depth}, size={self.num_gates})"
