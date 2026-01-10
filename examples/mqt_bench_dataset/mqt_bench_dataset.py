from qmetrics.dataset import get_mqt_tasks, generate_dataset
from qmetrics.features import (
    to_flattened_feature_dict,
    to_heavy_feature_dict,
    to_overhead_feature_dict,
    to_fast_feature_dict,
)
from qmetrics.utils import *
import networkx as nx
from qiskit.transpiler import CouplingMap

# --- Configuration ---
CREATION_TIMEOUT = 1800  # seconds
COLLECTION_TIMEOUT = 1800

HEAVY_ALGOS = [
    "multiplier",
    "hrs_cumulative_multiplier",
    "rg_qft_multiplier",
    "shor",
    "grover",
    "qwalk",
    "randomcircuit",
    "ae",
]

MIN_Q = 10
MAX_Q = 130
N_VALS = 10 

BASIS_GATES = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'cx', 'cp', 'cz', 'ecr', 'h',
                's', 'sdg', 't', 'tdg', 'u1', 'u2', 'u3', 'swap', 'iswap']

def get_2d_connectivity_cmap(num_qubits):
    G = nx.grid_2d_graph(int(np.ceil(np.sqrt(num_qubits))), int(np.ceil(np.sqrt(num_qubits))))
    A = nx.adjacency_matrix(G).toarray()

    adj_list = []
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] == 1:
                adj_list.append([i, j])
    
    return CouplingMap(adj_list)


def main():
    info(">>> Starting generation with NEW dataset API...")

    # ---------------------------------------------------------
    # 1. LIGHT ALGORITHMS (Easy Mode)
    # ---------------------------------------------------------
    # All algorithms EXCEPT heavy ones.
    # Full features (flattened).
    
    info("\n[1/3] Generating Light Algos (Full Features)...")
    
    static_cmap = get_2d_connectivity_cmap(MAX_Q)
    
    tasks_light = get_mqt_tasks(
        exclude=HEAVY_ALGOS,
        min_qubits=MIN_Q,
        max_qubits=MAX_Q,
        n_vals=N_VALS,
        basis_gates=BASIS_GATES,
        optimization_level=3,
        coupling_map=static_cmap # Overhead vs 2D grid
    )
    
    generate_dataset(
        tasks_light,
        output_path="data/features_indep_opt3.csv",
        metrics_func=to_flattened_feature_dict,
        overhead_metrics_func=to_overhead_feature_dict,
        timeout_create=CREATION_TIMEOUT,
        timeout_collect=COLLECTION_TIMEOUT
    )

    # ---------------------------------------------------------
    # 2. HEAVY ALGORITHMS (Fast/Light Features)
    # ---------------------------------------------------------
    # Heavy algos with "light" features (to avoid OOM/Timeout)
    
    info("\n[2/3] Generating Heavy Algos (Fast Features)...")
    
    tasks_heavy_fast = get_mqt_tasks(
        names=HEAVY_ALGOS,
        exclude=["randomcircuit"], # skipped in original
        min_qubits=MIN_Q,
        max_qubits=MAX_Q,
        n_vals=N_VALS,
        basis_gates=BASIS_GATES,
        optimization_level=3,
        coupling_map=static_cmap
    )
    
    generate_dataset(
        tasks_heavy_fast,
        output_path="data/features_indep_opt3_heavy_algos_fast.csv",
        metrics_func=to_fast_feature_dict,
        overhead_metrics_func=to_overhead_feature_dict,
        timeout_create=CREATION_TIMEOUT,
        timeout_collect=COLLECTION_TIMEOUT
    )
    
    # ---------------------------------------------------------
    # 3. HEAVY ALGORITHMS (Heavy Features) - Optional/Concurrent
    # ---------------------------------------------------------
    
    info("\n[3/3] Generating Heavy Algos (Heavy Features)...")
    
    tasks_heavy_full = get_mqt_tasks(
        names=HEAVY_ALGOS,
        exclude=["randomcircuit"],
        min_qubits=MIN_Q,
        max_qubits=MAX_Q,
        n_vals=N_VALS,
        basis_gates=BASIS_GATES,
        optimization_level=3,
        coupling_map=static_cmap
    )

    generate_dataset(
        tasks_heavy_full,
        output_path="data/features_indep_opt3_heavy_algos_full.csv",
        metrics_func=to_heavy_feature_dict,
        overhead_metrics_func=to_overhead_feature_dict,
        timeout_create=CREATION_TIMEOUT,
        timeout_collect=COLLECTION_TIMEOUT
    )

if __name__ == "__main__":
    main()
