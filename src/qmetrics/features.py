from .circuit import CircuitWrapper


def to_flattened_feature_dict(c: CircuitWrapper) -> dict[str, int | float]:
    feature_dict: dict[str, int | float] = {
        "num_qubits": c.num_qubits,
        "num_active_qubits": c.num_active_qubits,
        "num_gates": c.num_gates,
        "num_1q_gates": c.num_1q_gates,
        "num_2q_gates": c.num_2q_gates,
        "pct_2q_gates": c.pct_2q_gates,
        "num_3q_plus_gates": c.num_3q_plus_gates,
        "pct_3q_plus_gates": c.pct_3q_plus_gates,
        "num_mq_gates": c.num_multi_qubit_gates,
        "pct_mq_gates": c.pct_multi_qubit_gates,
        "depth": c.depth,
        "gate_density": c.gate_density,
        "ig_aspl": c.ig_aspl,
        "ig_std_adj_mat": c.ig_std_adj_mat,
        "ig_diameter": c.ig_diameter,
        "ig_max_betweenness": c.ig_max_betweenness,
        "ig_avg_degree": c.ig_avg_degree,
        "ig_max_degree": c.ig_max_degree,
        "ig_std_degree": c.ig_std_degree,
        "ig_avg_strength": c.ig_avg_strength,
        "ig_max_strength": c.ig_max_strength,
        "ig_std_strength": c.ig_std_strength,
        "ig_max_cliques_num": c.ig_max_cliques[0],
        "ig_max_cliques_size": c.ig_max_cliques[1],
        "ig_transitivity": c.ig_transitivity,
        "ig_avg_clustering_coef": c.ig_avg_clustering_coef,
        "ig_vertex_connectivity": c.ig_vertex_connectivity,
        "ig_edge_connectivity": c.ig_edge_connectivity,
        "ig_avg_coreness": c.ig_avg_coreness,
        "ig_min_coreness": c.ig_min_coreness,
        "ig_max_coreness": c.ig_max_coreness,
        "ig_std_coreness": c.ig_std_coreness,
        "ig_max_pagerank": c.ig_max_pagerank,
        "ig_min_pagerank": c.ig_min_pagerank,
        "ig_std_pagerank": c.ig_std_pagerank,
        "ig_normalized_hhi_pagerank": c.ig_normalized_hhi_pagerank,
        "gdg_critical_path_length": c.gdg_critical_path_length,
        "gdg_log_num_critical_paths": c.gdg_log_num_critical_paths,
        "gdg_log_total_paths": c.gdg_log_total_paths,
        "gdg_mean_path_length": c.gdg_mean_path_length,
        "gdg_std_path_length": c.gdg_std_path_length,
        "gdg_percentage_gates_in_critical_path": c.gdg_percentage_gates_in_critical_path,
        "op_qubit_volume": c.op_qubit_volume,
        "density_score": c.density_score,
        "idling_score": c.idling_score,
        "commutativity_score": c.commutation_score,
    }
    return feature_dict


def to_fast_feature_dict(c: CircuitWrapper) -> dict[str, int | float]:
    """
    Extracts ONLY computationally cheap features.

    INCLUDES:
      - Basic counts (gates, qubits)
      - Depth & Density
      - Interaction Graph (IG) metrics (Fast because N_qubits <= 130)

    EXCLUDES:
      - Gate Dependency Graph (GDG) metrics (Slow: depends on N_gates)
      - Commutativity/Idling scores (Slow: requires complex transpiler passes)
    """
    feature_dict: dict[str, int | float] = {
        "num_qubits": c.num_qubits,
        "num_active_qubits": c.num_active_qubits,
        "num_gates": c.num_gates,
        "num_1q_gates": c.num_1q_gates,
        "num_2q_gates": c.num_2q_gates,
        "pct_2q_gates": c.pct_2q_gates,
        "num_3q_plus_gates": c.num_3q_plus_gates,
        "pct_3q_plus_gates": c.pct_3q_plus_gates,
        "num_mq_gates": c.num_multi_qubit_gates,
        "pct_mq_gates": c.pct_multi_qubit_gates,
        "depth": c.depth,
        "gate_density": c.gate_density,
        "op_qubit_volume": c.op_qubit_volume,
        "density_score": c.density_score,
        "ig_aspl": c.ig_aspl,
        "ig_std_adj_mat": c.ig_std_adj_mat,
        "ig_diameter": c.ig_diameter,
        "ig_max_betweenness": c.ig_max_betweenness,
        "ig_avg_degree": c.ig_avg_degree,
        "ig_max_degree": c.ig_max_degree,
        "ig_std_degree": c.ig_std_degree,
        "ig_avg_strength": c.ig_avg_strength,
        "ig_max_strength": c.ig_max_strength,
        "ig_std_strength": c.ig_std_strength,
        "ig_max_cliques_num": c.ig_max_cliques[0],
        "ig_max_cliques_size": c.ig_max_cliques[1],
        "ig_transitivity": c.ig_transitivity,
        "ig_avg_clustering_coef": c.ig_avg_clustering_coef,
        "ig_vertex_connectivity": c.ig_vertex_connectivity,
        "ig_edge_connectivity": c.ig_edge_connectivity,
        "ig_avg_coreness": c.ig_avg_coreness,
        "ig_min_coreness": c.ig_min_coreness,
        "ig_max_coreness": c.ig_max_coreness,
        "ig_std_coreness": c.ig_std_coreness,
        "ig_max_pagerank": c.ig_max_pagerank,
        "ig_min_pagerank": c.ig_min_pagerank,
        "ig_std_pagerank": c.ig_std_pagerank,
        "ig_normalized_hhi_pagerank": c.ig_normalized_hhi_pagerank,
    }
    return feature_dict


def to_heavy_feature_dict(c: CircuitWrapper) -> dict[str, int | float]:
    """
    Extracts computationally heavy features.
    """
    feature_dict: dict[str, int | float] = {
        "num_qubits": c.num_qubits,
        "num_active_qubits": c.num_active_qubits,
        "gdg_critical_path_length": c.gdg_critical_path_length,
        "gdg_log_num_critical_paths": c.gdg_log_num_critical_paths,
        "gdg_log_total_paths": c.gdg_log_total_paths,
        "gdg_mean_path_length": c.gdg_mean_path_length,
        "gdg_std_path_length": c.gdg_std_path_length,
        "gdg_percentage_gates_in_critical_path": c.gdg_percentage_gates_in_critical_path,
        "op_qubit_volume": c.op_qubit_volume,
        "density_score": c.density_score,
        "idling_score": c.idling_score,
        "commutativity_score": c.commutation_score,
    }
    return feature_dict


def to_overhead_feature_dict(c: CircuitWrapper) -> dict:
    """
    For MAPPED (Target Y) -> The 'Cost' Metrics.
    These are the physical resources consumed on the device.
    """
    return {
        "num_qubits": c.num_qubits,
        "num_active_qubits": c.num_active_qubits,
        "mapped_2q_gates": c.num_2q_gates,
        "mapped_gates": c.num_gates,
        "mapped_depth": c.depth,
        "mapped_vol": c.op_qubit_volume,
    }
