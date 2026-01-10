from dataclasses import dataclass, field, asdict
import os
import gc
import multiprocessing
from qiskit.transpiler import CouplingMap
from pathlib import Path
import traceback

from typing import Callable, Dict, Iterable, List, Optional, Generator

try:
    import pandas as pd
    from func_timeout import func_timeout, FunctionTimedOut
    from mqt.bench import BenchmarkLevel
    from mqt.bench.benchmarks import get_benchmark_catalog
except ImportError as e:
    raise ImportError(
        """
        Required packages for dataset management are not installed. 
        Please install them using 'pip install qiskit-circuit-profiler[dataset]' 
        or install pandas, func-timeout, and mqt.bench manually.
        """
    ) from e

from .utils import bcolors, print_colored
from .features import to_flattened_feature_dict, to_overhead_feature_dict
from .circuit import CircuitWrapper

INDEX = ["source", "name", "num_qubits"]


@dataclass
class BenchmarkTask:
    """Represents a single job: A circuit to be generated/loaded and analyzed."""

    name: str
    n_qubits: int

    source: str  # 'mqt', 'file', 'qiskit'
    coupling_map: CouplingMap | None = None

    file_path: Optional[str] = None
    is_qasm3: bool = False

    filter_gates: list[str] | None = None
    basis_gates: list[str] | None = None
    optimization_level: int = 0

    params: dict = field(default_factory=dict)

    @property
    def description(self) -> str:
        if self.source == "file":
            return f"File: {self.name}"
        return f"{self.name} ({self.n_qubits}q)"


def _prep_qubit_range(
    name: str, n_vals: int = 8, min_qubits: int = 2, max_qubits: int = 130
) -> List[int]:
    """Prepare qubit ranges for different benchmark circuits."""
    assert n_vals > 0, "n_vals must be positive"
    assert min_qubits > 1, "min_qubits must be at least 2"
    assert (
        max_qubits >= min_qubits and max_qubits <= 130
    ), "max_qubits must be >= min_qubits and <= 130"

    RULES: Dict[str, Callable[[int], bool]] = {
        "shor": lambda n: n in {18, 42, 58, 74},
        "half_adder": lambda n: (n % 2 == 1) and (n >= 3),
        "full_adder": lambda n: (n % 2 == 0) and (n >= 4),
        "modular_adder": lambda n: (n % 2 == 0),
        "rg_qft_multiplier": lambda n: (n % 4 == 0) and (n >= 4),
        "cdkm_ripple_carry_adder": lambda n: (n % 2 == 0) and (n >= 4),
        "hrs_cumulative_multiplier": lambda n: ((n - 1) % 4 == 0) and (n >= 5),
        "draper_qft_adder": lambda n: (n % 2 == 0),
        "bmw_quark_copula": lambda n: (n % 2 == 0),
        "hhl": lambda n: (n >= 3),
        "qwalk": lambda n: (n >= 3),
        "graphstate": lambda n: (n >= 3),
        "vbe_ripple_carry_adder": lambda n: ((n - 1) % 3 == 0) and (n >= 4),
        "multiplier": lambda n: (n % 4 == 0) and (n <= 64),
    }

    candidates = [n for n in range(min_qubits, max_qubits + 1)]
    pred = RULES.get(name)
    if pred:
        candidates = [n for n in candidates if pred(n)]

    if not candidates:
        print(
            bcolors.WARNING
            + f"Warning: No valid qubit counts for circuit {name} in range [{min_qubits}, {max_qubits}]."
            + bcolors.ENDC
        )
        return []

    if len(candidates) > n_vals:
        idxs = {round(i * (len(candidates) - 1) / (n_vals - 1)) for i in range(n_vals)}
        candidates = [candidates[i] for i in sorted(idxs)]

    return candidates


def get_mqt_tasks(
    names: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    min_qubits: int = 2,
    max_qubits: int = 130,
    n_vals: int = 10,
    filter_gates: tuple[str] = ("measure", "reset", "barrier", "delay"),
    basis_gates: Optional[List[str]] = None,
    optimization_level: int = 0,
    coupling_map: Optional[CouplingMap] = None,
    **mqt_kwargs,
) -> Generator[BenchmarkTask, None, None]:
    """Yields tasks for MQTBench circuits without creating them yet."""
    if get_benchmark_catalog is None:
        raise ImportError(
            "MQTBench not installed. Install with 'pip install .[benchmarks]'"
        )

    catalog = list(get_benchmark_catalog().keys())

    # benchmark filtering
    if names:
        catalog = [n for n in catalog if n in names]
    if exclude:
        catalog = [n for n in catalog if n not in exclude]

    for name in catalog:
        # gets valid qubit counts for this specific algo
        qubit_counts = _prep_qubit_range(name, n_vals, min_qubits, max_qubits)

        for n in qubit_counts:
            effective = CircuitWrapper._predict_mqt_size(name, n)
            if effective > max_qubits:
                print(
                    bcolors.WARNING
                    + f"Warning: Circuit {name} with {n} qubits would have {effective} effective qubits, which is greater than {max_qubits}. Skipping. (You will get less qubits than expected.)"
                    + bcolors.ENDC
                )
                continue
            yield BenchmarkTask(
                name=name,
                n_qubits=n,
                filter_gates=filter_gates,
                basis_gates=basis_gates,
                optimization_level=optimization_level,
                source="mqt",
                coupling_map=coupling_map,
                params=mqt_kwargs,
            )


def _is_qasm3_file(path: Path) -> bool:
    if path.suffix.lower() == ".qasm3":
        return True
    # optional sniff:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(200)
        return "OPENQASM 3" in head
    except Exception:
        return False

def get_local_tasks(
    directory: str | Path,
    qasm3: bool = True,
    recursive: bool = True,
    parse_name: bool = False,
    filter_gates: tuple[str] = ("measure", "reset", "barrier", "delay"),
    basis_gates: Optional[List[str]] = None,
    optimization_level: int = 0,
) -> Generator[BenchmarkTask, None, None]:
    """Yields tasks for every .qasm/.qasm3 file in a directory."""
    path = Path(directory)
    pattern = "**/*.qasm*" if recursive else "*.qasm*"

    for fpath in path.glob(pattern):
        # name from filename (e.g. "qft_10.qasm" -> "qft")
        name = fpath.stem.split("_")[0] if parse_name else fpath.stem

        yield BenchmarkTask(
            name=name,
            n_qubits=0, # Unknown until loaded
            source="file",
            file_path=str(fpath),
            is_qasm3=_is_qasm3_file(fpath) if qasm3 else False,
            filter_gates=filter_gates,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            params={"path": str(fpath)},
        )


# worker + crash-safe function


def _worker_process(
    task: BenchmarkTask,
    metrics_func: Callable[[CircuitWrapper], Dict],
    overhead_metrics_func: Callable[[CircuitWrapper], Dict] | None,
    result_queue: multiprocessing.Queue,
    creation_timeout: int,
    collection_timeout: int,
):
    """Executes the full lifecycle: Create -> Measure -> Destroy."""
    try:
        # CREATION
        cw = None
        try:
            if task.source == "mqt":

                kwargs = task.params.copy()
                kwargs["filter_gates"] = task.filter_gates
                kwargs["basis_gates"] = task.basis_gates
                kwargs["opt_level"] = task.optimization_level

                cw = func_timeout(
                    creation_timeout,
                    CircuitWrapper.from_mqt_bench,
                    args=(task.name, task.n_qubits),
                    kwargs=kwargs,
                )
            elif task.source == "file":
               cw = func_timeout(
                    creation_timeout,
                    CircuitWrapper.from_qasm_file,
                    args=(task.params["path"],),
                    kwargs=dict(
                        qasm3_file=task.is_qasm3,
                        name=task.name,
                        filter_gates=task.filter_gates,
                        basis_gates=task.basis_gates,
                        opt_level=task.optimization_level,
                    ),
                )

        except FunctionTimedOut:
            result_queue.put(("ERROR", "Creation Timed Out"))
            return
        except Exception as e:
            result_queue.put(("ERROR", f"Creation Failed: {e}"))
            return

        if cw is None:
            result_queue.put(("ERROR", "Circuit resulted in None"))
            return

        #METRICS COLLECTION
        try:
            feats = func_timeout(collection_timeout, metrics_func, args=(cw,))
            
            # Overhead / Coupling Map logic
            if task.coupling_map is not None and overhead_metrics_func is not None:
                try:
                    # Apply connectivity
                    mapped_cw = cw.apply_connectivity(
                        connectivity=task.coupling_map,
                        basis_gates=task.basis_gates,
                        optimization_level=task.optimization_level or 0
                    )
                    # Compute overhead
                    overhead_feats = overhead_metrics_func(mapped_cw)
                    for k, v in overhead_feats.items():
                        feats[f"overhead_{k}"] = v
                except Exception as e:
                    raise RuntimeError(f"Mapping/Overhead calculation failed: {e}") from e

            # ensures index columns exist
            feats["source"] = task.source
            feats["name"] = task.name
            feats["num_qubits"] = cw.num_qubits
            feats["file_path"] = task.file_path

            feats["requested_qubits"] = task.n_qubits
            feats["filter_gates"] = task.filter_gates
            feats["basis_gates"] = task.basis_gates
            feats["optimization_level"] = task.optimization_level

            result_queue.put(("SUCCESS", feats))
        except FunctionTimedOut:
            result_queue.put(("ERROR", "Collection Timed Out"))
        except Exception as e:
            tb = traceback.format_exc()
            result_queue.put(("ERROR", f"Collection Failed: {e}\n{tb}"))

    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put(("ERROR", f"Worker Crash: {e}\n{tb}"))
    finally:
        #cleanup
        del cw
        gc.collect()


# consumer function


def generate_dataset(
    tasks: Iterable[BenchmarkTask],
    output_path: str = "./data/dataset.csv",
    metrics_func: Callable[[CircuitWrapper], Dict] = to_flattened_feature_dict,
    overhead_metrics_func: Callable[[CircuitWrapper], Dict] | None = to_overhead_feature_dict,
    timeout_create: int = 15 * 60,
    timeout_collect: int = 15 * 60,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Iterates through tasks, spawns isolated workers, and saves data incrementally.
    """
    results = []

    #ensures directory exists
    if save_csv and output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    print(f"Starting Benchmark Run -> {output_path}")

    for task in tasks:
        print(f"Processing: {task.description}...")

        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_worker_process,
            args=(task, metrics_func, overhead_metrics_func, queue, timeout_create, timeout_collect),
        )

        p.start()
        #waits for finish or hard timeout
        p.join(timeout=timeout_create + timeout_collect + 10)

        #handles zombies/hangs
        if p.is_alive():
            print_colored(
                f"  -> HARD TIMEOUT: Killing worker for {task.description}",
                bcolors.FAIL,
            )
            p.terminate()
            p.join()
            continue

        if p.exitcode != 0:
            print_colored(
                f"  -> CRASH: Worker died (Exit code {p.exitcode})", bcolors.FAIL
            )
            continue

        #retrieves result
        if not queue.empty():
            status, payload = queue.get()
            if status == "SUCCESS":
                results.append(payload)

                #incremental save
                if save_csv:
                    _append_to_csv(payload, output_path)

                print_colored(f"  -> Success.", bcolors.OKGREEN)
            else:
                print_colored(f"  -> {payload}", bcolors.WARNING)
        else:
            print_colored("  -> No result returned.", bcolors.WARNING)

    return pd.DataFrame(results)


def _append_to_csv(row_dict: dict, path: str):
    """Robust CSV appender that handles new columns and index merging."""
    new_df = pd.DataFrame([row_dict]).set_index(INDEX)

    if not os.path.exists(path):
        new_df.to_csv(path)
        return

    #loads existing to align columns
    existing_df = pd.read_csv(path, index_col=INDEX)
    # adds new columns/rows from new_df to existing_df
    # but doesnt update existing values
    combined = existing_df.combine_first(new_df)
    # updates existing values from new_df to existing_df
    combined.update(new_df)
    # also i dont want new columns to appear somewhere in the middle
    existing_cols = existing_df.columns.tolist()
    new_cols = [c for c in new_df.columns if c not in existing_cols]
    # so we reorder to ensure existing_cols stay at the left
    combined = combined[existing_cols + new_cols]
    # and we save it back
    combined.to_csv(path)
