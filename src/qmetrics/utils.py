from datetime import datetime
import numpy as np


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def log_to_file(msg):
    """Writes messages to a log file instead of the console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("circuit.log", "a") as f:
        f.write(f"[{timestamp}] WARNING: {msg}\n")


def warning(message: str) -> None:
    log_to_file(message)
    print(f"{bcolors.WARNING}WARNING: {message}{bcolors.ENDC}")


def error(message: str) -> None:
    log_to_file(message)
    print(f"{bcolors.FAIL}ERROR: {message}{bcolors.ENDC}")


def info(message: str) -> None:
    print(f"{bcolors.OKGREEN}INFO: {message}{bcolors.ENDC}")


def debug(message: str) -> None:
    print(f"{bcolors.OKBLUE}DEBUG: {message}{bcolors.ENDC}")


def print_colored(message: str, color: bcolors) -> None:
    print(f"{color}{message}{bcolors.ENDC}")


def hist_dict_to_array(d: dict[int, int], max_bin: int = None) -> np.ndarray:

    if max_bin is None:
        max_bin = max(d.keys(), default=0)

    arr = np.zeros(max_bin + 1, dtype=int)
    for k, v in d.items():
        if 0 <= k <= max_bin:
            arr[k] = v
    return arr
