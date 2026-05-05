import sinter
import pymatching
import numpy as np
from typing import List, Callable, Tuple
from collections import Counter, defaultdict
import argparse
import cirq
import stimcirq
import json, hashlib
import time
import datetime
import os
import math

from full_clifford_sim.main_complied_fxns import full_circuit, generate_sv_cirquit
from full_clifford_sim.gap_sampler import *
from full_clifford_sim.cirq_utilities import *

## take DEM from ref_circuit, and decode shots from sample circutis


USE_CUSTOM_CCZ = True
CCZ_LABEL_SHORT_RANGE = "rydberg_ccz_pulse2_short_range"
CCZ_LABEL_SHORT_RANGE_20260504A = "rydberg_ccz_pulse2_short_range_20260504a"
CCZ_LABEL_SHORT_RANGE_20260504B = "rydberg_ccz_pulse2_short_range_20260504b"
CCZ_LABEL_LONG_RANGE = "rydberg_ccz_pulse2_long_range"
CCZ_LABEL_LONG_RANGE_20260504A = "rydberg_ccz_pulse2_long_range_20260504a"
CCZ_LABEL_LONG_RANGE_20260504B = "rydberg_ccz_pulse2_long_range_20260504b"

# Paste the 8x8 short-range and long-range CCZ-like matrices here.
# The three local CCZ placements use CCZ_gate_short_range.
# The corner-to-corner CCZ placement uses CCZ_gate_long_range.
#
# Leave either matrix as None to use the ideal Cirq CCZ at that location.
CCZ_gate_short_range = np.array(
    [
        [-0.978934 - 0.204178j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, -0.978983 - 0.203944j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, -0.978977 - 0.20397j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.978787 - 0.204875j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.978964 - 0.204035j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.978773 - 0.20494j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.978768 - 0.204966j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.979054 + 0.203443j],
    ],
    dtype=np.complex128,
)
CCZ_gate_short_range_20260504A = np.array(
    [
        [-0.983593 + 0.180403j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, -0.982398 + 0.186801j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, -0.984153 + 0.177323j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.984242 + 0.176825j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.984159 + 0.177286j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.984248 + 0.176788j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.982457 + 0.18647j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.983475 - 0.181011j],
    ],
    dtype=np.complex128,
)
CCZ_gate_short_range_20260504B = np.array(
    [
        [-0.983593 + 0.180403j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, -0.982398 + 0.186801j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, -0.984153 + 0.177323j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.984242 + 0.176825j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.984159 + 0.177286j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.984248 + 0.176788j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.982457 + 0.18647j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.983475 - 0.181011j],
    ],
    dtype=np.complex128,
)
CCZ_gate_long_range = np.array(
    [
        [0.764706 + 0.64438j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.974864 + 0.2228j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.903458 + 0.428677j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.85623 + 0.49756j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.903406 + 0.428786j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.85617 + 0.497663j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.952194 + 0.304366j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.805784 - 0.572354j],
    ],
    dtype=np.complex128,
)
CCZ_gate_long_range_20260504A = np.array(
    [
        [0.513761 + 0.857934j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.638146 + 0.769911j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.452879 + 0.891568j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.456939 + 0.889325j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.452825 + 0.891596j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.456885 + 0.889353j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.641322 + 0.766525j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.509414 - 0.859307j],
    ],
    dtype=np.complex128,
)
CCZ_gate_long_range_20260504B = np.array(
    [
        [0.55068 + 0.834716j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.655927 + 0.754744j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.472811 + 0.881095j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.492863 + 0.868891j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.472825 + 0.881088j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.492876 + 0.868883j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.673334 + 0.738492j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.529569 - 0.845028j],
    ],
    dtype=np.complex128,
)

# Supported values:
# - "unitary": require CUSTOM_CCZ_MATRIX to be unitary and use a coherent gate.
# - "contractive_kraus": treat CUSTOM_CCZ_MATRIX as a 3-qubit Kraus operator K
#   and complete it into a channel with sqrt(I - K^\dagger K).
# - "auto": use "unitary" if possible, otherwise try "contractive_kraus".
#
# This trajectory runner defaults to the non-unitary channel interpretation.
# Each call to cirq.Simulator.simulate(...) will then sample a single Kraus
# branch for each custom CCZ application, which is the Monte Carlo wavefunction
# / quantum trajectory approximation to the same channel that a density-matrix
# simulation would average exactly.
#
# Caveat: only the Cirq exact-sim prefix uses the custom CCZ channel directly.
# The later Stim/sinter resampling stage still operates on stabilizer circuits
# keyed by the exact-sim outputs; Stim itself is not carrying the non-unitary
# CCZ channel.
CUSTOM_CCZ_MODE = "contractive_kraus"
CUSTOM_CCZ_CONTRACTIVE_ATOL = 1e-6
CUSTOM_CCZ_NEAR_UNITARY_ATOL = 1e-4
DEFAULT_SIM_METHOD = "trajectory"
DEFAULT_MOVEMENT_MODEL = "none"
DEFAULT_ZONE_SEPARATION_UM = 40.0
DEFAULT_TRANSPORT_BENCHMARK_DISTANCE_UM = 610.0
DEFAULT_TRANSPORT_BENCHMARK_TIME_US = 1600.0
DEFAULT_TRANSFER_ONE_WAY_US = 400.0
DEFAULT_PREVIOUS_ROUND_CYCLE_US = 4450.0
DEFAULT_PREVIOUS_ROUND_MOVEMENT_US = 2570.0
DEFAULT_MOVE_DEPHASE_RATE_PER_US = -math.log(0.99953) / DEFAULT_TRANSPORT_BENCHMARK_TIME_US
DEFAULT_MOVE_LOSS_RATE_PER_US = -math.log(0.998) / DEFAULT_PREVIOUS_ROUND_MOVEMENT_US


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def current_ccz_label(raw_matrix, custom_label: str) -> str:
    if USE_CUSTOM_CCZ and raw_matrix is not None:
        return custom_label
    return "ideal"


def resolve_short_range_preset(preset: str) -> tuple[np.ndarray | None, str]:
    preset = preset.lower()
    if preset == "default":
        return CCZ_gate_short_range, CCZ_LABEL_SHORT_RANGE
    if preset == "20260504a":
        return CCZ_gate_short_range_20260504A, CCZ_LABEL_SHORT_RANGE_20260504A
    if preset == "20260504b":
        return CCZ_gate_short_range_20260504B, CCZ_LABEL_SHORT_RANGE_20260504B
    raise ValueError(f"Unsupported short_range_preset={preset!r}.")


def resolve_long_range_preset(preset: str) -> tuple[np.ndarray | None, str]:
    preset = preset.lower()
    if preset == "default":
        return CCZ_gate_long_range, CCZ_LABEL_LONG_RANGE
    if preset == "20260504a":
        return CCZ_gate_long_range_20260504A, CCZ_LABEL_LONG_RANGE_20260504A
    if preset == "20260504b":
        return CCZ_gate_long_range_20260504B, CCZ_LABEL_LONG_RANGE_20260504B
    if preset == "same_as_short":
        return None, "same_as_short"
    raise ValueError(f"Unsupported long_range_preset={preset!r}.")


def _as_8x8_complex_matrix(raw_matrix) -> np.ndarray | None:
    if raw_matrix is None:
        return None
    matrix = np.asarray(raw_matrix, dtype=np.complex128)
    if matrix.shape != (8, 8):
        raise ValueError(f"Expected an 8x8 custom CCZ matrix, got {matrix.shape}.")
    return matrix


def _is_unitary(matrix: np.ndarray, atol: float = 1e-8) -> bool:
    return np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=atol)


def _nearest_unitary(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix)
    return u @ vh


def _completed_kraus_ops_from_matrix(matrix: np.ndarray, atol: float = CUSTOM_CCZ_CONTRACTIVE_ATOL) -> tuple[np.ndarray, np.ndarray]:
    residual = np.eye(matrix.shape[0], dtype=np.complex128) - matrix.conj().T @ matrix
    residual = 0.5 * (residual + residual.conj().T)
    evals, evecs = np.linalg.eigh(residual)
    if np.min(evals) < -atol:
        raise ValueError(
            "CUSTOM_CCZ_MATRIX is not unitary and also cannot be treated as a single "
            "Kraus operator because I - K^dagger K is not positive semidefinite."
        )
    evals = np.clip(evals, 0.0, None)
    completion = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
    return matrix, completion


class ContractiveCustomCCZGate(cirq.Gate):
    def __init__(self, matrix: np.ndarray):
        self.matrix = _as_8x8_complex_matrix(matrix)
        self._kraus_ops = _completed_kraus_ops_from_matrix(self.matrix)

    def _num_qubits_(self) -> int:
        return 3

    def _kraus_(self):
        return self._kraus_ops

    def _circuit_diagram_info_(self, args):
        return ("CCZ*", "CCZ*", "CCZ*")


def build_ccz_factory(
    raw_matrix,
    ccz_mode: str,
    near_unitary_atol: float = CUSTOM_CCZ_NEAR_UNITARY_ATOL,
) -> tuple[Callable[[cirq.Qid, cirq.Qid, cirq.Qid], cirq.Operation], str]:
    if not USE_CUSTOM_CCZ or raw_matrix is None:
        return lambda q0, q1, q2: cirq.CCZ(q0, q1, q2), "ideal_unitary"

    matrix = _as_8x8_complex_matrix(raw_matrix)

    mode = ccz_mode.lower()
    if mode not in {"auto", "unitary", "contractive_kraus"}:
        raise ValueError(f"Unsupported ccz_mode={ccz_mode!r}.")

    if mode in {"auto", "unitary"}:
        if _is_unitary(matrix):
            gate = cirq.MatrixGate(matrix)
            return lambda q0, q1, q2: gate.on(q0, q1, q2), "exact_unitary"

        # Option 1: treat the supplied gate as a coherent near-unitary pulse
        # and project it onto the closest exact unitary.
        if _is_unitary(matrix, atol=near_unitary_atol):
            projected = _nearest_unitary(matrix)
            gate = cirq.MatrixGate(projected)
            return lambda q0, q1, q2: gate.on(q0, q1, q2), "projected_unitary"

    if mode == "unitary":
        raise ValueError(
            "ccz_mode='unitary' requires a unitary or near-unitary 8x8 matrix. "
            "For leakage-like behavior, switch to 'contractive_kraus' or 'auto'."
        )

    # Non-unitary path: represent the supplied 8x8 matrix as a Kraus operator
    # and complete it into a trace-preserving channel. The caller decides
    # whether to average this channel with DensityMatrixSimulator or sample
    # trajectories with cirq.Simulator.
    gate = ContractiveCustomCCZGate(matrix)
    return lambda q0, q1, q2: gate.on(q0, q1, q2), "kraus"


def build_simulator(sim_method: str, gate_model: str, seed: int) -> tuple[cirq.SimulatesFinalState, bool]:
    sim_method = sim_method.lower()
    if sim_method not in {"trajectory", "density_matrix"}:
        raise ValueError(f"Unsupported sim_method={sim_method!r}.")

    if sim_method == "density_matrix":
        return cirq.DensityMatrixSimulator(seed=seed), True

    # trajectory mode always uses the pure-state simulator. For unitary gates
    # this is exact; for Kraus gates it is the Monte Carlo unraveling.
    return cirq.Simulator(seed=seed), False


def _get_final_density_matrix(sim_result) -> np.ndarray:
    if hasattr(sim_result, "final_density_matrix"):
        return np.asarray(sim_result.final_density_matrix, dtype=np.complex128)

    density_matrix_method = getattr(sim_result, "density_matrix", None)
    if callable(density_matrix_method):
        try:
            return np.asarray(density_matrix_method(copy=False), dtype=np.complex128)
        except TypeError:
            return np.asarray(density_matrix_method(), dtype=np.complex128)

    raise TypeError("Could not extract a final density matrix from the Cirq simulation result.")


def _single_qubit_density_matrix(full_density_matrix: np.ndarray, qubit_index: int, num_qubits: int) -> np.ndarray:
    dm_tensor = full_density_matrix.reshape((2,) * (2 * num_qubits))
    current_qubits = num_qubits
    current_index = qubit_index

    for axis in range(num_qubits - 1, -1, -1):
        if axis == qubit_index:
            continue
        dm_tensor = np.trace(dm_tensor, axis1=axis, axis2=axis + current_qubits)
        if axis < current_index:
            current_index -= 1
        current_qubits -= 1

    return dm_tensor.reshape((2, 2))


def _bloch_vector_from_density_matrix(full_density_matrix: np.ndarray, qubit_index: int, num_qubits: int) -> np.ndarray:
    rho = _single_qubit_density_matrix(full_density_matrix, qubit_index, num_qubits)
    return np.array(
        [
            2 * np.real(rho[0, 1]),
            -2 * np.imag(rho[0, 1]),
            np.real(rho[0, 0] - rho[1, 1]),
        ],
        dtype=np.float64,
    )


def logical_bloch_vector(sim_result, logical_qubit: cirq.Qid, qubit_order: tuple[cirq.Qid, ...], use_density_sim: bool) -> np.ndarray:
    if not use_density_sim:
        return np.asarray(sim_result.bloch_vector_of(qubit=logical_qubit), dtype=np.float64)

    logical_index = qubit_order.index(logical_qubit)
    final_density_matrix = _get_final_density_matrix(sim_result)
    return _bloch_vector_from_density_matrix(final_density_matrix, logical_index, len(qubit_order))


def _phase_flip_probability(rate_per_us: float, duration_us: float) -> float:
    if rate_per_us <= 0 or duration_us <= 0:
        return 0.0
    return 0.5 * (1.0 - math.exp(-rate_per_us * duration_us))


def _event_probability(rate_per_us: float, duration_us: float) -> float:
    if rate_per_us <= 0 or duration_us <= 0:
        return 0.0
    return 1.0 - math.exp(-rate_per_us * duration_us)


def build_movement_profile(
    movement_model: str,
    moved_qubits: tuple[cirq.Qid, ...],
    zone_separation_um: float,
    transport_benchmark_distance_um: float,
    transport_benchmark_time_us: float,
    transfer_one_way_us: float,
    previous_round_cycle_us: float,
    previous_round_movement_us: float,
    dephase_rate_per_us: float,
    loss_rate_per_us: float,
) -> dict | None:
    if movement_model == "none":
        return None
    if movement_model != "parked_all_qubits":
        raise ValueError(f"Unsupported movement_model={movement_model!r}.")

    zone_cross_us = transport_benchmark_time_us * (
        zone_separation_um / transport_benchmark_distance_um
    )
    move_in_us = transfer_one_way_us + zone_cross_us + transfer_one_way_us
    move_out_us = move_in_us
    per_window_dephase_p = _phase_flip_probability(dephase_rate_per_us, move_in_us)
    per_window_loss_p = _event_probability(loss_rate_per_us, move_in_us)
    parked_round_ops_us = previous_round_cycle_us - previous_round_movement_us
    parked_check_us = 2.0 * parked_round_ops_us
    previous_check_cycle_us = 2.0 * previous_round_cycle_us
    movement_cycle_us = parked_check_us + move_in_us + move_out_us

    return {
        "model": movement_model,
        "num_qubits": len(moved_qubits),
        "zone_separation_um": zone_separation_um,
        "transport_benchmark_distance_um": transport_benchmark_distance_um,
        "transport_benchmark_time_us": transport_benchmark_time_us,
        "transfer_one_way_us": transfer_one_way_us,
        "zone_cross_us": zone_cross_us,
        "move_in_us": move_in_us,
        "move_out_us": move_out_us,
        "per_window_dephase_p": per_window_dephase_p,
        "per_window_loss_p": per_window_loss_p,
        "dephase_rate_per_us": dephase_rate_per_us,
        "loss_rate_per_us": loss_rate_per_us,
        "parked_round_ops_us": parked_round_ops_us,
        "parked_check_us": parked_check_us,
        "previous_round_cycle_us": previous_round_cycle_us,
        "previous_round_movement_us": previous_round_movement_us,
        "previous_check_cycle_us": previous_check_cycle_us,
        "movement_cycle_us": movement_cycle_us,
        "movement_overhead_us": move_in_us + move_out_us,
        "cycle_ratio_vs_previous": movement_cycle_us / previous_check_cycle_us,
    }


def add_movement_dephasing(
    circuit: cirq.Circuit,
    moved_qubits: tuple[cirq.Qid, ...],
    per_window_dephase_p: float,
) -> cirq.Circuit:
    if per_window_dephase_p <= 0 or not moved_qubits:
        return circuit

    wrapped = cirq.Circuit()
    wrapped.append(
        cirq.Moment(
            cirq.phase_flip(per_window_dephase_p).on(qubit) for qubit in moved_qubits
        )
    )
    wrapped += circuit
    wrapped.append(
        cirq.Moment(
            cirq.phase_flip(per_window_dephase_p).on(qubit) for qubit in moved_qubits
        )
    )
    return wrapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one handoff shard for the HXY workflow with a trajectory-simulated custom CCZ gate."
    )
    parser.add_argument("p_milli", type=int, help="Physical error rate in units of 1e-3.")
    parser.add_argument("tasknum", type=int, help="Shard / seed identifier.")
    parser.add_argument("--dfinal", type=int, default=7, help="Final code distance parameter.")
    parser.add_argument(
        "--prep",
        choices=["unitstab", "hookinj", "optunit"],
        default="unitstab",
        help="Preparation strategy.",
    )
    parser.add_argument("--ghz-size", type=int, default=3, help="GHZ ancilla size.")
    parser.add_argument(
        "--latter-rounds",
        type=int,
        default=3,
        help="Number of latter stabilizer rounds recorded in metadata.",
    )
    parser.add_argument(
        "--basis",
        default="HXY",
        help="Logical basis label. The exact-simulation flow expects HXY.",
    )
    parser.add_argument(
        "--num-shots-per-sverr",
        type=int,
        default=int(os.environ.get("NUM_SHOTS_PER_SVERR", "20")),
        help="Number of Clifford shots sampled per statevector outcome.",
    )
    parser.add_argument(
        "--num-sverr-shots",
        type=int,
        default=int(os.environ.get("NUM_SVERR_SHOTS", "10")),
        help="Number of statevector samples taken before Clifford resampling.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag used in metadata and output directory names. Defaults to MMDD.",
    )
    parser.add_argument(
        "--fault-distance-tag",
        type=int,
        default=None,
        help="Optional f-tag to add into the output metadata.",
    )
    parser.add_argument(
        "--sim-method",
        choices=["trajectory", "density_matrix"],
        default=os.environ.get("CCZ_SIM_METHOD", DEFAULT_SIM_METHOD),
        help="How to handle non-unitary CCZs in the exact-sim prefix.",
    )
    parser.add_argument(
        "--ccz-mode",
        choices=["auto", "unitary", "contractive_kraus"],
        default=os.environ.get("CCZ_MODE", CUSTOM_CCZ_MODE),
        help="How to interpret CUSTOM_CCZ_MATRIX when building the custom gate.",
    )
    parser.add_argument(
        "--long-range-ccz-mode",
        choices=["auto", "unitary", "contractive_kraus"],
        default=os.environ.get("LONG_RANGE_CCZ_MODE", None),
        help="Optional override for the long-range CCZ interpretation mode only.",
    )
    parser.add_argument(
        "--short-range-preset",
        choices=["default", "20260504a", "20260504b"],
        default=os.environ.get("SHORT_RANGE_PRESET", "default"),
        help="Named short-range CCZ matrix preset to use for the three local placements.",
    )
    parser.add_argument(
        "--ccz-near-unitary-atol",
        type=float,
        default=float(os.environ.get("CCZ_NEAR_UNITARY_ATOL", str(CUSTOM_CCZ_NEAR_UNITARY_ATOL))),
        help="Tolerance used when deciding whether to project a near-unitary custom CCZ matrix onto the closest exact unitary.",
    )
    parser.add_argument(
        "--ideal-ccz-locations",
        choices=["none", "short", "long", "all"],
        default=os.environ.get("IDEAL_CCZ_LOCATIONS", "none"),
        help="Replace selected custom CCZ placements with ideal Cirq CCZ gates inside the trajectory workflow.",
    )
    parser.add_argument(
        "--long-range-preset",
        choices=["default", "20260504a", "20260504b", "same_as_short"],
        default=os.environ.get("LONG_RANGE_PRESET", "default"),
        help="Named long-range CCZ matrix preset to use for the corner-to-corner placement.",
    )
    parser.add_argument(
        "--movement-model",
        choices=["none", "parked_all_qubits"],
        default=os.environ.get("MOVEMENT_MODEL", DEFAULT_MOVEMENT_MODEL),
        help="Simple zone-movement model. 'parked_all_qubits' moves all qubits into the entangling zone once, keeps them parked through the full check block, and moves them back out.",
    )
    parser.add_argument(
        "--zone-separation-um",
        type=float,
        default=float(os.environ.get("ZONE_SEPARATION_UM", str(DEFAULT_ZONE_SEPARATION_UM))),
        help="Approximate storage-to-entangling zone separation in microns.",
    )
    parser.add_argument(
        "--transport-benchmark-distance-um",
        type=float,
        default=float(
            os.environ.get(
                "TRANSPORT_BENCHMARK_DISTANCE_UM",
                str(DEFAULT_TRANSPORT_BENCHMARK_DISTANCE_UM),
            )
        ),
        help="Reference move distance used to infer short-zone-cross timing.",
    )
    parser.add_argument(
        "--transport-benchmark-time-us",
        type=float,
        default=float(
            os.environ.get(
                "TRANSPORT_BENCHMARK_TIME_US",
                str(DEFAULT_TRANSPORT_BENCHMARK_TIME_US),
            )
        ),
        help="Reference move time corresponding to --transport-benchmark-distance-um.",
    )
    parser.add_argument(
        "--transfer-one-way-us",
        type=float,
        default=float(os.environ.get("TRANSFER_ONE_WAY_US", str(DEFAULT_TRANSFER_ONE_WAY_US))),
        help="One-way coherent SLM<->AOD transfer time in microseconds.",
    )
    parser.add_argument(
        "--previous-round-cycle-us",
        type=float,
        default=float(
            os.environ.get(
                "PREVIOUS_ROUND_CYCLE_US",
                str(DEFAULT_PREVIOUS_ROUND_CYCLE_US),
            )
        ),
        help="Reference per-round cycle time for the previous ancilla-swap architecture.",
    )
    parser.add_argument(
        "--previous-round-movement-us",
        type=float,
        default=float(
            os.environ.get(
                "PREVIOUS_ROUND_MOVEMENT_US",
                str(DEFAULT_PREVIOUS_ROUND_MOVEMENT_US),
            )
        ),
        help="Movement-only portion of the reference per-round cycle time.",
    )
    parser.add_argument(
        "--movement-dephase-rate-per-us",
        type=float,
        default=float(
            os.environ.get(
                "MOVEMENT_DEPHASE_RATE_PER_US",
                str(DEFAULT_MOVE_DEPHASE_RATE_PER_US),
            )
        ),
        help="Effective dephasing rate per microsecond during movement windows.",
    )
    parser.add_argument(
        "--movement-loss-rate-per-us",
        type=float,
        default=float(
            os.environ.get(
                "MOVEMENT_LOSS_RATE_PER_US",
                str(DEFAULT_MOVE_LOSS_RATE_PER_US),
            )
        ),
        help="Effective heralded-loss rate per microsecond during movement windows.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tasknum = args.tasknum
    rng = np.random.default_rng(tasknum)
    
    num_shots_per_sverr = args.num_shots_per_sverr
    num_sverr_shots = args.num_sverr_shots

    p = 0.001 * args.p_milli
    errmodel = 'uniform'
    basis = args.basis
    if basis == 'HXY':
        true_bloch_rep = [1/np.sqrt(2),1/np.sqrt(2),0]
        magic = True
    else:
        magic = False
    ghz_size = args.ghz_size
    errD = False
    cX =  True
    d2 = args.dfinal
    prep = args.prep
    latter_rounds = args.latter_rounds
    sim_method = args.sim_method.lower()
    sim_method_slug = _slugify(sim_method)
    ccz_mode = args.ccz_mode.lower()
    long_range_ccz_mode = args.long_range_ccz_mode.lower() if args.long_range_ccz_mode is not None else ccz_mode
    ideal_ccz_locations = args.ideal_ccz_locations.lower()
    short_range_preset = args.short_range_preset.lower()
    long_range_preset = args.long_range_preset.lower()
    movement_model = args.movement_model.lower()
    movement_slug = _slugify(movement_model)
    resolved_short_range_matrix, resolved_short_range_label = resolve_short_range_preset(short_range_preset)
    short_range_matrix = None if ideal_ccz_locations in {"short", "all"} else resolved_short_range_matrix
    resolved_long_range_matrix, resolved_long_range_label = resolve_long_range_preset(long_range_preset)
    if long_range_preset == "same_as_short":
        resolved_long_range_matrix = resolved_short_range_matrix
        resolved_long_range_label = f"{resolved_short_range_label}_reused_for_long_range"
    long_range_matrix = None if ideal_ccz_locations in {"long", "all"} else resolved_long_range_matrix
    short_range_ccz_label = current_ccz_label(short_range_matrix, resolved_short_range_label)
    long_range_ccz_label = current_ccz_label(long_range_matrix, resolved_long_range_label)
    ccz_label_slug = _slugify(
        f"sr-{short_range_ccz_label}__lr-{long_range_ccz_label}"
    )

    if prep == "unitstab":
        endofinj = 8
    elif prep == "hookinj":
        endofinj = 12
    elif prep == "optunit":
        endofinj = 2
    else:
        raise ValueError("injection stratey not well defined")

    x = datetime.datetime.now()
    tag = args.tag or x.strftime("%m%d")
    movement_path_tag = "" if movement_model == "none" else f"_{movement_slug}"
    path = f'./realcirqhandoffmid_{ccz_label_slug}_{sim_method_slug}{movement_path_tag}_p{int(1000*p)}_outs{tag}/'
        
    #create a reference circuit used for matching
    ref_circuit = full_circuit(nm=p, 
                         dfinal=d2,
                         ps_on_d3=1,
                         prep='',
                         neutralatom=False,
                         handoff=True,
                         ref=True,
                         )
    ref_dem = ref_circuit.detector_error_model()
    matcher = pymatching.Matching.from_detector_error_model(ref_dem)

    #generate a truncated cirq circuit from stim circuit
    num_errors = 0
    cc = generate_sv_cirquit(nm=p, dfinal=d2, 
                            prep=prep,
                            cirq=True, 
                            skip_gauge_fix=True,
                            add_check_noise=not magic)
    criq_rep = stimcirq.stim_circuit_to_cirq_circuit(remove_QCs(cc,p, reverse_Y=False))
    short_range_ccz_factory, short_range_gate_model = build_ccz_factory(
        raw_matrix=short_range_matrix,
        ccz_mode=ccz_mode,
        near_unitary_atol=args.ccz_near_unitary_atol,
    )
    long_range_ccz_factory, long_range_gate_model = build_ccz_factory(
        raw_matrix=long_range_matrix,
        ccz_mode=long_range_ccz_mode,
        near_unitary_atol=args.ccz_near_unitary_atol,
    )
    effective_ccz_mode = (
        "ideal"
        if short_range_gate_model == "ideal_unitary" and long_range_gate_model == "ideal_unitary"
        else ccz_mode
    )

    if magic: #break down circuit 

        #### process Clifford circuit into nonClifford Circ
        edited_cirq_rep = cirq.Circuit()
        sevendiaggate = cirq.MatrixGate(np.array([[0,1],[1j,0]]))
        sodddiaggate = cirq.MatrixGate(np.array([[0,1],[-1j,0]]))

        cy_moments = 0
        ghzh_counter = 0

        for midx, moment in enumerate(criq_rep):
            new_moment = []
            this_moment_is_CY = False

            for op in moment:

                if op.gate == cirq.ops.SingleQubitCliffordGate(_clifford_tableau=cirq.CliffordTableau( #ONLY FOR UNITARY PREP
                    1, rs=np.array([True, False]), xs=np.array([[True], [True]]), zs=np.array([[False], [True]]))):
                    new_moment.append(cirq.PhasedXPowGate(phase_exponent=3/4, exponent=1/2).on(op.qubits[0]))
                
                elif op.gate == cirq.ControlledGate(cirq.Y): #add a CH gate layer
                    if this_moment_is_CY:
                        pass # print(f"{op} at moment {midx} is removed")
                    else:
                        ghzqub1 = op.qubits[0].x

                        if cy_moments % 3 == 0: #gate layer 1: CCZs
                            ccz1 = short_range_ccz_factory(cirq.LineQubit(ghzqub1), cirq.LineQubit(4), cirq.LineQubit(4*d2))
                            ccz2 = short_range_ccz_factory(cirq.LineQubit(ghzqub1+1), cirq.LineQubit(2*d2+6), cirq.LineQubit(6*d2+2))
                            ccz3 = short_range_ccz_factory(cirq.LineQubit(ghzqub1+2), cirq.LineQubit(4*d2+8), cirq.LineQubit(8*d2+4))
                            edited_cirq_rep.append(cirq.Moment([ccz1, ccz2, ccz3]))
                            
                            #add 3q and idle noise
                            n1 = cirq.depolarize(p=p, n_qubits=3)(cirq.LineQubit(ghzqub1) , cirq.LineQubit(4), cirq.LineQubit(4*d2))
                            n2 = cirq.depolarize(p=p, n_qubits=3)(cirq.LineQubit(ghzqub1+1) ,cirq.LineQubit(2*d2+6) ,cirq.LineQubit(6*d2+2) )
                            n3 = cirq.depolarize(p=p, n_qubits=3)(cirq.LineQubit(ghzqub1+2) , cirq.LineQubit(4*d2+8) , cirq.LineQubit(8*d2+4))

                            nidles1 = [cirq.depolarize(p=p, n_qubits=1)(k) for k in [cirq.LineQubit(0), 
                                                                                     cirq.LineQubit(2*d2+2), cirq.LineQubit(4*d2+4),
                                                                                      cirq.LineQubit(6*d2+6), cirq.LineQubit(8*d2+8),
                                                                                     cirq.LineQubit(8) , cirq.LineQubit(8*d2)]]
                            new_moment.append([n1, n2, n3] + nidles1)

                        elif cy_moments % 3 == 1: #gate layer 2 diagonal gates
                            cs1 =  sevendiaggate( cirq.LineQubit(0)).controlled_by(cirq.LineQubit(ghzqub1)) 
                            csdag2 = sodddiaggate( cirq.LineQubit(2*d2+2)).controlled_by(cirq.LineQubit(ghzqub1+1)) 
                            cs3 = sevendiaggate( cirq.LineQubit(4*d2+4)).controlled_by(cirq.LineQubit(ghzqub1+2)) 
                            edited_cirq_rep.append(cirq.Moment([cs1, csdag2, cs3]))

                            #add 2q and idle noise
                            n1 = cirq.depolarize(p=p, n_qubits=2)(cirq.LineQubit(ghzqub1) , cirq.LineQubit(0))
                            n2 = cirq.depolarize(p=p, n_qubits=2)(cirq.LineQubit(ghzqub1+1) ,cirq.LineQubit(2*d2+2) )
                            n3 = cirq.depolarize(p=p, n_qubits=2)(cirq.LineQubit(ghzqub1+2) , cirq.LineQubit(4*d2+4))
                            
                            nidles2 = [cirq.depolarize(p=p, n_qubits=1)(k) for k in [cirq.LineQubit(4), cirq.LineQubit(4*d2), 
                                                                                     cirq.LineQubit(2*d2+6),cirq.LineQubit(6*d2+2),
                                                                                      cirq.LineQubit(4*d2+8) , cirq.LineQubit(8*d2+4),
                                                                                    cirq.LineQubit(8) , cirq.LineQubit(8*d2),
                                                                                    cirq.LineQubit(6*d2+6), cirq.LineQubit(8*d2+8)]]
                            new_moment.append([n1, n2, n3] + nidles2)

                        elif cy_moments % 3 == 2: #gate layer 3: 2 diags one off diag
                            csdag1 = sodddiaggate( cirq.LineQubit(6*d2+6)).controlled_by(cirq.LineQubit(ghzqub1)) 
                            cs2 = sevendiaggate( cirq.LineQubit(8*d2+8)).controlled_by(cirq.LineQubit(ghzqub1+1)) 
                            ccz3last = long_range_ccz_factory(cirq.LineQubit(ghzqub1+2), cirq.LineQubit(8), cirq.LineQubit(8*d2))
                            edited_cirq_rep.append(cirq.Moment([csdag1, cs2, ccz3last]))

                            #add 2q & 3q and idle noise
                            n1 = cirq.depolarize(p=p, n_qubits=2)(cirq.LineQubit(ghzqub1) , cirq.LineQubit(6*d2+6))
                            n2 = cirq.depolarize(p=p, n_qubits=2)(cirq.LineQubit(ghzqub1+1) ,cirq.LineQubit(8*d2+8) )
                            n3 = cirq.depolarize(p=p, n_qubits=3)(cirq.LineQubit(ghzqub1+2), cirq.LineQubit(8) , cirq.LineQubit(8*d2))

                            
                            nidles3 = [cirq.depolarize(p=p, n_qubits=1)(k) for k in [cirq.LineQubit(4), cirq.LineQubit(4*d2), 
                                                                                     cirq.LineQubit(2*d2+6),cirq.LineQubit(6*d2+2),
                                                                                      cirq.LineQubit(4*d2+8) , cirq.LineQubit(8*d2+4),
                                                                                    cirq.LineQubit(0) , cirq.LineQubit(2*d2+2),
                                                                                    cirq.LineQubit(4*d2+4)]]
                            new_moment.append([n1, n2, n3] + nidles3)

                        this_moment_is_CY = True 

                elif op.gate == cirq.H and op.qubits[0].x == 2*(d2+1)**2 +1:
                    if ghzh_counter % 2 == 1:
                        edited_cirq_rep.append(cirq.Moment(cirq.ZPowGate(exponent=-0.25)(op.qubits[0])))
                        new_moment.append(op) #("Adding T before meas")
                    else:
                        new_moment.append(op)
                    ghzh_counter+=1
                    
                else:
                    new_moment.append(op)

            if this_moment_is_CY:
                cy_moments+=1
            edited_cirq_rep.append(cirq.Moment(new_moment))
    else: #no need to reprocess circuit
        edited_cirq_rep = criq_rep

    moved_qubits = tuple(sorted(edited_cirq_rep.all_qubits()))
    movement_profile = build_movement_profile(
        movement_model=movement_model,
        moved_qubits=moved_qubits,
        zone_separation_um=args.zone_separation_um,
        transport_benchmark_distance_um=args.transport_benchmark_distance_um,
        transport_benchmark_time_us=args.transport_benchmark_time_us,
        transfer_one_way_us=args.transfer_one_way_us,
        previous_round_cycle_us=args.previous_round_cycle_us,
        previous_round_movement_us=args.previous_round_movement_us,
        dephase_rate_per_us=args.movement_dephase_rate_per_us,
        loss_rate_per_us=args.movement_loss_rate_per_us,
    )
    if movement_profile is not None:
        edited_cirq_rep = add_movement_dephasing(
            circuit=edited_cirq_rep,
            moved_qubits=moved_qubits,
            per_window_dephase_p=movement_profile["per_window_dephase_p"],
        )



    ##### now simulate
    if True: #turn off to only sample clifford part
        errloc_sv = {}
        post_sel_shots = np.array([0,0,0])
        movement_prefix_discards = 0
        logical_qubit = cirq.LineQubit(4*d2+4)
        qubit_order = tuple(sorted(edited_cirq_rep.all_qubits()))
        simulator, use_density_sim = build_simulator(
            sim_method=sim_method,
            gate_model="kraus"
            if "kraus" in {short_range_gate_model, long_range_gate_model}
            else "unitary",
            seed=tasknum,
        )

        for r in range(num_sverr_shots):
            if movement_profile is not None:
                loss_hits = rng.random((2, movement_profile["num_qubits"])) < movement_profile["per_window_loss_p"]
                if np.any(loss_hits):
                    movement_prefix_discards += 1
                    continue

            result = simulator.simulate(edited_cirq_rep)
            sorted_items = sorted(result.measurements.items(), key=lambda item: int(item[0]))
            msmt_list = ([value[0] for key, value in sorted_items])
            post_sel_msmts = msmt_list[:-12]
            dec_msmts = msmt_list[-12:]

            if sum(post_sel_msmts) == 0:
                key = get_pauli_err_from_syndrome(dec_msmts, cirq_conversion=True)
                # logical_bloch_vector knows how to extract the logical qubit's
                # reduced state from either a pure state (unitary path) or the
                # final density matrix (non-unitary contractive_kraus path).
                bl_logical_sv = [
                    round(elem, 5)
                    for elem in logical_bloch_vector(
                        sim_result=result,
                        logical_qubit=logical_qubit,
                        qubit_order=qubit_order,
                        use_density_sim=use_density_sim,
                    )
                ]
                
                #update bloch vector
                if (dec_msmts[10] + dec_msmts[1] + dec_msmts[0]) % 2 == 1: #zstabs from x error
                    bl_logical_sv[1] = -bl_logical_sv[1]
                    bl_logical_sv[2] = -bl_logical_sv[2]
                if (dec_msmts[9] + dec_msmts[5]) % 2 == 1: #xstabs from z error
                    bl_logical_sv[1] = -bl_logical_sv[1]
                    bl_logical_sv[0] = -bl_logical_sv[0]
                
                add_errloc_entry(errloc=key, sv=(bl_logical_sv), target_dict=errloc_sv)
            else:
                if sum(post_sel_msmts[:endofinj]) > 0: post_sel_shots[0] = post_sel_shots[0] +1 
                elif sum(post_sel_msmts[endofinj:endofinj+3]) > 0: post_sel_shots[1] = post_sel_shots[1] +1
                elif sum(post_sel_msmts[endofinj+3:endofinj+6]) > 0: post_sel_shots[2] = post_sel_shots[2] +1


        post_sel_shots = [int(v) for v in post_sel_shots]
        data = {
            "errloc_sv": errloc_sv,
            "post_select_stage": post_sel_shots,
            "movement_prefix_discards": int(movement_prefix_discards),
        }
        if movement_profile is not None:
            data["movement_profile"] = movement_profile
        print("errloc_sv", errloc_sv)
        print("post_select_stage", post_sel_shots)
        print("movement_prefix_discards", movement_prefix_discards)

        if not os.path.exists(path):
            os.makedirs(path)
        # save data to a JSON file
        with open (path+f'b{basis}_prep{prep}_dfinal{d2}_' + str(tasknum) + 'v2.json', 'w') as f:
            json.dump(data,f,cls=NumpyEncoder)

    if True: # start post sv outputs. turn off to only sample svs

        
        # Generate the file path
        file_path = path+f'b{basis}_prep{prep}_dfinal{d2}_' + str(tasknum) + 'v2.json'

        # Open the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # get the error dictionary for the specified cshot_num
        err_list = data.get('errloc_sv', np.nan)
        effective_cult_disc = np.sum(data.get('post_select_stage', 0))*num_shots_per_sverr
        effective_movement_disc = int(data.get("movement_prefix_discards", 0)) * num_shots_per_sverr

        overall_discards = 0
        overall_errors = 0
        overall_time = 0
        overall_fid = 0
        interim_custom_counts  = defaultdict(int)

        task_json_md = {
            'p': p,
            'b': basis,
            'noise': 'uniform',
            'd2': d2,
            'c': f'Tcirq-handoff-{ccz_label_slug}-{sim_method_slug}{"" if movement_model == "none" else f"-{movement_slug}"}-{tag}',
            'ghz_size': ghz_size,
            'latter_rounds': latter_rounds,
            'short_range_ccz_label': short_range_ccz_label,
            'long_range_ccz_label': long_range_ccz_label,
            'use_custom_ccz': USE_CUSTOM_CCZ,
            'custom_ccz_mode': effective_ccz_mode if USE_CUSTOM_CCZ else 'ideal',
            'long_range_ccz_mode': long_range_ccz_mode,
            'ccz_near_unitary_atol': args.ccz_near_unitary_atol,
            'short_range_ccz_gate_model': short_range_gate_model,
            'long_range_ccz_gate_model': long_range_gate_model,
            'ideal_ccz_locations': ideal_ccz_locations,
            'short_range_preset': short_range_preset,
            'long_range_preset': long_range_preset,
            'sim_method': sim_method,
            'movement_model': movement_model,
        }
        if movement_profile is not None:
            task_json_md.update(
                {
                    'movement_cycle_us': movement_profile['movement_cycle_us'],
                    'previous_cycle_us': movement_profile['previous_check_cycle_us'],
                    'movement_overhead_us': movement_profile['movement_overhead_us'],
                    'movement_parked_check_us': movement_profile['parked_check_us'],
                    'movement_move_in_us': movement_profile['move_in_us'],
                    'movement_move_out_us': movement_profile['move_out_us'],
                    'movement_zone_cross_us': movement_profile['zone_cross_us'],
                    'movement_num_qubits': movement_profile['num_qubits'],
                    'movement_dephase_rate_per_us': movement_profile['dephase_rate_per_us'],
                    'movement_loss_rate_per_us': movement_profile['loss_rate_per_us'],
                    'movement_per_window_dephase_p': movement_profile['per_window_dephase_p'],
                    'movement_per_window_loss_p': movement_profile['per_window_loss_p'],
                    'movement_cycle_ratio_vs_previous': movement_profile['cycle_ratio_vs_previous'],
                }
            )
        if args.fault_distance_tag is not None:
            task_json_md['f'] = args.fault_distance_tag
        

        for errloc, svs in err_list.items(): #each errloc is a pauli destabilizer
        
            samples_with_errloc = (sum(num for _, num in svs))

            #prepare probabilistisc svs
            unedited_svs = [lsv_pair[0] for lsv_pair in svs]
            lsv_probabilities = np.array([lsv_pair[1] for lsv_pair in svs]) / samples_with_errloc
            probabilistic_sv_locs = rng.choice(
                len(unedited_svs),
                size=num_shots_per_sverr * samples_with_errloc,
                p=lsv_probabilities,
            )
            probabilistic_svs = [unedited_svs[i] for i in probabilistic_sv_locs]

            print("\nSampling ", num_shots_per_sverr * samples_with_errloc, "shots for", errloc)

            # The non-unitary CCZ does not appear directly in this Stim stage.
            # Instead, the exact-sim stage above has already converted the
            # custom gate physics into:
            # - the distribution over accepted error locations, and
            # - the logical Bloch vectors attached to those locations.
            # This stage then resamples the corresponding stabilizer circuits.
            errored_circ = full_circuit(nm=p, 
                         dfinal=d2,
                         ps_on_d3=1,
                         prep='',
                         neutralatom=False,
                         handoff=True,
                         ref=False,
                         err=int(errloc)
                         )

            #sinter sampling - to replace traditional sampling
            task = [sinter.Task(
                circuit= errored_circ,
                detector_error_model=ref_dem,
                json_metadata=task_json_md,
            )]
            res = sinter.collect(tasks=task, 
                            num_workers=1, 
                            decoders = ['pymatching-gap'],
                            custom_decoders=sinter_samplers(),
                            max_shots=num_shots_per_sverr * samples_with_errloc,  # Set a reasonable default
                            )
                            
            res = res[0] #contains all statistics for this errloc
            print(res)
            overall_time += res.seconds
            total_ds_errs = res.errors
            overall_discards += res.discards
        
            #start post processing
            gapped_errs = 0
            trivial_gapped_errs = 0
            fid = 0
            shot_considered_idx = 0
            # Count the mistakes.

            for key in (res.custom_counts): #go through gapped shots
                for shot_considered in range(res.custom_counts[key]):

                    unedited_sv = probabilistic_svs[shot_considered_idx]
                    shot_considered_idx+=1
                    
                    #0 is z meas, 1 is x meas
                    if key[0] == 'Y': 
                        res_sc =  [-unedited_sv[0], unedited_sv[1], -unedited_sv[2]]
                        trivial_gapped_errs +=1
                    elif key[0] == 'X': 
                        res_sc = [unedited_sv[0], -unedited_sv[1], -unedited_sv[2]]
                        gapped_errs += 1
                    elif key[0] == 'Z':
                        res_sc = [-unedited_sv[0], -unedited_sv[1], unedited_sv[2]]
                        gapped_errs += 1
                    else:
                        res_sc = unedited_sv

                    if np.allclose(res_sc, true_bloch_rep, atol=1e-4):
                        icc_key = 'C' + key[1:]
                    else:
                        icc_key = 'E' + key[1:]
                        overall_errors += 1
                        #print("error", icc_key, "error added due to SV output", unedited_sv, "which became", res_sc,  flush=True)
                    
                    interim_custom_counts[icc_key] += 1
            
            #print("gapped errors vs yerrs vs total errors", gapped_errs, trivial_gapped_errs, total_ds_errs)
                    
            overall_fid += fid

        print(interim_custom_counts)
        
        finp_res = sinter.TaskStats(shots = int(num_shots_per_sverr*num_sverr_shots),
                                    discards= int(round(overall_discards+effective_cult_disc+effective_movement_disc)),
                                    errors = int(round(overall_errors)),
                                    decoder = 'pymatching-gap',
                                    strong_id = hashlib.sha256((json.dumps(task_json_md, sort_keys=True)).encode()).hexdigest(),
                                    seconds = float(overall_time),
                                    custom_counts= Counter(interim_custom_counts),
                                    json_metadata=task_json_md)
                               
        file_path = f"realcirqhandoffoutputs_{ccz_label_slug}_{sim_method_slug}{movement_path_tag}_p{str(int(1000*p))}_{tag}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        try:
            with open(file_path+f"/TaskStats_b{basis}p{int(1000*p)}_prep{prep}_dfinal{d2}_{ccz_label_slug}_{sim_method_slug}{movement_path_tag}_{tasknum}.json", "w") as f:
                print(finp_res.to_csv_line(), file=f)
        except IOError as e:
            print(f"Error saving file: {e}")
