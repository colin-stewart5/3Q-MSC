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

from full_clifford_sim.main_complied_fxns import full_circuit, generate_sv_cirquit
from full_clifford_sim.gap_sampler import *
from full_clifford_sim.cirq_utilities import *

## take DEM from ref_circuit, and decode shots from sample circutis


USE_CUSTOM_CCZ = True
CUSTOM_CCZ_LABEL = "rydberg_ccz_pulse2"

# Paste your 8x8 CCZ-like matrix here. Leave as None to keep the ideal Cirq CCZ.
CUSTOM_CCZ_MATRIX = np.array(
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

# Supported values:
# - "unitary": require CUSTOM_CCZ_MATRIX to be unitary and use cirq.Simulator.
# - "contractive_kraus": treat CUSTOM_CCZ_MATRIX as a 3-qubit Kraus operator K,
#   automatically completing the channel with sqrt(I - K^\dagger K). This
#   requires K^\dagger K <= I and uses cirq.DensityMatrixSimulator.
# - "auto": use "unitary" if possible, otherwise try "contractive_kraus".
#
# Non-unitary CCZ support is wired in here already. To force the non-unitary
# path for a physical gate model:
# 1. Set CUSTOM_CCZ_MODE = "contractive_kraus".
# 2. Put the measured 8x8 operator into CUSTOM_CCZ_MATRIX.
# 3. Ensure it is contractive: K^\dagger K <= I up to
#    CUSTOM_CCZ_CONTRACTIVE_ATOL, otherwise the script will reject it.
# 4. The exact-simulation stage below will then switch to
#    cirq.DensityMatrixSimulator automatically.
#
# Caveat: only the Cirq exact-sim prefix uses the non-unitary gate directly.
# The later Stim/sinter resampling stage still operates on the stabilizerized
# errored circuits keyed by the exact-sim outputs; Stim itself is not carrying
# the non-unitary CCZ channel.
CUSTOM_CCZ_MODE = "auto"
CUSTOM_CCZ_CONTRACTIVE_ATOL = 1e-6
CUSTOM_CCZ_NEAR_UNITARY_ATOL = 1e-4


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def current_ccz_label() -> str:
    if USE_CUSTOM_CCZ:
        return CUSTOM_CCZ_LABEL
    return "ideal"


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


def build_ccz_factory() -> tuple[Callable[[cirq.Qid, cirq.Qid, cirq.Qid], cirq.Operation], bool]:
    if not USE_CUSTOM_CCZ:
        return lambda q0, q1, q2: cirq.CCZ(q0, q1, q2), False

    matrix = _as_8x8_complex_matrix(CUSTOM_CCZ_MATRIX)
    if matrix is None:
        raise ValueError("USE_CUSTOM_CCZ=True but CUSTOM_CCZ_MATRIX is None.")

    mode = CUSTOM_CCZ_MODE.lower()
    if mode not in {"auto", "unitary", "contractive_kraus"}:
        raise ValueError(f"Unsupported CUSTOM_CCZ_MODE={CUSTOM_CCZ_MODE!r}.")

    if mode in {"auto", "unitary"}:
        if _is_unitary(matrix):
            gate = cirq.MatrixGate(matrix)
            return lambda q0, q1, q2: gate.on(q0, q1, q2), False

        # Option 1: treat the supplied gate as a coherent near-unitary pulse
        # and project it onto the closest exact unitary so the exact-sim stage
        # can stay on the statevector simulator.
        if _is_unitary(matrix, atol=CUSTOM_CCZ_NEAR_UNITARY_ATOL):
            projected = _nearest_unitary(matrix)
            gate = cirq.MatrixGate(projected)
            return lambda q0, q1, q2: gate.on(q0, q1, q2), False

    if mode == "unitary":
        raise ValueError(
            "CUSTOM_CCZ_MODE='unitary' requires a unitary or near-unitary 8x8 matrix. "
            "For leakage-like behavior, switch to 'contractive_kraus' or 'auto'."
        )

    # Non-unitary path: represent the supplied 8x8 matrix as a Kraus operator
    # and complete it into a trace-preserving channel. Returning True here is
    # what tells the caller to use DensityMatrixSimulator instead of the
    # statevector simulator.
    gate = ContractiveCustomCCZGate(matrix)
    return lambda q0, q1, q2: gate.on(q0, q1, q2), True


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one handoff shard for the HXY workflow with a custom CCZ gate."
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tasknum = args.tasknum
    
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
    ccz_label = current_ccz_label()
    ccz_label_slug = _slugify(ccz_label)

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
    path = f'./realcirqhandoffmid_{ccz_label_slug}_p{int(1000*p)}_outs{tag}/'
        
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
    ccz_factory, use_density_sim = build_ccz_factory()

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
                            ccz1 = ccz_factory(cirq.LineQubit(ghzqub1), cirq.LineQubit(4), cirq.LineQubit(4*d2))
                            ccz2 = ccz_factory(cirq.LineQubit(ghzqub1+1), cirq.LineQubit(2*d2+6), cirq.LineQubit(6*d2+2))
                            ccz3 = ccz_factory(cirq.LineQubit(ghzqub1+2), cirq.LineQubit(4*d2+8), cirq.LineQubit(8*d2+4))
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
                            ccz3last = ccz_factory(cirq.LineQubit(ghzqub1+2), cirq.LineQubit(8), cirq.LineQubit(8*d2))
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



    ##### now simulate
    if True: #turn off to only sample clifford part
        errloc_sv = {}
        post_sel_shots = np.array([0,0,0])
        logical_qubit = cirq.LineQubit(4*d2+4)
        qubit_order = tuple(sorted(edited_cirq_rep.all_qubits()))
        # If build_ccz_factory returned use_density_sim=True, we are in the
        # non-unitary CCZ branch above and must propagate a density matrix
        # through the exact-simulation prefix.
        if use_density_sim:
            simulator = cirq.DensityMatrixSimulator(seed=tasknum)
        else:
            simulator = cirq.Simulator(seed=tasknum)

        for r in range(num_sverr_shots):

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
        data = {"errloc_sv": errloc_sv,  "post_select_stage":post_sel_shots}
        print("errloc_sv", errloc_sv)
        print( "post_select_stage",post_sel_shots)

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
            'c': f'Tcirq-handoff-{ccz_label_slug}-{tag}',
            'ghz_size': ghz_size,
            'latter_rounds': latter_rounds,
            'ccz_label': ccz_label,
            'use_custom_ccz': USE_CUSTOM_CCZ,
            'custom_ccz_mode': CUSTOM_CCZ_MODE if USE_CUSTOM_CCZ else 'ideal',
        }
        if args.fault_distance_tag is not None:
            task_json_md['f'] = args.fault_distance_tag
        

        for errloc, svs in err_list.items(): #each errloc is a pauli destabilizer
        
            samples_with_errloc = (sum(num for _, num in svs))

            #prepare probabilistisc svs
            unedited_svs = [lsv_pair[0] for lsv_pair in svs]
            lsv_probabilities = np.array([lsv_pair[1] for lsv_pair in svs]) / samples_with_errloc
            probabilistic_sv_locs = np.random.choice(range(len(unedited_svs)), 
                                           num_shots_per_sverr * samples_with_errloc,
                                           p = lsv_probabilities)
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
                                    discards= int(round(overall_discards+effective_cult_disc)),
                                    errors = int(round(overall_errors)),
                                    decoder = 'pymatching-gap',
                                    strong_id = hashlib.sha256((json.dumps(task_json_md, sort_keys=True)).encode()).hexdigest(),
                                    seconds = float(overall_time),
                                    custom_counts= Counter(interim_custom_counts),
                                    json_metadata=task_json_md)
                               
        file_path = f"realcirqhandoffoutputs_{ccz_label_slug}_p{str(int(1000*p))}_{tag}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        try:
            with open(file_path+f"/TaskStats_b{basis}p{int(1000*p)}_prep{prep}_dfinal{d2}_{ccz_label_slug}_{tasknum}.json", "w") as f:
                print(finp_res.to_csv_line(), file=f)
        except IOError as e:
            print(f"Error saving file: {e}")
