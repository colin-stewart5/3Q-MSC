import sinter
import pymatching
import numpy as np
from typing import List, Callable, Tuple
from collections import Counter, defaultdict
import cirq
import stimcirq
import sys
import json, hashlib
import time
import datetime
import os

from full_clifford_sim.main_complied_fxns import full_circuit, generate_sv_cirquit
from full_clifford_sim.gap_sampler import *
from full_clifford_sim.cirq_utilities import *

## take DEM from ref_circuit, and decode shots from sample circutis


tasknum = int(sys.argv[2])

if __name__ == '__main__':
    
    num_shots_per_sverr = 20
    num_sverr_shots = 10

    p = 0.001 * int(sys.argv[1])
    errmodel = 'uniform'
    basis =  'HXY'
    if basis == 'HXY':
        true_bloch_rep = [1/np.sqrt(2),1/np.sqrt(2),0]
        magic = True
    else:
        magic = False
    ghz_size = 3
    errD = False
    cX =  True
    d2 = 7
    prep = 'unitstab'

    if prep == "unitstab":
        endofinj = 8
    elif prep == "hookinj":
        endofinj = 12
    elif prep == "optunit":
        endofinj = 2
    else:
        raise ValueError("injection stratey not well defined")

    x = datetime.datetime.now()
    path = './realcirqhandoffmid_p'+str(int(1000*p))+'_outs'+x.strftime("%m%d")+'/'
        
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
                            ccz1 = cirq.CCZ(cirq.LineQubit(ghzqub1) , cirq.LineQubit(4), cirq.LineQubit(4*d2))
                            ccz2 = cirq.CCZ(cirq.LineQubit(ghzqub1+1) ,cirq.LineQubit(2*d2+6) ,cirq.LineQubit(6*d2+2) )
                            ccz3 = cirq.CCZ(cirq.LineQubit(ghzqub1+2) , cirq.LineQubit(4*d2+8) , cirq.LineQubit(8*d2+4))
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
                            ccz3last = cirq.CCZ(cirq.LineQubit(ghzqub1+2) , cirq.LineQubit(8) , cirq.LineQubit(8*d2))
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
        simulator = cirq.Simulator(seed=tasknum)

        for r in range(num_sverr_shots):

            result = simulator.simulate(edited_cirq_rep)
            sorted_items = sorted(result.measurements.items(), key=lambda item: int(item[0]))
            msmt_list = ([value[0] for key, value in sorted_items])
            post_sel_msmts = msmt_list[:-12]
            dec_msmts = msmt_list[-12:]

            if sum(post_sel_msmts) == 0:
                key = get_pauli_err_from_syndrome(dec_msmts, cirq_conversion=True)
                bl_logical_sv = [round(elem,5) for elem in result.bloch_vector_of( qubit=cirq.LineQubit(4*d2+4))]
                
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


        post_sel_shots = list(post_sel_shots)
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

        task_json_md = {'p': p, 'b':basis, 'noise':'uniform', 'd2':d2, 
                                    'c':f'Tcirq-handoff-{x.strftime("%m%d")}',
                                    'ghz_size':ghz_size, 'latter_rounds':3}
        

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
        
        finp_res = sinter.TaskStats(shots = num_shots_per_sverr*num_sverr_shots,
                                    discards= round(overall_discards+effective_cult_disc),
                                    errors = round(overall_errors),
                                    decoder = 'pymatching-gap',
                                    strong_id = hashlib.sha256((json.dumps(task_json_md, sort_keys=True)).encode()).hexdigest(),
                                    seconds = overall_time,
                                    custom_counts= Counter(interim_custom_counts),
                                    json_metadata=task_json_md)
                               
        file_path = f"realcirqhandoffoutputs_p{str(int(1000*p))}_{x.strftime('%m%d')}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        try:
            with open(file_path+f"/TaskStats_b{basis}p{int(1000*p)}_prep{prep}_dfinal{d2}_{tasknum}.json", "w") as f:
                print(finp_res.to_csv_line(), file=f)
        except IOError as e:
            print(f"Error saving file: {e}")