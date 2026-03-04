import stim
import numpy as np
import json

def get_pauli_err_from_syndrome(msmt_list_unr, cirq_conversion:bool = False):
    # """ This function takes in ms and outputs a Pauli error distribution, according to a choice of destabilizers. 
    # Index of data qubits:
    # Z-------Z-------Z
    # 0 ───── 1 ───── 2
    # │ \   /   \   / │ 
    # │   3       4   │
    # │ /   \   /   \ │
    # ...

    if not cirq_conversion: msmt_list  =  np.flip(msmt_list_unr).astype(int)
    else: msmt_list = np.array(msmt_list_unr).astype(int); #print("not flipping meas list")

    x_syndromes = [msmt_list[3], msmt_list[2], (msmt_list[5]+msmt_list[3])%2, msmt_list[8],msmt_list[9],(msmt_list[11]+msmt_list[8])%2]
    z_syndromes = [msmt_list[7], msmt_list[0], (msmt_list[10]+msmt_list[7])%2, msmt_list[4],msmt_list[6],(msmt_list[1]+msmt_list[10]+msmt_list[4]+msmt_list[6])%2]

    """x_syndromes = [msmt_list[11], msmt_list[10],
                   msmt_list[6], msmt_list[5],
                   msmt_list[1], msmt_list[0]]
    z_syndromes = [msmt_list[4], msmt_list[9],
                   msmt_list[3], msmt_list[8],
                   msmt_list[2], msmt_list[7]]"""


    # Destabilizers are defined such that they take an invalid X syndrome and put horizontal strings of Z errors to the right boundary of the code.
    # invalid Z syndromes have vertical X errors to the bottom boundary of the code.
    # """
    # Define the destabilizers (there are 12 of them)
    dx = [[1,2],[2], # (X destabilizers anticommute with an X syndrome. These are the indices of the data qubits that the Pauli Z operator will be applied on.)
        [6,7],[7], 
        [11,12],[12]]
    dz =  [[10],[5,10],# (Z destabilizers anticommute with a Z syndrome)
            [11],[6,11],
            [12],[7,12]]
    # create dictionary to store each data qubit and whether it has a pauli X/Z on it.
    err_dict = {}
    for i in range(13):
        err_dict[i] = {}
        err_dict[i]['x']=0
        err_dict[i]['z']=0

    dx_ids = np.flatnonzero(x_syndromes)
    dz_ids = np.flatnonzero(z_syndromes)

    for dx_id in dx_ids:
        for datqub in dx[dx_id]:
            err_dict[datqub]['z']= err_dict[datqub]['z'] ^ 1
    for dz_id in dz_ids:
        for datqub in dz[dz_id]:
            err_dict[datqub]['x']= err_dict[datqub]['x'] ^ 1


    ### convert the dictionary into a more compact format ###
    def convert_dict(d:dict):
        # each qubit i has either I:0, X:1, Z:2, or Y:3
        # the value of qubit i is given by 4^i * (0,1,2, or 3)

        val = 0
        for qubit, q in d.items():
            match q['x'],q['z']:
                case 0,0:
                    val += 4**qubit * 0
                case 1,0: # X error
                    val += 4**qubit * 1
                case 0,1: # Z error
                    val += 4**qubit * 2
                case 1,1: # Y error
                    val += 4**qubit * 3
        return val
    
    val = convert_dict(err_dict)

    return val

def add_errloc_entry(errloc:int, sv:dict, target_dict:dict):

    sv= list(sv)
    if errloc not in target_dict:
        target_dict[errloc] = [(sv, 1)]
    else:
        for i, (existing_sv, count) in enumerate(target_dict[errloc]):
            if existing_sv == sv:
                target_dict[errloc][i] = (existing_sv, count + 1)
                break
        else:
            target_dict[errloc].append((sv, 1))

def remove_QCs(cc_orig:stim.Circuit, meas_err:float, reverse_Y: bool) -> stim.Circuit:

    cc = stim.Circuit()

    #first remove detectors and coords
    for line in (cc_orig):
        if line.name != "QUBIT_COORDS" and line.name != "DETECTOR" and line.name != "SHIFT_COORDS" and line.name != "H_YZ":
            cc.append(line)
        if line.name == "H_YZ": #-Y eigenstate for testing
            if reverse_Y:
                cc.append("SQRT_X", line.targets_copy())
            else:
                cc.append(line)


    #now add meas errors
    redduced_criq = stim.Circuit()

    lines = len(cc)

    for lidx, line in enumerate(cc):
        if line.name == "M" and lidx < lines - 5:
            redduced_criq.append("X_ERROR", arg=meas_err, targets=line.targets_copy())
        redduced_criq.append(line)
        

    return redduced_criq

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)