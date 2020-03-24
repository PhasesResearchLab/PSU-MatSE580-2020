import matplotlib.pyplot as plt
import numpy as np
from pycalphad import calculate, variables as v
def PT_phase_diagram(dbf, comps, phases, conds, x=v.T):
    # Compute based on crossover temperatures in calculate
    # Only works for systems without any internal degrees of freedom
    from pycalphad.core.utils import unpack_condition
    from collections import defaultdict
    T_phase_pair_lines = defaultdict(list)
    P_phase_pair_lines = defaultdict(list)
    all_phase_pairs = set()
    pressures = unpack_condition(conds[v.P])
    for P in pressures:
        cr = calculate(dbf, comps, phases, P=P, T=conds[v.T], N=conds[v.N])
        phase_values = cr["Phase"].values.squeeze()
        phase_idx = cr.GM.values.squeeze().argmin(axis=1)
        phase_change_temp_index = np.nonzero(phase_idx[:-1] != phase_idx[1:])    
        for temp_idx in phase_change_temp_index[0]:
            phase_pair = phase_values[0, phase_idx[temp_idx]] + ' + ' + cr["Phase"].values.squeeze()[0, phase_idx[temp_idx+1]]
            T_phase_pair_lines[phase_pair].append(cr["T"][temp_idx])
            P_phase_pair_lines[phase_pair].append(P)
            all_phase_pairs |= {phase_pair}
    for phase_pair in all_phase_pairs:
        if x == v.T:
            plt.plot(T_phase_pair_lines[phase_pair], P_phase_pair_lines[phase_pair], label=phase_pair)
        elif x == v.P:
            plt.plot(P_phase_pair_lines[phase_pair], T_phase_pair_lines[phase_pair], label=phase_pair)
    if x == v.T:
        plt.xlabel('Temperature [K]')
        plt.ylabel('Pressure [Pa]')
        plt.xlim(conds[v.T][0], conds[v.T][1])
        plt.ylim(conds[v.P][0], conds[v.P][1])
    else:
        plt.xlabel('Pressure [Pa]')
        plt.ylabel('Temperature [K]')
        plt.xlim(conds[v.P][0], conds[v.P][1])
        plt.ylim(conds[v.T][0], conds[v.T][1])
    plt.legend(loc=(1.1, 0.5))
