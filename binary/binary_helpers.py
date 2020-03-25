from pycalphad import equilibrium, variables as v
import numpy as np
import matplotlib.pyplot as plt

def plot_convex_hull(dbf, comps, phases, conds, ax=None):
    if ax is None:
        ax = plt.gca()      

    result = equilibrium(dbf, comps, phases, conds)
    unique_phase_sets = np.unique(result['Phase'].values.squeeze(), axis=0)
    
    comp_cond = [c for c in conds.keys() if isinstance(c, v.X)]
    if len(comp_cond) != 1:
        raise ValueError(f"Exactly one composition condition required for plotting convex hull. Got {len(comp_cond)} with conditions for {comp_cond}")
    comp_cond = str(comp_cond[0])
        
    for phase_set in unique_phase_sets:
        label = '+'.join([ph for ph in phase_set if ph != ''])
        # composition indices with the same unique phase
        unique_phase_idx = np.nonzero(np.all(result['Phase'].values.squeeze() == phase_set, axis=1))[0]
        masked_result = result.isel(**{comp_cond: unique_phase_idx})
        ax.plot(masked_result[comp_cond].squeeze(), masked_result.GM.squeeze(), color='limegreen', linestyle=':', lw=4, label='convex hull', zorder=3)
    return ax