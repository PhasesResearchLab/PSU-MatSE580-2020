"""
The eqplot module contains functions for general plotting of
the results of equilibrium calculations.
"""
from pycalphad.core.utils import unpack_condition
from pycalphad.plot.utils import phase_legend
import pycalphad.variables as v
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# TODO: support other state variables here or make isinstance elif == v.T or v.P
_plot_labels = {v.T: 'Temperature (K)', v.P: 'Pressure (Pa)'}


def _axis_label(ax_var):
    if isinstance(ax_var, v.MoleFraction):
        return f'X({ax_var.species.name})'
    elif isinstance(ax_var, v.ChemicalPotential):
        return f'MU({ax_var.species.name})'
    elif isinstance(ax_var, v.StateVariable):
        return _plot_labels[ax_var]
    else:
        return ax_var

    
def _map_coord_to_variable(coord):
    """
    Map a coordinate to a StateVariable object.

    Parameters
    ----------
    coord : str
        Name of coordinate in equilibrium object.

    Returns
    -------
    pycalphad StateVariable
    """
    vals = {'T': v.T, 'P': v.P}
    if coord.startswith('X_'):
        return v.X(coord[2:])
    elif coord in vals:
        return vals[coord]
    else:
        return coord
    
    
MOLAR_QUANTITIES = (v.entropy, v.volume)
POTENTIALS = (v.temperature, v.pressure)

def combine_phase_labels(separator, phase_labels, sort=True):
    # for constistency in labeling sort the phase names and combine them,
    # i.e. [[BETA, ALPHA], [ALPHA, BETA]] -> [[ALPHA, BETA], [ALPHA, BETA]]
    phase_labels = np.sort(phase_labels)
    # combine, collapsing along the vertex axis
    combined_labels = np.empty_like(phase_labels[:, 0])
    sep = np.full_like(phase_labels[:, 0], separator)
    first_iter = True
    for nn in range(phase_labels.shape[-1]):
        if first_iter:
            first_iter = False
        else:
            combined_labels = np.char.add(combined_labels, sep)
        combined_labels = np.char.add(combined_labels, phase_labels[:, nn])
    return combined_labels


def is_molar_quantity(variable):
    # TODO: how can this be extended to user variables?
    if any(variable is mq for mq in MOLAR_QUANTITIES):
        return True
    elif isinstance(variable, v.MoleFraction):
        return True
    elif any(variable is pot for pot in POTENTIALS):
        return False
    elif isinstance(variable, v.ChemicalPotential):
        return False
    else:
        raise ValueError(f"Cannot determined whether the variable `{variable}` is a potential or molar quantity. Make sure you are using instances of `pycalphad.variables` objects.")



def get_eq_axis_data(eq_dataset, x, y, num_phases):
    """Get plotting data for the desired axis based on phases in equilibrium"""
    tielines_in_diagram_plane = any(map(is_molar_quantity, (x, y)))

    # find all N-phase regions
    n_phase_idx = np.nonzero(np.sum(eq_dataset.Phase.values != '', axis=-1, dtype=np.int) == num_phases)


    data = []
    for axis_variable in (x, y):
        if isinstance(axis_variable, v.MoleFraction):
            phase_compositions = eq_dataset.X.sel(component=axis_variable.species.name).values[n_phase_idx][..., :num_phases]
            data.append(phase_compositions)
        elif isinstance(axis_variable, v.ChemicalPotential):
            # TODO: handle tie-line case
            eq_chemical_potentials = eq_dataset.MU.sel(component=axis_variable.species.name).values[n_phase_idx]
            if tielines_in_diagram_plane:
                # each molar quantity needs a corresponding value at this 
                # potential to plot. Tile to shape: (num_pots, num_phases)
                eq_chemical_potentials = np.tile(eq_chemical_potentials, (num_phases, 1)).T
            data.append(eq_chemical_potentials)
        elif any(axis_variable is pot for pot in POTENTIALS):
            pot_idx = eq_dataset.Phase.dims.index(str(axis_variable))
            eq_pots = eq_dataset[str(axis_variable)].values[n_phase_idx[pot_idx]]
            if tielines_in_diagram_plane:
                # each molar quantity needs a corresponding value at this 
                # potential to plot. Tile to shape: (num_pots, num_phases)
                eq_pots = np.tile(eq_pots, (num_phases, 1)).T
            data.append(eq_pots)
        else:
            raise NotImplementedError(f"Cannot plot axis variable {desired_axis_variable}")
    # Set up phase labels
    if tielines_in_diagram_plane:
        # labels correspond to all the phases individually
        labels = eq_dataset.Phase.values[n_phase_idx][..., :num_phases]
    else:
        # labels correspond to the combination of phases at each equilibrium point
        phase_labels = eq_dataset.Phase.values[n_phase_idx][..., :num_phases]
        labels = combine_phase_labels('+', phase_labels)
    return data[0], data[1], labels
    
    

def eqplot(eq, ax=None, x=None, y=None, z=None, tielines=True, resize=False, **kwargs):
    """
    Plot the result of an equilibrium calculation.

    The type of plot is controlled by the degrees of freedom in the equilibrium calculation.

    Parameters
    ----------
    eq : xarray.Dataset
        Result of equilibrium calculation.
    ax : matplotlib.Axes
        Default axes used if not specified.
    x : StateVariable, optional
    y : StateVariable, optional
    z : StateVariable, optional
    tielines : bool
        If True, will plot tielines
    kwargs : kwargs
        Passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    matplotlib AxesSubplot
    """
    # TODO: add kwargs for tie-lines with defaults

    conds = OrderedDict([(_map_coord_to_variable(key), unpack_condition(np.asarray(value)))
                         for key, value in sorted(eq.coords.items(), key=str)
                         if (key in ('T', 'P', 'N')) or (key.startswith('X_'))])
    indep_comps = sorted([key for key, value in conds.items() if isinstance(key, v.MoleFraction) and len(value) > 1], key=str)
    indep_pots = [key for key, value in conds.items() if (type(key) is v.StateVariable) and len(value) > 1]

    # we need to wrap the axes handling in these, becase we don't know ahead of time what projection to use.
    # contractually: each inner loops must 
    #   1. Define the ax (calling plt.gca() with the correct projection if none are passed)
    #   2. Define legend_handles
    if len(indep_comps) == 1 and len(indep_pots) == 1:
        # binary system with composition and a potential as coordinates
        ax = ax if ax is not None else plt.gca()
        # find x and y, default to x=v.X and y=v.T
        x = x if x is not None else indep_comps[0]
        y = y if y is not None else indep_pots[0]
    
        # Get all two phase data
        x_2, y_2, labels = get_eq_axis_data(eq, x, y, 2)
        if any(map(is_molar_quantity, (x, y))):
            # The diagram has tie-lines that must be plotted.
            legend_handles, colormap = phase_legend(sorted(np.unique(labels)))
            # plot x vs. y for each phase (phase index 0 and 1)
            kwargs.setdefault('s', 20)
            for phase_idx in range(2):
                # TODO: kwargs.setdefault('c', [colormap[ph] for ph in labels[..., phase_idx]])
                ax.scatter(x_2[..., phase_idx], y_2[..., phase_idx], c=[colormap[ph] for ph in labels[..., phase_idx]], **kwargs)
            if tielines:
                ax.plot(x_2.T, y_2.T, c=[0, 1, 0, 1], linewidth=0.5, zorder=-1)
        else:
            # This diagram does not have tie-lines, we plot x vs. y directly
            legend_handles, colormap = phase_legend(sorted(np.unique(labels)))
            kwargs.setdefault('s', 20)
            # TODO: kwargs colors
            colorlist = [colormap[ph] for ph in labels]
            ax.scatter(x_2, y_2, c=colorlist, **kwargs)

    elif len(indep_comps) == 2 and len(indep_pots) == 0:
        # This is a ternary isothermal, isobaric calculation
        # Default to x and y of mole fractions
        x = x if x is not None else indep_comps[0]
        y = y if y is not None else indep_comps[1]
        # Find two and three phase data
        x2, y2, labels_2 = get_eq_axis_data(eq, x, y, 2)
        x3, y3, labels_3 = get_eq_axis_data(eq, x, y, 3)
        if any(map(is_molar_quantity, (x, y))):
            # The diagram has tie-lines that must be plotted.
            if isinstance(x, v.MoleFraction) and isinstance(x, v.MoleFraction):
                # Both the axes are mole fractions, so the the compositions
                # form a simplex. Use Gibbs "triangular" projection.
                ax = ax if ax is not None else plt.gca(projection='triangular')
                # TODO: code for handling projection specific things here
                ax.yaxis.label.set_rotation(60)
                # Here we adjust the x coordinate of the ylabel.
                # We make it reasonably comparable to the position of the xlabel from the xaxis
                # As the figure size gets very large, the label approaches ~0.55 on the yaxis
                # 0.55*cos(60 deg)=0.275, so that is the xcoord we are approaching.
                ax.yaxis.label.set_va('baseline')
                fig_x_size = ax.figure.get_size_inches()[0]
                y_label_offset = 1 / fig_x_size
                ax.yaxis.set_label_coords(x=(0.275 - y_label_offset), y=0.5)
            else:
                ax = ax if ax is not None else plt.gca()

            # Plot two phase data
            legend_handles, colormap = phase_legend(sorted(np.unique(labels_2)))
            kwargs.setdefault('s', 20)
            # TODO: color kwargs
            for phase_idx in range(2):
                ax.scatter(x2[..., phase_idx], y2[..., phase_idx], c=[colormap[ph] for ph in labels_2[..., phase_idx]], **kwargs)
            if tielines:
                # Plot tie-lines between two phases
                ax.plot(x2.T, y2.T, c=[0, 1, 0, 1], linewidth=0.5, zorder=-1)

            # Find and plot three phase tie-triangles
            # plot lines between all three pairs of phases to form tie-triangles
            for phase_idx_pair in ((0, 1), (0, 2), (1, 2)):
                ax.plot(x3[:, phase_idx_pair].T, y3[:, phase_idx_pair].T, c=[1, 0, 0, 1], lw=0.5, zorder=-1)

        else:
            # This diagram does not have tie-lines, we plot x vs. y directly
            ax = ax if ax is not None else plt.gca()
            combined_labels = np.unique(labels_2).tolist() + np.unique(labels_3).tolist()
            legend_handles, colormap = phase_legend(sorted(combined_labels))
            kwargs.setdefault('s', 20)
            ax.scatter(x2, y2, c=[colormap[ph] for ph in labels_2], **kwargs)
            ax.scatter(x3, y3, c=[colormap[ph] for ph in labels_3], **kwargs)
    else:
        raise ValueError('The eqplot projection is not defined and cannot be autodetected. There are {} independent compositions and {} indepedent potentials.'.format(len(indep_comps), len(indep_pots)))
            

    # position the phase legend and configure plot
    if resize:
        if not 'Triangular' in str(type(ax)):
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # TODO: special limits resizing handling for different axes types
#         ax.set_xlim([np.min(conds[x]) - 1e-2, np.max(conds[x]) + 1e-2])
#         ax.set_ylim([np.min(conds[y]), np.max(conds[y])])
        ax.tick_params(axis='both', which='major', labelsize=12)


    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
    comps = eq.component.values.tolist()
    plot_title = '-'.join([component.title() for component in sorted(comps) if component != 'VA'])
    ax.set_title(plot_title, fontsize=20)
    ax.set_xlabel(_axis_label(x), labelpad=15, fontsize=20)
    ax.set_ylabel(_axis_label(y), fontsize=20)

    return ax