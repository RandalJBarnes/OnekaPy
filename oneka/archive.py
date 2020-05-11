"""
Archive, dump and load, an Oneka run into a bzip2 pickle file.

Classes
-------
None

Raises
------
None.

Functions
---------
dump_oneka(
        projectname, runtime,    
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        buffer, spacing, umbra, smooth,
        confined, tol, maxstep,
        pfield)

load_oneka(filename)
    
Authors
-------
Dr. Randal J. Barnes
Department of Civil, Environmental, and Geo- Engineering
University of Minnesota

Richard Soule
Source Water Protection
Minnesota Department of Health

Version
-------
11 May 2020
"""

import bz2
from datetime import datetime
import pickle


# ------------------------------------------------------------------------------
def dump_oneka(
        projectname, runtime,
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        buffer, spacing, umbra, smooth,
        confined, tol, maxstep,
        pfield):

    # Create the dump file name.
    filename='logs\\Oneka' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.bz2'

    # Create a dictionary.
    oneka_dict = {
        'projectname' : projectname,
        'runtime' : runtime,
        'target' : target,
        'npaths' : npaths, 
        'duration' : duration, 
        'nrealizations' : nrealizations,
        'base' : base, 
        'c_dist' : c_dist, 
        'p_dist' : p_dist, 
        't_dist' : t_dist,
        'stochastic_wells' : stochastic_wells, 
        'observations' : observations,
        'buffer' : buffer, 
        'spacing' : spacing, 
        'umbra' : umbra,
        'smooth' : smooth,
        'confined' : confined, 
        'tol' : tol, 
        'maxstep' : maxstep,
        'pfield' : pfield}

    with bz2.BZ2File(filename, "w") as fp:
        pickle.dump(oneka_dict, fp)

# ------------------------------------------------------------------------------
def load_oneka(filename):
    with bz2.BZ2File(filename, "r") as fp:
        oneka_dict = pickle.load(fp)

    return (oneka_dict)
