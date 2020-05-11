"""
The entry point for the Oneka project.

Classes
-------
None

Raises
------
None.

Functions
---------
oneka(
        projectname, runtime,
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        buffer=100, spacing=10, umbra=10, smooth=0,
        confined=True, tol=1, maxstep=10)
    The entry-point for the Oneka project. As currently written,
    this driver computes and plots the stochastic capture zone.

log_the_run(
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        buffer, spacing, umbra,
        confined, tol, maxstep)
    Print the banner and run information to the log file.

Notes
-----
o   This package is a work in progress.

o   We need to think about what events to log.

o   This module currently generates plots using python's matplotlib
    facility. We will remove these plots when we integrate into
    ArcGIS Pro.

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

import logging
import matplotlib.pyplot as plt
import numpy as np

from oneka.archive import dump_oneka
from oneka.probabilityfield import ProbabilityField
from oneka.stochastic import create_stochastic_capturezone, isdistribution
from oneka.utilities import filter_obs, summary_statistics
from oneka.visualize import create_probability_plot, create_impact_plot

log = logging.getLogger('Oneka')

VERSION = '10 May 2020'


# ------------------------------------------------------------------------------
def oneka(
        projectname, runtime,
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        buffer=100, spacing=10, umbra=10, smooth=0,
        confined=True, tol=1, maxstep=10):
    """
    The entry-point for the Oneka project. As currently written,
    this driver computes and plots the stochastic capture zone.

    Arguments
    ---------
    projectname : string
        An identifying name.

    runtime : string
        The asctime string for this run.

    target : int
        The index identifying the target well in the stochastic_wells.
        That is, the well for which we will compute a stochastic
        capture zone. This uses python's 0-based indexing.

    npaths : int
        The minimum number of paths (starting points for the backtraces)
        to generate uniformly around the target well.

    duration : float
        The duration of the capture zone [d]. For example, a 10-year
        capture zone would have a duration = 10*365.25.

    nrealizations : int
        The number of realizations of the random model.

    base : float
        The aquifer base elevation [m].

    c_dist : scalar, pair, or triple
        The distribution of the aquifer conductivity [m/d]:
            scalar -> constant,
            pair   -> (min, max) for a uniform distribution, or
            triple -> (min, mode, max) for a triangular distribution.

    p_dist : scalar, pair, or triple
        The distribution of the aquifer porosity [.]:
            scalar -> constant,
            pair   -> (min, max) for a uniform distribution, or
            triple -> (min, mode, max) for a triangular distribution.

    t_dist : scalar, pair, or triple
        The distribution of the aquifer thickness [m]:
            scalar -> constant,
            pair   -> (min, max) for a uniform distribution, or
            triple -> (min, mode, max) for a triangular distribution.

    stochastic_wells : list of stochastic well tuples
        A well tuple contains four values (sort of): (xw, yw, rw, qdist)
            xw : float
                The x-coordinate of the well [m].
            yw : float
                The y-coordinate of the well [m].
            rw : float
                The radius of the well [m].
            q_dist : scalar, pair, or triple
                The distribution of the well discharge. If
                    scalar -> constant,
                    pair -> (min, max) for a uniform distribution, or
                    triple -> (min, mode, max) for a triangular distribution.

    observations : list of observation tuples.
        An observation tuple contains four values: (x, y, z_ev, z_std), where
            x : float
                The x-coordinate of the observation [m].
            y : float
                The y-coordinate of the observation [m].
            z_ev : float
                The expected value of the observed static water level elevation [m].
            z_std : float
                The standard deviation of the observed static water level elevation [m].

    buffer : float, optional [default = 100]
        The buffer distance [m] around each well. If an obs falls
        within buffer of any well, it is removed.

    spacing : float, optional
        The spacing of the rows and the columns [m] in the square
        ProbabilityField grids. Default is 10.

    umbra : float, optional
        The vector-to-raster range [m] when mapping a particle path
        onto the ProbabilityField grids. If a grid node is within
        umbra of a particle path, it is marked as visited. Default is 10.

    smooth : scalar, optional
        The nominal 'standard deviation' for the Gaussian kernel smoother
        (scipy.ndimage.gaussian_filter) to be applied to the probability
        contours. The units are in grids.

    confined : boolean, optional
        True if it is safe to assume that the aquifer is confined
        throughout the domain of interest, False otherwise. This is a
        speed kludge. Default is True.

    tol : float, optional
        The tolerance [m] for the local error when solving the
        backtrace differential equation. This is an inherent
        parameter for an adaptive Runge-Kutta method. Default is 1.

    maxstep : float, optional
        The maximum allowed step in space [m] when solving the
        backtrace differential equation. This is a maximum space
        step and NOT a maximum time step. Default is 10.

    Returns
    -------
    A capturezone probability field.

    Notes
    -----
    o Most of the work outlined above is orchestrated by the
      create_capturezone function.
    """

    # Validate the arguments.
    assert(isinstance(projectname, str))
    assert(isinstance(runtime, str))

    assert(isinstance(target, int) and 0 <= target < len(stochastic_wells))
    assert(isinstance(npaths, int) and 0 < npaths)
    assert((isinstance(duration, int) or isinstance(duration, float)) and 0 < duration)

    assert(isinstance(base, int) or isinstance(base, float))
    assert(isdistribution(c_dist, 0, np.inf))
    assert(isdistribution(p_dist, 0, 1))
    assert(isdistribution(t_dist, 0, np.inf))

    assert(isinstance(stochastic_wells, list) and len(stochastic_wells) >= 1)
    for we in stochastic_wells:
        assert(len(we) == 4 and
               (isinstance(we[0], int) or isinstance(we[0], float)) and
               (isinstance(we[1], int) or isinstance(we[1], float)) and
               (isinstance(we[2], int) or isinstance(we[2], float)) and 0 < we[2] and
               isdistribution(we[3], -np.inf, np.inf))

    assert(isinstance(observations, list) and len(observations) > 6)
    for ob in observations:
        assert(len(ob) == 4 and
               (isinstance(ob[0], int) or isinstance(ob[0], float)) and
               (isinstance(ob[1], int) or isinstance(ob[1], float)) and
               (isinstance(ob[2], int) or isinstance(ob[2], float)) and
               (isinstance(ob[3], int) or isinstance(ob[3], float)) and 0 <= ob[3])

    assert((isinstance(buffer, int) or isinstance(buffer, float)) and 0 < buffer)
    assert((isinstance(spacing, int) or isinstance(spacing, float)) and 0 < spacing)
    assert((isinstance(umbra, int) or isinstance(umbra, float)) and 0 < umbra)
    assert((isinstance(smooth, int) or isinstance(smooth, float)) and 0 < smooth)

    assert(isinstance(confined, bool))
    assert((isinstance(tol, int) or isinstance(tol, float)) and 0 < tol)
    assert((isinstance(maxstep, int) or isinstance(maxstep, float)) and 0 < maxstep)

    # Log the run information.
    log_the_run(
        projectname, runtime,        
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        buffer, spacing, umbra, smooth,
        confined, tol, maxstep)

    # Filter out all of the obs that are too close to any pumping well.
    obs = filter_obs(observations, stochastic_wells, buffer)
    assert(len(obs) > 6)

    # Log the summary statistics.
    buf = summary_statistics(obs, ['Easting', 'Northing', 'Head', 'Stdev'], 
        ['12.2f', '12.2f', '12.2f', '12.2f'], 'Retained Observations')
    log.info('\n')
    log.info(buf.getvalue())

    # Create the stochastic capture zone for the target well.
    pfield = create_stochastic_capturezone(
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, obs,
        spacing, umbra, confined,
        tol, maxstep)

    # Archive the run.
    dump_oneka(
        projectname, runtime,
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        buffer, spacing, umbra, smooth,
        confined, tol, maxstep,
        pfield)

    # Make the impact plot.
    pr, area = create_impact_plot(spacing, pfield)

    # Log the decile results.
    log.info('\n')
    log.info('===========================================')
    log.info(' Pr(capture)         Area             Area ')
    log.info(' Exceeds            [m^2]          [acres] ')
    log.info('-------------------------------------------')
    for p in np.linspace(0.05, 0.95, 19):
        i = np.argmax(pr<=p)
        log.info('{0:8.3f} {1:16,.0f} {2:16,.2f}'
            .format(pr[i], area[i], area[i]/4046.86))               # 4046.86 m^2/acre
    log.info('===========================================')

    # Make the filled contour plot.
    create_probability_plot(target, stochastic_wells, obs, pfield, smooth)

    plt.show()

# ------------------------------------------------------------------------------
def log_the_run(
        projectname, runtime,    
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        buffer, spacing, umbra, smooth,
        confined, tol, maxstep):
    """
    Log all of the defining arguments for the model run.
    """

    log.info('')
    log.info('========================================')
    log.info(' OOOOOO  NN   N  EEEEEE  K   KK  AAAAAA ')
    log.info(' O    O  N N  N  E       K KK    A    A ')
    log.info(' O    O  N  N N  EEEEE   KK      AAAAAA ')
    log.info(' O    O  N   NN  E       K KK    A    A ')
    log.info(' OOOOOO  N    N  EEEEEE  K   KK  A    A ')
    log.info('========================================')
    log.info('Version: {0}'.format(VERSION))
    log.info('')

    log.info('project name  = {0}'.format(projectname))
    log.info('run time      = {0}'.format(runtime))
    log.info('target        = {0:d}'.format(target))
    log.info('npaths        = {0:d}'.format(npaths))
    log.info('duration      = {0:.2f}'.format(duration))
    log.info('nrealizations = {0:d}'.format(nrealizations))
    log.info('base          = {0:.2f}'.format(base))
    log.info('c_dist        = {0}'.format(c_dist))
    log.info('p_dist        = {0}'.format(p_dist))
    log.info('t_dist        = {0}'.format(t_dist))
    log.info('buffer        = {0:.2f}'.format(buffer))
    log.info('spacing       = {0:.2f}'.format(spacing))
    log.info('umbra         = {0:.2f}'.format(umbra))
    log.info('smooth        = {0}'.format(smooth))
    log.info('confined      = {0}'.format(confined))
    log.info('tol           = {0:.2f}'.format(tol))
    log.info('maxstep       = {0:.2f}'.format(maxstep))

    log.info('\n')
    log.info('stochastic_wells: {0}'.format(len(stochastic_wells)))
    for we in stochastic_wells:
        log.info('    {0}'.format(we))

    log.info('\n')
    log.info('observations: {0}'.format(len(observations)))
    for ob in observations:
        log.info('    {0}'.format(ob))

    log.info('\n')
