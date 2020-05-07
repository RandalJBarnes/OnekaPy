"""
The entry point for the OnekaPy project.

Classes
-------
    None

Exceptions
----------
    None.

Functions
---------
    oneka(
            target, npaths, duration, nrealizations,
            base, c_dist, p_dist, t_dist,
            wellfield, observations,
            buffer=100, spacing=10, umbra=10,
            confined=True, tol=1, maxstep=10)
        The entry-point for the OnekaPy project. As currently written,
        this driver computes and plots the stochastic capture zone.

    filter_obs(observations, wellfield, buffer)
        Partition the obs into retained and removed. An observation is
        removed if it is within buffer of a well. Duplicate observations
        (i.e. obs at the same loction) are average using a minimum
        variance weighted average.

    log_the_run(
            target, npaths, duration, nrealizations,
            base, c_dist, p_dist, t_dist,
            wellfield, observations,
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
    07 May 2020
"""

import logging
import matplotlib.pyplot as plt
import numpy as np

from onekapy.probabilityfield import ProbabilityField
from onekapy.stochastic import compute_stochastic_capturezone, isdistribution
from onekapy.utilities import contour_head, filter_obs, summary_statistics

log = logging.getLogger('OnekaPy')

VERSION = '02 May 2020'


# ------------------------------------------------------------------------------
def oneka(
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        wellfield, observations,
        buffer=100, spacing=10, umbra=10,
        confined=True, tol=1, maxstep=10):
    """
    The entry-point for the OnekaPy project. As currently written,
    this driver computes and plots the stochastic capture zone.

    Parameters
    ----------
    target : int
        The index identifying the target well in the wellfield.
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

    wellfield : list of stochastic well tuples
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
    assert(isinstance(target, int) and 0 <= target < len(wellfield))
    assert(isinstance(npaths, int) and 0 < npaths)
    assert((isinstance(duration, int) or isinstance(duration, float)) and 0 < duration)

    assert(isinstance(base, int) or isinstance(base, float))
    assert(isdistribution(c_dist, 0, np.inf))
    assert(isdistribution(p_dist, 0, 1))
    assert(isdistribution(t_dist, 0, np.inf))

    assert(isinstance(wellfield, list) and len(wellfield) >= 1)
    for we in wellfield:
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

    assert(isinstance(confined, bool))
    assert((isinstance(tol, int) or isinstance(tol, float)) and 0 < tol)
    assert((isinstance(maxstep, int) or isinstance(maxstep, float)) and 0 < maxstep)

    # Log the run information.
    log_the_run(
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        wellfield, observations,
        buffer, spacing, umbra,
        confined, tol, maxstep)

    # Filter out all of the obs that are too close to any pumping well.
    obs = filter_obs(observations, wellfield, buffer)
    assert(len(obs) > 6)

    # Initialize the probability field.
    xtarget, ytarget, rtarget = wellfield[target][0:3]
    cz = ProbabilityField(spacing, spacing, xtarget, ytarget)

    # Compute the stochastic capture zone for the target well.
    compute_stochastic_capturezone(
        xtarget, ytarget, rtarget,
        npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        wellfield, obs,
        cz, umbra, confined,
        tol, maxstep)

    # Make the probability contour plot.
    plt.figure()
    plt.clf()
    plt.axis('equal')

    if cz.total_weight > 0:
        X = np.linspace(cz.xmin, cz.xmax, cz.ncols)
        Y = np.linspace(cz.ymin, cz.ymax, cz.nrows)
        Z = cz.pgrid/cz.total_weight
        plt.contourf(X, Y, Z, np.linspace(0, 1, 11), cmap='tab10')
        plt.colorbar(ticks=np.linspace(0, 1, 11))
        plt.contour(X, Y, Z, np.linspace(0, 1, 11), colors=['black'])

        plt.xlabel('UTM Easting [m]')
        plt.ylabel('UTM Northing [m]')
        plt.title('{0} Realizations, {1} Paths, Duration = {2:.1f} days'
                  .format(nrealizations, npaths, duration), fontsize=20)
        plt.grid(True)

    else:
        log.warning(' There were no valid realizations.')

    plot_locations(plt, target, wellfield, obs)

    plt.show()

# ------------------------------------------------------------------------------
def plot_locations(plt, target, wellfield, obs):

    # Plot the wells as o markers.
    xw = [we[0] for we in wellfield]
    yw = [we[1] for we in wellfield]
    plt.plot(xw, yw, 'o', markeredgecolor='k', markerfacecolor='w')

    # Plot the target well as a star marker.
    xtarget, ytarget = wellfield[target][0:2]
    plt.plot(xtarget, ytarget, '*', markeredgecolor='k', markerfacecolor='w', markersize=12)

    # Plot the retained observations as fat + markers.
    xo = [ob[0] for ob in obs]
    yo = [ob[1] for ob in obs]
    plt.plot(xo, yo, 'P', markeredgecolor='k', markerfacecolor='w')


# ------------------------------------------------------------------------------
def log_the_run(
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        wellfield, observations,
        buffer, spacing, umbra,
        confined, tol, maxstep):

    log.info('')
    log.info(' ========================================================')
    log.info('  OOOOOO  NN   N  EEEEEE  K   KK  AAAAAA  PPPPPP  Y    Y ')
    log.info('  O    O  N N  N  E       K KK    A    A  P    P   Y  Y  ')
    log.info('  O    O  N  N N  EEEEE   KK      AAAAAA  PPPPPP    YY   ')
    log.info('  O    O  N   NN  E       K KK    A    A  P         Y    ')
    log.info('  OOOOOO  N    N  EEEEEE  K   KK  A    A  P         Y    ')
    log.info(' ========================================================')
    log.info(' Version: {0}'.format(VERSION))
    log.info('')

    log.info(' target        = {0:d}'.format(target))
    log.info(' npaths        = {0:d}'.format(npaths))
    log.info(' duration      = {0:.2f}'.format(duration))
    log.info(' nrealizations = {0:d}'.format(nrealizations))
    log.info(' base          = {0:.2f}'.format(base))
    log.info(' c_dist        = {0}'.format(c_dist))
    log.info(' p_dist        = {0}'.format(p_dist))
    log.info(' t_dist        = {0}'.format(t_dist))
    log.info(' buffer        = {0:.2f}'.format(buffer))
    log.info(' spacing       = {0:.2f}'.format(spacing))
    log.info(' umbra         = {0:.2f}'.format(umbra))
    log.info(' confined      = {0}'.format(confined))
    log.info(' tol           = {0:.2f}'.format(tol))
    log.info(' maxstep       = {0:.2f}'.format(maxstep))

    log.info('\n')
    log.info(' wellfield: {0}'.format(len(wellfield)))
    for we in wellfield:
        log.info('     {0}'.format(we))

    log.info('\n')
    log.info(' observations: {0}'.format(len(observations)))
    for ob in observations:
        log.info('     {0}'.format(ob))

    log.info('\n')
