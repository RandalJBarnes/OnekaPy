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
    oneka
        The entry-point for the OnekaPy project. As currently written,
        this driver computes and plots the stochastic capture zone.

    filter_obs(observations, wells, buffer):
        Partition the obs into retained and removed. An observation is
        removed if it is within buffer of a well. Duplicate observations
        (i.e. obs at the same loction) are average using a minimum
        variance weighted average.

    log_banner()
        Sends a program banner to the log file.

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
    26 April 2020
"""

import logging
import matplotlib.pyplot as plt
import numpy as np

from capturezone import compute_capturezone
from utility import isnumber, isposnumber, isposint, isvalidindex, isvaliddist


log = logging.getLogger(__name__)

VERSION = '26 April 2020'


# -----------------------------------------------
def oneka(
        target, minpaths, duration, nrealizations,
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

    minpaths : int
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
    o The following steps are carried out for each realization.

        (1) Generate a new set of random well discharges. The
            locations are fixed, but the discharges may be random.

        (2) Generate a new set of random aquifer properties:
            conductivity, porosity, and thickness.

        (3) Fit the model to the observations to determine the
            expected value vector and covariance matrix of the six
            regional flow parameters, A - F.

        (4) Generate the regional flow parameters as a realization
            of a multivariate normal distribution using the expected
            value vector and covariance matrix generated in step (3).

        (5) Generate and backtrack minpaths of particles uniformly
            distributed around the target well.

        (6) Infill additional paths where necessary.

        (7) Chronicle the particle traces in the ProbabilityField
            grid.

    o Most of the work outlined above is orchestrated by the
      create_capturezone function.
    """

    # Validate the arguments. This is minimal validation, but it
    # should catch the typos and simple tranpositional errors.
    assert(isposint(minpaths))
    assert(isposnumber(duration))
    assert(isposint(nrealizations))

    assert(isvaliddist(c_dist, 0, np.inf))
    assert(isvaliddist(p_dist, 0, 1))
    assert(isvaliddist(t_dist, 0, np.inf))

    assert(isinstance(wellfield, list) and len(wellfield) >= 1)
    for we in wellfield:
        assert(len(we) == 4 and isnumber(we[0]) and isnumber(we[1]) and
               isposnumber(we[2]) and isvaliddist(we[3], -np.inf, np.inf))
    assert(isvalidindex(target, len(wellfield)))

    assert(isinstance(observations, list) and len(observations) > 6)
    for ob in observations:
        assert(len(ob) == 4 and isnumber(ob[0]) and isnumber(ob[1]) and
               isnumber(ob[2]) and isposnumber(ob[3]))

    assert(isposnumber(buffer))
    assert(isposnumber(spacing))
    assert(isposnumber(umbra))

    assert(isinstance(confined, bool))
    assert(isposnumber(tol))
    assert(isposnumber(maxstep))

    # Log the run information.
    log_the_run(
        target, minpaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        wellfield, observations,
        buffer, spacing, umbra,
        confined, tol, maxstep)

    # Filter out all of the obs that are too close to any pumping well.
    obs = filter_obs(observations, wellfield, buffer)
    assert(len(obs) > 6)

    # Compute the capture zone for the target well.
    cz = compute_capturezone(
        target, minpaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        wellfield, obs,
        spacing, umbra, confined, tol, maxstep)

    # Make the probability contour plot.
    plt.figure()
    plt.clf()
    plt.axis('equal')

    if cz.total_weight > 0:
        X = np.linspace(cz.xmin, cz.xmax, cz.ncols)
        Y = np.linspace(cz.ymin, cz.ymax, cz.nrows)
        Z = cz.pgrid/cz.total_weight
        plt.contourf(X, Y, Z, np.linspace(0, 1, 21), cmap='bwr')
        plt.colorbar()
    else:
        log.warning(' There were no valid realizations.')

    # Plot the wells as o markers.
    xw = [we[0] for we in wellfield]
    yw = [we[1] for we in wellfield]
    plt.plot(xw, yw, 'o', markeredgecolor='k', markerfacecolor='w')

    # Plot the target well as a star marker.
    xtarget, ytarget, rtarget = wellfield[target][0:3]
    plt.plot(xtarget, ytarget, '*', markeredgecolor='k', markerfacecolor='w', markersize=12)

    # Plot the retained observations as fat + markers.
    xo = [ob[0] for ob in obs]
    yo = [ob[1] for ob in obs]
    plt.plot(xo, yo, 'P', markeredgecolor='k', markerfacecolor='w')

    return cz


# -------------------------------------
def filter_obs(observations, wells, buffer):
    """
    Partition the obs into retained and removed. An observation is
    removed if it is within buffer of a well. Duplicate observations
    (i.e. obs at the same loction) are average using a minimum
    variance weighted average.

    Parameters
    ----------
    observations : list
        A list of observation tuples where the first two fields
        are x and y:
            x : float
                The x-coordinate of the observation [m].
            y : float
                The y-coordinate of the observation [m].

    wells : list
        A list of well tuples where the first two fields of the
        tuples are xw and yw:
            xw : float
                The x-coordinate of the well [m].

            yw : float
                The y-coordinate of the well [m].

        Note: the well tuples may have other fields, but the first
        two must be xw and yw.

    buffer : float
        The buffer distance [m] around each well. If an obs falls
        within buffer of any well, it is removed.

    Returns
    -------
    retained_obs : list
        A list of the retained observations. The fields are the
        same as those in obs. These include averaged duplicates.

    Notes
    -----
    o   Duplicate observations are averaged and the associated
        standard deviation is updated to reflect this. We use a
        weighted average, with the weight for the i'th obs
        proportional to 1/sigma^2_i. This is the minimum variance
        estimator. See, for example,
        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    """

    # Remove all observations that are too close to pumping wells.
    obs = []
    for ob in observations:
        flag = True
        for we in wells:
            if np.hypot(ob[0]-we[0], ob[1]-we[1]) <= buffer:
                flag = False
                break
        if flag:
            obs.append(ob)
        else:
            log.info(' observation removed: {0}'.format(ob))

    # Replace any duplicate observations with their weighted average.
    # Assume that the duplicate errors are statistically independent.
    obs.sort()
    retained_obs = []

    i = 0
    while i < len(obs):
        j = i+1
        while (j < len(obs)) and (np.hypot(obs[i][0]-obs[j][0], obs[i][1]-obs[j][1]) < 1):
            j += 1

        if j-i > 1:
            num = 0
            den = 0
            for k in range(i, j):
                num += obs[k][2]/obs[k][3]**2
                den += 1/obs[k][3]**2
                log.info(' duplicate observation: {0}'.format(obs[k]))
            retained_obs.append((obs[i][0], obs[i][1], num/den, np.sqrt(1/den)))
        else:
            retained_obs.append(obs[i])
        i = j

    log.info('active observations: {0}'.format(len(retained_obs)))
    for ob in retained_obs:
        log.info('     {0}'.format(ob))

    return retained_obs


# -------------------------------------
def log_the_run(
        target, minpaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        wellfield, observations,
        buffer, spacing, umbra,
        confined, tol, maxstep):

    log.info('                                                   ')
    log.info(' ==================================================')
    log.info('  OOOOOO NN   N EEEEEE K   KK AAAAAA PPPPPP Y    Y ')
    log.info('  O    O N N  N E      K KK   A    A P    P  Y  Y  ')
    log.info('  O    O N  N N EEEEE  KK     AAAAAA PPPPPP   YY   ')
    log.info('  O    O N   NN E      K KK   A    A P        Y    ')
    log.info('  OOOOOO N    N EEEEEE K   KK A    A P        Y    ')
    log.info(' ==================================================')
    log.info(' Version: {0}'.format(VERSION))
    log.info('                                                   ')

    log.info(' target        = {0:d}'.format(target))
    log.info(' minpaths      = {0:d}'.format(minpaths))
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

    log.info(' wellfield: {0}'.format(len(wellfield)))
    for we in wellfield:
        log.info('     {0}'.format(we))

    log.info(' observations: {0}'.format(len(observations)))
    for ob in observations:
        log.info('     {0}'.format(ob))

    log.info(' ')
