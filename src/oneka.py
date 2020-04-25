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

Notes
-----
o   This package is a work in progress.

o   This package uses the python logging facility. As expected, we log
    warnings and errors. We also log quite a bit a bit of information
    about the run.

o   We need to think about what events to log.

o   This package currently generates plots using python's matplotlib
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
    25 April 2020
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import random

from capturezone import compute_capturezone
from utility import isnumber, isposnumber, isposint, isvalidindex, isvaliddist


log = logging.getLogger(__name__)


# -----------------------------------------------
def oneka(
        target, nrays, duration, tol, maxstep, nrealizations,
        base, c_dist, p_dist, t_dist, confined,
        wellfield, observations, buffer,
        deltax, deltay, umbra):
    """
    The entry-point for the OnekaPy project. As currently written
    the driver computes and plots the stochatic capture zone.

    Parameters
    ----------
    target : int
        The index identifying the target well in the wellfield.
        That is, the well for which we will compute a stochastic
        capture zone. This uses python's 0-based indexing.

    nrays : int
        The number of rays (starting points for the backtraces) to
        generate uniformly around the target well.

    duration : float
        The duration of the capture zone [d]. For example, a 10-year
        capture zone would have a duration = 10*365.25.

    tol : float
        The tolerance [m] for the local error when solving the
        backtrace differential equation. This is an inherent
        parameter for an adaptive Runge-Kutta method.

    maxstep : float
        The maximum allowed step in space [m] when solving the
        backtrace differential equation. This is a maximum space
        step and NOT a maximum time step.

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

    confined : boolean
        True if it is safe to assume that the aquifer is confined
        throughout the domain of interest, False otherwise. This is a
        speed kludge.

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

    buffer : float
        The buffer distance [m] around each well. If an obs falls
        within buffer of any well, it is removed.

    deltax : float
        The spacing of the columns [m] in the ProbabilityField grids.

    deltay : float
        The spacing of the rows [m] in the ProbabilityField grids.

    umbra : float
        The vector-to-raster range [m] when mapping a particle path
        onto the ProbabilityField grids. If a grid node is within
        umbra of a particle path, it is marked as visited.

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

        (5) Generate and backtrack nrays of particles uniformly
            distributed around the target well.

        (6) Chronicle the particle traces in the ProbabilityField
            grid.

    o Most of the work outlined above is orchestrated by the capturezone
        function.

    o We may be able to parallelize the tracking particles to speed up
        the overall execution.
    """

    # Validate the arguments. This is minimal validation, but it
    # should catch the typos and simple tranpositional errors.
    assert(isposint(nrays))
    assert(isposnumber(duration))
    assert(isposnumber(tol))
    assert(isposnumber(maxstep))
    assert(isposint(nrealizations))

    assert(isvaliddist(c_dist, 0, np.inf))
    assert(isvaliddist(p_dist, 0, 1))
    assert(isvaliddist(t_dist, 0, np.inf))

    assert(isinstance(confined, bool))

    assert(isinstance(wellfield, list))
    for we in wellfield:
        assert(len(we) == 4 and isnumber(we[0]) and isnumber(we[1]) and
               isposnumber(we[2]) and isvaliddist(we[3], -np.inf, np.inf))
    assert(isvalidindex(target, len(wellfield)))

    assert(isinstance(observations, list))
    for ob in observations:
        assert(len(ob) == 4 and isnumber(ob[0]) and isnumber(ob[1]) and
               isnumber(ob[2]) and isposnumber(ob[3]))

    assert(isposnumber(buffer))

    assert(isposnumber(deltax))
    assert(isposnumber(deltay))
    assert(isposnumber(umbra))

    # Setup the constellation of starting points.
    xtarget, ytarget, rtarget = wellfield[target][0:3]

    xy_start = []
    theta_start = 2*np.pi*random.random()

    for i in range(nrays):
        theta = theta_start + i*2*np.pi/nrays
        x = (rtarget + 1) * np.cos(theta) + xtarget
        y = (rtarget + 1) * np.sin(theta) + ytarget
        xy_start.append((x, y))

    # Filter out all of the obs that are too close to any pumping well.
    obs = filter_obs(observations, wellfield, buffer)
    assert(len(obs) > 6)

    # Compute the capture zone for the target well.
    cz = compute_capturezone(
        xy_start, duration, tol, maxstep, nrealizations,
        base, c_dist, p_dist, t_dist, confined,
        wellfield, obs,
        deltax, deltay, umbra)

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
        log.warning('There were no valid realizations.')

    # Plot the wells as o markers.
    xw = [we[0] for we in wellfield]
    yw = [we[1] for we in wellfield]
    plt.plot(xw, yw, 'o', markeredgecolor='k', markerfacecolor='w')

    # Plot the target well as a star marker.
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
            log.info('observation removed: {0}'.format(ob))

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
                log.info('duplicate observation: {0}'.format(obs[k]))
            retained_obs.append((obs[i][0], obs[i][1], num/den, np.sqrt(1/den)))
        else:
            retained_obs.append(obs[i])
        i = j

    log.info('observations: {0} observations retained.'.format(len(retained_obs)))
    for ob in retained_obs:
        log.info('\t active observations: {0}'.format(ob))

    return retained_obs
