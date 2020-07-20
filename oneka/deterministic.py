"""
Defines and implements a single well capture zone.

Classes
-------
None

Exceptions
----------
Error
DistributionError

Functions
---------
create_deterministic_capturezone(
    target, npaths, duration,
    base, c_dist, p_dist, t_dist,
    stochastic_wells, observations,
    spacing, umbra, confined,
    tol, maxstep):
Compute the DETERMINISTIC capture zone for the target well.

Notes
-----
o   We may be able to parallelize the tracking particles to speed up
    the overall execution.

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
20 July 2020

"""
import logging
import numpy as np

from oneka.capturezone import compute_capturezone
from oneka.model import Model
from oneka.probabilityfield import ProbabilityField
from oneka.stochastic import compute_variate_mean

log = logging.getLogger(__name__)


class Error(Exception):
    """Base class for module errors."""
    pass


class DistributionError(Error):
    """Invalid distribution."""
    pass


# ------------------------------------------------------------------------------
def create_deterministic_capturezone(
        target, npaths, duration,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        spacing, umbra, confined,
        tol, maxstep):
    """
    Compute the deterministic capture zone for the target well.

    The detreministic capture zone is computed by using the mean of each of the
    parameters: aquifer properties, pumping rates, and the six fitted Oneka
    coefficients.

    Arguments
    ---------
    target : int
        The index identifying the target well in the stochastic_wells.
        That is, the well for which we will compute a stochastic
        capture zone. This uses python's 0-based indexing.

    npaths : int
        The minimum number of paths (starting points for the backtraces)
        to generate uniformly around the target well. 0 < npaths.

    duration : float
        The duration of the capture zone [d]; e.g. a ten year capture zone
        will have a duration = 10*365.25. 0 < duration.

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

    spacing : float, optional
        The spacing of the rows and the columns [m] in the square
        ProbabilityField grids. Default is 10.

    umbra : float
        The vector-to-raster range [m] when mapping a particle path onto
        the ProbabilityField grids. If a grid node is within umbra of a
        particle path, it is marked as visited.

    confined : boolean
        True if it is safe to assume that the aquifer is confined
        throughout the domain of interest, False otherwise. This is a
        speed kludge.

    tol : float
        The tolerance [m] for the local error when solving the backtrace
        differential equation. This is an inherent parameter for an
        adaptive Runge-Kutta method.

    maxstep : float
        The maximum allowed step in space [m] when solving the backtrace
        differential equation. This is NOT the maximum time step.

    Returns
    -------
    pfield : ProbabilityField
        The auto-expanding, axis-aligned, grid-based probability field.

    Notes
    -----
    o   This is a very time-consuming function.

    o   We may be able to speed this up by parallelizing the backtrace
        calls.

    """

    # Initialize.
    xtarget, ytarget, rtarget = stochastic_wells[target][0:3]
    pfield = ProbabilityField(spacing, spacing, xtarget, ytarget)

    # Generate a realization of a random well field:
    # fixed loations, random discharges.
    wells = []
    for w in stochastic_wells:
        xw, yw, rw = w[0:3]
        qw = compute_variate_mean(w[3])
        wells.append([xw, yw, rw, qw])

    # Generate a realization of random aquifer properties.
    conductivity = compute_variate_mean(c_dist)
    porosity = compute_variate_mean(p_dist)
    thickness = compute_variate_mean(t_dist)

    # Create the model with the random components.
    mo = Model(base, conductivity, porosity, thickness, wells)

    # Generate the realizations for the regional flow coefficients.
    coef_ev, coef_cov = mo.fit_regional_flow(observations, xtarget, ytarget)
    coef_ev = np.reshape(coef_ev, [6, ])
    mo.coef = np.random.default_rng().multivariate_normal(coef_ev, coef_cov)

    # Log some basic information about the realization.
    recharge = 2*(mo.coef[0] + mo.coef[1])
    log.info('Deterministic: {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.4e}'
             .format(base, conductivity, porosity, thickness, recharge))
    log.info('    coef ev:  ({0:+.4e}, {1:+.4e}, {2:+.4e}, {3:+.4e}, {4:+.4e}, {5:+.4e})'
             .format(coef_ev[0], coef_ev[1], coef_ev[2], coef_ev[3], coef_ev[4], coef_ev[5]))
    log.info('    coef rng: ({0:+.4e}, {1:+.4e}, {2:+.4e}, {3:+.4e}, {4:+.4e}, {5:+.4e})'
             .format(mo.coef[0], mo.coef[1], mo.coef[2], mo.coef[3], mo.coef[4], mo.coef[5]))

    # Define the local backtracing velocity function.
    if confined:
        def feval(xy):
            Vx, Vy = mo.compute_velocity_confined(xy[0], xy[1])
            return np.array([-Vx, -Vy])
    else:
        def feval(xy):
            Vx, Vy = mo.compute_velocity(xy[0], xy[1])
            return np.array([-Vx, -Vy])

    # Compute and register the capture zone for the specific (now
    # deterministic) realization of the stochastic problem.
    compute_capturezone(xtarget, ytarget, rtarget, npaths, duration,
                        pfield, umbra, 1.0, tol, maxstep, feval)

    return pfield
