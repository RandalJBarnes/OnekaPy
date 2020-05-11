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
create_stochastic_capturezone(
    target, npaths, duration, nrealizations,
    base, c_dist, p_dist, t_dist,
    stochastic_wells, observations,
    spacing, umbra, confined,
    tol, maxstep):
Compute the stochastic capture zone for the target well.

generate_random_variate(arg) :
    Generate a random variate from a dirac (constant), uniform,
    or triangular distribution, depending on the argument tuple.

isdistribution(arg, lb, ub):
    Do the given arguments define a valid distribution?

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
11 May 2020
"""

import logging
import numpy as np
import progressbar

from oneka.capturezone import compute_capturezone
from oneka.model import Model
from oneka.probabilityfield import ProbabilityField


log = logging.getLogger(__name__)


class Error(Exception):
    """Base class for module errors."""
    pass


class DistributionError(Error):
    """Invalid distribution."""
    pass


# ------------------------------------------------------------------------------
def create_stochastic_capturezone(
        target, npaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        stochastic_wells, observations,
        spacing, umbra, confined,
        tol, maxstep):
    """
    Compute the stochastic capture zone for the target well.

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

        (5) Generate and backtrack npaths of particles uniformly
            distributed around the target well.

        (6) Chronicle the particle traces in the ProbabilityField
            grid.
    """

    # Initialize.
    xtarget, ytarget, rtarget = stochastic_wells[target][0:3]
    pfield = ProbabilityField(spacing, spacing, xtarget, ytarget)

    # Initialize the progress bar.
    bar = progressbar.ProgressBar(max_value=nrealizations)
    bar.update(0)

    # Generate and register the capture zone realizations.
    for i in range(nrealizations):

        # Generate a realization of a random well field:
        # fixed loations, random discharges.
        wells = []
        for w in stochastic_wells:
            xw, yw, rw = w[0:3]
            qw = generate_random_variate(w[3])
            wells.append([xw, yw, rw, qw])

        # Generate a realization of random aquifer properties.
        conductivity = generate_random_variate(c_dist)
        porosity = generate_random_variate(p_dist)
        thickness = generate_random_variate(t_dist)

        # Create the model with the random components.
        mo = Model(base, conductivity, porosity, thickness, wells)

        # Generate the realizations for the regional flow ceofficients.
        coef_ev, coef_cov = mo.fit_regional_flow(observations, xtarget, ytarget)
        coef_ev = np.reshape(coef_ev, [6, ])
        mo.coef = np.random.default_rng().multivariate_normal(coef_ev, coef_cov)

        # Log some basic information about the realization.
        recharge = 2*(mo.coef[0] + mo.coef[1])
        log.info('Realization #{0:d}: {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.4e}'
                 .format(i, base, conductivity, porosity, thickness, recharge))
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

        # Update the progress bar.
        bar.update(i+1)

    return pfield    


# ------------------------------------------------------------------------------
def generate_random_variate(arg):
    """
    Generate a random variate from a dirac (constant), uniform, or triangular
    distribution, depending on the argument tuple.

    Arguments
    ---------
    arg : scalar, pair, or triple
        scalar -> constant,
        pair -> (min, max) for a uniform distribution, or
        triple -> (min, mode, max) for a triangular distribution.

    Raises
    ------
    TypeError
        <arg> must be a scalar, pair, or triple.

    Returns
    -------
    value : float
        A random variate from the specified distribution.
    """

    if type(arg) is not tuple:
        value = arg
    elif len(arg) == 2:
        value = np.random.uniform(arg[0], arg[1])
    elif len(arg) == 3:
        value = np.random.triangular(arg[0], arg[1], arg[2])
    else:
        raise DistributionError('<arg> must be a scalar, pair, or triple.')

    return value


# ------------------------------------------------------------------------------
def isdistribution(arg, lb, ub):
    """
    Do the given arguments define a valid distribution?

    Arguments
    ---------
    arg : singleton, pair, or triple
        The characterization of a distribution.

    lb : float
        lower bound

    ub : float
        upper bound

    Notes
    -----
    o   This is a companion function with generate_random_variate. As the
        possible distributions become more complex, we must update this
        function to reflect the changes.
    """

    if isinstance(arg, int) or isinstance(arg, float):
        return lb <= arg <= ub

    if isinstance(arg, tuple) or isinstance(arg, list):
        if len(arg) == 2:
            return lb <= arg[0] <= arg[1] <= ub
        elif len(arg) == 3:
            return lb <= arg[0] <= arg[1] <= arg[2] <= ub

    return False
