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
    compute_capturezone(
            xy_start, duration, tol, maxstep,  nrealizations,
            base, c_dist, p_dist, t_dist,
            wellfield, observations,
            spacing, umbra, confined) :
        Run backtraces from all points xy_start using multiple
        realizations of a random model. The results of the
        realizations a collated and returned as a ProbabilityField.

    generate_random_variate(arg) :
        Generate a random variate from a dirac (constant), uniform,
        or triangular distribution, depending on the argument tuple.

    filter_obs(obs, wells, buffer):
        Partition the obs into retained and removed. An observation
        is removed if it is within buffer of a well.

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
    26 April 2020
"""

import logging
import numpy as np
import progressbar

from model import Model, AquiferError
from probabilityfield import ProbabilityField
from utility import isnumber, isposnumber, isposint, isvalidindex, isvaliddist


log = logging.getLogger(__name__)


class Error(Exception):
    """Base class for module errors."""
    pass


class DistributionError(Error):
    """Invalid distribution."""
    pass


# -------------------------------------
def compute_capturezone(
        target, minpaths, duration, nrealizations,
        base, c_dist, p_dist, t_dist,
        wellfield, observations,
        spacing, umbra, confined, tol, maxstep):
    """
    Run backtraces from all points xy_start using multiple realizations of a
    random model. The results of the realizations a collated and returned as
    a ProbabilityField.

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
        The duration of the capture zone [d]; e.g. a ten year capture zone
        will have a duration = 10*365.25.

    tol : float
        The tolerance [m] for the local error when solving the backtrace
        differential equation. This is an inherent parameter for an
        adaptive Runge-Kutta method.

    maxstep : float
        The maximum allowed step in space [m] when solving the backtrace
        differential equation. This is NOT the maximum time step.

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

    spacing : float
        The spacing of the rows and the columns [m] in the square
        ProbabilityField grids.

    umbra : float
        The vector-to-raster range [m] when mapping a particle path onto
        the ProbabilityField grids. If a grid node is within umbra of a
        particle path, it is marked as visited.

    confined : boolean
        True if it is safe to assume that the aquifer is confined
        throughout the domain of interest, False otherwise. This is a
        speed kludge.

    Returns
    -------
    capturezone : ProbabilityField
        The probability filed resulting from the stochastic simulations.

    Notes
    -----
    o   This is a time-consuming function.

    o   We may be able to speed this up by parallelizing the backtrace
        calls.
    """

    # Validate the arguments.
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

    assert(isposnumber(spacing))
    assert(isposnumber(umbra))
    assert(isinstance(confined, bool))
    assert(isposnumber(tol))
    assert(isposnumber(maxstep))

    # Local constants.
    STEPAWAY = 1

    # Initialize the progress bar.
    bar = progressbar.ProgressBar(max_value=nrealizations)

    # Setup the constellation of starting points.
    xtarget, ytarget, rtarget = wellfield[target][0:3]

    xy_start = []
    theta = np.linspace(0, 2*np.pi, minpaths+1)[0:-1]
    for a in theta:
        x = (rtarget + STEPAWAY) * np.cos(a) + xtarget
        y = (rtarget + STEPAWAY) * np.sin(a) + ytarget
        xy_start.append((x, y))

    # Initialize the probability field.
    capturezone = ProbabilityField(spacing, spacing)

    # Generate and register the capture zone realizations.
    for i in range(nrealizations):

        # Generate a realization of a random well field: fixed loations, random discharges.
        wells = []
        for w in wellfield:
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
        coef_ev, coef_cov = mo.fit_coefficients(observations)
        mo.coef = np.random.default_rng().multivariate_normal(coef_ev, coef_cov)

        # Log some basic information about the realization.
        recharge = 2*(mo.coef[0] + mo.coef[1])
        log.info(' Realization #{0:d}: {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.4e}'
                 .format(i, base, conductivity, porosity, thickness, recharge))
        log.info('     coef ev:  ({0:+.4e}, {1:+.4e}, {2:+.4e}, {3:+.4e}, {4:+.4e}, {5:+.4e})'
                 .format(coef_ev[0], coef_ev[1], coef_ev[2], coef_ev[3], coef_ev[4], coef_ev[5]))
        log.info('     coef rng: ({0:+.4e}, {1:+.4e}, {2:+.4e}, {3:+.4e}, {4:+.4e}, {5:+.4e})'
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

        # Generate the base backtraces.
        xy_end = []
        for xys in xy_start:
            vertices, length = compute_backtrace(xys, duration, tol, maxstep, feval)
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            capturezone.rasterize(x, y, umbra)
            xy_end.append(vertices[-1])

        # Generate infill backtraces as needed.
        xy_infill = []



        # Register the backtraces.
        capturezone.register(1)

        # Update the progress bar.
        bar.update(i+1)

    return capturezone


# -------------------------------------
def compute_backtrace(xys, duration, tol, maxstep, feval):
    """
    Compute a single backtrace using the Dormand-Prince adaptive
    Runge-Kutta explicit solver.

    Parameters
    ----------
    xys : (float, float)
        The (x-coordinate, y-coordinate) [m] of the starting point.

    duration : float
        The duration of the capture zone [d]; e.g. a ten year capture zone
        will have a duration = 10*365.25.

    tol : float
        The tolerance [m] for the local error when solving the backtrace
        differential equation. This is an inherent parameter for an
        adaptive Runge-Kutta method.

    maxstep : float
        The maximum allowed step in space [m] when solving the backtrace
        differential equation. This is NOT the maximum time step.

    feval : function
        The backtracing velocity function.

    Returns
    -------
    vertices : list
        The list of (x, y) tuples containing the vertices of the
        backtrace path.

    length : float
        The total length of the backtrace path.

    Notes
    -----
    o   This function use the Dormand-Prince implementation of an
        adaptive Runge-Kutta algorithm to solve the backtrace ode.
        See, for example,

            https://en.wikipedia.org/wiki/Dormand-Prince_method

    o   Much of the Dormand-Prince Runge-Kutta code is translated
        and adpated from the MATLAB implementation given by
        www.mathtools.com.

    o   This implementation is unique is two significant ways:

        --  The feval is NOT a function of time, as we are working
            with steadystate flow.

        --  The maximum time step is governed by a maximum space step.

    o   The Dormand-Prince Runge-Kutta is currently the default method
        in MATLAB's ode45, as well as SciPy's ode integration library.

    References
    ----------
    o   Dormand, J. R.; Prince, P. J. (1980), A family of embedded
        Runge-Kutta formulae, Journal of Computational and Applied
        Mathematics, 6 (1): 19–26, doi:10.1016/0771-050X(80)90013-3.

    o   Dormand, John R. (1996), Numerical Methods for Differential
        Equations: A Computational Approach, Boca Raton: CRC Press,
        pp. 82–84, ISBN 0-8493-9433-3.

    o   https://www.mathstools.com/section/main/dormand_prince_method
    """

    # Local constants.
    EPS = np.finfo(float).eps

    # Dormand-Prince constants for an adaptive Runge-Kutta integrator.
    a2 = np.array([1/5])
    a3 = np.array([3/40, 9/40])
    a4 = np.array([44/45, -56/15, 32/9])
    a5 = np.array([19372/6561, -25360/2187, 64448/6561, -212/729])
    a6 = np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656])
    a7 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])

    e = np.array([71/57600, -1/40, -71/16695, 71/1920, -17253/339200, 22/525])

    # Initialize: the starting point is the first vertex.
    t = 0
    dt = 0.1                # This is an arbitrary choice.
    length = 0

    vertices = [xys]
    xy = np.array([xys[0], xys[1]])

    try:
        k1 = feval(xy)

        while (t < duration):
            # Do not step past duration.
            if (t + dt > duration):
                dt = duration - t

            # The 5th order Runge-Kutta with Dormand-Prince constants.
            k2 = feval(xy + dt*(a2[0]*k1))
            k3 = feval(xy + dt*(a3[0]*k1 + a3[1]*k2))
            k4 = feval(xy + dt*(a4[0]*k1 + a4[1]*k2 + a4[2]*k3))
            k5 = feval(xy + dt*(a5[0]*k1 + a5[1]*k2 + a5[2]*k3 + a5[3]*k4))
            k6 = feval(xy + dt*(a6[0]*k1 + a6[1]*k2 + a6[2]*k3 + a6[3]*k4 + a6[4]*k5))

            xyt = xy + dt*(a7[0]*k1 + a7[2]*k3 + a7[3]*k4 + a7[4]*k5 + a7[5]*k6)

            # Control time step for maximum space step and maximum estimated error.
            k2 = feval(xyt)
            est = np.linalg.norm(dt*(e[0]*k1 + e[1]*k2 + e[2]*k3 + e[3]*k4 +
                                     e[4]*k5 + e[5]*k6), np.inf)
            ds = np.linalg.norm(xyt - xy)

            if (est < tol) and (ds < maxstep):
                t = t + dt
                k1 = k2
                xy = xyt
                vertices.append(xy)
                length += ds

            dt = 0.9 * min((tol/(est + EPS))**(1/5), maxstep/(ds + EPS), 10) * dt

    except AquiferError:
        log.warning(' trace terminated prematurely at t = {0:.2f} < duration.'.format(t))

    finally:
        return (vertices, length)


# -------------------------------------
def generate_random_variate(arg):
    """
    Generate a random variate from a dirac (constant), uniform, or triangular
    distribution, depending on the argument tuple.

    Parameters
    ----------
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
