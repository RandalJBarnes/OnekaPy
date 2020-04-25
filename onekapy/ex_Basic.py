"""
A basic manufactured example driver file for OnekaPy.

=============
Required Data
=============
target : int
    The index identifying the target well in the wellfield.
    That is, the well for which we will compute a stochastic
    capture zone. This uses python's 0-based indexing.

npaths : int
    The number of rays (starting points for the backtraces) to
    generate uniformly around the target well.

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

buffer : float
    The buffer distance [m] around each well. If an obs falls
    within buffer of any well, it is removed.

spacing : float
    The spacing of the rows and the columns [m] in the square
    ProbabilityField grids.

umbra : float
    The vector-to-raster range [m] when mapping a particle path
    onto the ProbabilityField grids. If a grid node is within
    umbra of a particle path, it is marked as visited.

confined : boolean (optional, default = True)
    True if it is safe to assume that the aquifer is confined
    throughout the domain of interest, False otherwise. This is a
    speed kludge.

tol : float (optional, default = 1)
    The tolerance [m] for the local error when solving the
    backtrace differential equation. This is an inherent
    parameter for an adaptive Runge-Kutta method.

maxstep : float (optional, default = 10)
    The maximum allowed step in space [m] when solving the
    backtrace differential equation. This is a maximum space
    step and NOT a maximum time step.

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

from datetime import datetime
import logging
import time

import cProfile
import pstats
from pstats import SortKey

from oneka import oneka


# ======================================
# Here are the necessary data.
# ======================================
TARGET = 0
NPATHS = 20
DURATION = 10*365.25
NREALIZATIONS = 50

BASE = 0.0
C_DIST = (5.0, 10.0, 25.0)
T_DIST = (10.0, 15.0, 20.0)
P_DIST = (0.20, 0.25)

WELLFIELD = [
    (2250, 2250, 0.25, (600, 750, 900)),
    (1750, 2750, 0.25, (600, 750, 900))
    ]

OBSERVATIONS = [
    (1000, 1000, 100, 2),
    (1000, 1500, 105, 2),
    (1000, 2000, 110, 2),
    (1000, 2500, 115, 2),
    (1000, 3000, 120, 2),
    (1500, 1000, 95, 2),
    (1500, 1500, 100, 2),
    (1500, 2000, 105, 2),
    (1500, 2500, 110, 2),
    (1500, 3000, 115, 2),
    (2000, 1000, 90, 2),
    (2000, 1500, 95, 2),
    (2000, 2000, 100, 2),
    (2000, 2500, 105, 2),
    (2000, 3000, 110, 2),
    (2500, 1000, 85, 2),
    (2500, 1500, 90, 2),
    (2500, 2000, 95, 2),
    (2500, 2500, 100, 2),
    (2500, 3000, 105, 2),
    (3000, 1000, 80, 2),
    (3000, 1500, 85, 2),
    (3000, 2000, 90, 2),
    (3000, 2500, 95, 2),
    (3000, 3000, 100, 2)
    ]

BUFFER = 100
SPACING = 10
UMBRA = 10

CONFINED = True
TOL = 1
MAXSTEP = 50


# ======================================
# Here is the form of the base call.
# ======================================
def main():
    # Initialize the run.
    start_time = time.time()

    logging.basicConfig(
        filename='..\\logs\\Basic' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.log',
        filemode='w',
        level=logging.INFO)
    log = logging.getLogger(__name__)

    # Call the working function.
    oneka(
        TARGET, NPATHS, DURATION, NREALIZATIONS,
        BASE, C_DIST, P_DIST, T_DIST,
        WELLFIELD, OBSERVATIONS,
        BUFFER, SPACING, UMBRA,
        CONFINED, TOL, MAXSTEP)

    # Shutdown the run.
    elapsedtime = time.time() - start_time
    log.info('Total elapsed time = %.4f seconds' % elapsedtime)
    logging.shutdown()

    print('\n\nTotal elapsed time = %.4f seconds' % elapsedtime)


# -------------------------------------
def profile_me():
    cProfile.run('ex_Basic.main()', 'restats')
    p = pstats.Stats('restats')
    p.strip_dirs()
    p.sort_stats(SortKey.TIME).print_stats(20)


# -------------------------------------
if __name__ == "__main__":
    # execute only if run as a script
    main()
