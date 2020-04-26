"""
A basic manufactured example driver file for OnekaPy.

Version
-------
    26 April 2020
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
PROJECTNAME = 'ex_Basic example'

TARGET = 0
MINPATHS = 25
DURATION = 10*365.25
NREALIZATIONS = 100

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
MAXSTEP = 20


# ======================================
# Here is the form of the base call.
# ======================================
def main():
    # Initialize the run.
    start_time = time.time()

    logging.basicConfig(
        filename='..\\logs\\OnekaPy' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.log',
        filemode='w',
        level=logging.INFO)
    log = logging.getLogger(__name__)

    log.info(' Project: {0}'.format(PROJECTNAME))
    log.info(' Run date: {0}'.format(time.asctime()))

    # Call the working function.
    oneka(
        TARGET, MINPATHS, DURATION, NREALIZATIONS,
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
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    p = pstats.Stats(pr)
    p.strip_dirs()
    p.sort_stats(SortKey.TIME).print_stats(10)


# -------------------------------------
if __name__ == "__main__":
    # execute only if run as a script
    main()
