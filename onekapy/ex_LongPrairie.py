"""
Long Prairie example driver file for OnekaPy.

Version
-------
    27 April 2020
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
PROJECTNAME = 'ex_LongPrairie example'

TARGET = 0
MINPATHS = 80
DURATION = 10*365.25
NREALIZATIONS = 1

BASE = 0.0
C_DIST = (1.0, 9.0, 25.0)
T_DIST = (10.0, 12.6, 15.0)
P_DIST = (0.15, 0.20, 0.25)

BUFFER = 100
SPACING = 10
UMBRA = 20

CONFINED = True
TOL = 1
MAXSTEP = 20

WELLFIELD = [
    (355731, 5091141, 0.1524, 1097.56),       # relaeid 0000542955 !!TARGET WELL!!
    (356276, 5092259, 0.0762,   27.98),       # relaeid 0000123593
    (356276, 5092259, 0.0762,  100.53),       # relaeid 0000123593
    (355796, 5091952, 0.0762,   12.44),       # relaeid 0000137314
    (355800, 5092454, 0.0635,    8.29),       # relaeid 0000477507
    (356366, 5092269, 0.0762,   89.13),       # relaeid 0000600268
    (356366, 5092269, 0.0762,   24.87),       # relaeid 0000600268
    (355674, 5091161, 0.2032,  962.83),       # relaeid 0000550068
    (355306, 5091163, 0.0508,    8.52),       # relaeid 0000640683
    (356467, 5092921, 0.1016,  274.65),       # relaeid 0000480438
    (355736, 5091176, 0.2032,  563.81)        # relaeid 0000699113
    ]

OBSERVATIONS = [
    (353676, 5092700, 400.2, 2.0),
    (355056, 5092739, 385.9, 2.0),
    (353996, 5092723, 394.6, 2.0),
    (354040, 5092723, 393.9, 2.0),
    (353745, 5092665, 402.2, 2.0),
    (355474, 5092596, 391.2, 2.0),
    (354098, 5092694, 395.3, 2.0),
    (353957, 5092687, 396.6, 2.0),
    (353512, 5092499, 401.2, 2.0),
    (355494, 5092570, 393.5, 2.0),
    (355810, 5092480, 394.2, 2.0),
    (357436, 5092465, 403.7, 2.0),
    (354593, 5092404, 392.9, 2.0),
    (354613, 5092340, 389.8, 2.0),
    (356299, 5092242, 394.8, 2.0),
    (356413, 5092268, 395.0, 2.0),
    (355280, 5092210, 393.8, 2.0),
    (356896, 5092108, 394.4, 2.0),
    (356984, 5092143, 400.4, 2.0),
    (356956, 5092055, 393.3, 2.0),
    (353664, 5092008, 402.9, 2.0),
    (355773, 5091947, 397.7, 2.0),
    (354068, 5091785, 398.0, 2.0),
    (357258, 5091739, 401.7, 2.0),
    (354322, 5091700, 403.9, 2.0),
    (357108, 5091713, 397.3, 2.0),
    (357055, 5091589, 402.3, 2.0),
    (357269, 5091528, 403.6, 2.0),
    (357134, 5091500, 401.9, 2.0),
    (354252, 5091329, 400.6, 2.0),
    (357348, 5091262, 398.4, 2.0),
    (354444, 5091299, 397.6, 2.0),
    (355731, 5091141, 394.8, 2.0),
    (357970, 5091139, 404.4, 2.0),
    (355736, 5091176, 394.6, 2.0),
    (354297, 5091170, 399.2, 2.0),
    (355306, 5091163, 395.1, 2.0),
    (355674, 5091161, 394.4, 2.0),
    (357429, 5091234, 399.9, 2.0),
    (356099, 5090755, 393.4, 2.0),
    (356079, 5090655, 392.7, 2.0),
    (356108, 5090698, 393.7, 2.0),
    (356108, 5090700, 397.1, 2.0),
    (356088, 5090659, 392.8, 2.0),
    (356747, 5090656, 397.9, 2.0),
    (356253, 5090481, 399.0, 2.0),
    (357783, 5090485, 401.0, 2.0),
    (355945, 5090333, 393.2, 2.0),
    (356308, 5090204, 399.9, 2.0),
    (358419, 5090200, 399.5, 2.0),
    (354335, 5090043, 399.9, 2.0),
    (356466, 5090007, 401.4, 2.0),
    (353569, 5089997, 415.7, 2.0),
    (356274, 5090073, 398.8, 2.0),
    (356502, 5089983, 394.6, 2.0),
    (355175, 5089923, 394.3, 2.0),
    (355324, 5089757, 396.5, 2.0),
    (355379, 5089050, 392.4, 2.0),
    (355326, 5089115, 396.7, 2.0),
    (354864, 5088290, 397.2, 2.0),
    (355249, 5088248, 397.1, 2.0)
    ]


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
