"""


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
    00 May 2020
"""

from datetime import datetime
import logging
import time

from onekapy.oneka import oneka


# ======================================
# Here are the necessary data.
# ======================================
PROJECTNAME = '   example'

TARGET =
NPATHS = 500
DURATION = 10*365.25
NREALIZATIONS =

BASE = 0.0
C_DIST =
P_DIST = 0.25
T_DIST =

BUFFER = 100
SPACING = 10
UMBRA = 20

CONFINED = True
TOL = 1
MAXSTEP = 20

WELLS = [()
    ]

OBSERVATIONS = [()
    ]

# ======================================
# Here is the form of the base call.
# ======================================
if __name__ == "__main__":
    # execute only if run as a script

    # Initialize the run.
    start_time = time.time()

    logging.basicConfig(
        filename='OnekaPy' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.log',
        filemode='w',
        level=logging.INFO)
    log = logging.getLogger(__name__)

    log.info(' Project: {0}'.format(PROJECTNAME))
    log.info(' Run date: {0}'.format(time.asctime()))

    # Call the working function.
    oneka(
        TARGET, NPATHS, DURATION, NREALIZATIONS,
        BASE, C_DIST, P_DIST, T_DIST,
        WELLS, OBSERVATIONS,
        BUFFER, SPACING, UMBRA,
        CONFINED, TOL, MAXSTEP)

    # Shutdown the run.
    elapsedtime = time.time() - start_time
    log.info('Total elapsed time = %.4f seconds' % elapsedtime)
    logging.shutdown()

    print('\n\nTotal elapsed time = %.4f seconds' % elapsedtime)
