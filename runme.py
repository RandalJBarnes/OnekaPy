"""
A simple driver for OnekaPy.

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
25 July 2020

"""
from datetime import datetime
import logging
import time

import cProfile
import pstats
from pstats import SortKey

from oneka.oneka import oneka

import importlib


# ------------------------------------------------------------------------------
def main(module_name):
    try:
        m = importlib.import_module(module_name)
        print('Running: {0}'.format(m.PROJECTNAME))
    except ModuleNotFoundError:
        print('ModuleNotFoundError: No module named {0}'.format(module_name))
        return

    # Initialize the run.
    start_time = time.time()
    RUNTIME = time.asctime()

    # Setup the logging.
    logger = logging.getLogger('Oneka')
    logger.setLevel(logging.INFO)

    # Create file handler which logs all messages.
    fname='logs\\Oneka' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.log'
    fh = logging.FileHandler(filename=fname)
    fh.setLevel(logging.INFO)

    # Add the handlers to logger
    logger.addHandler(fh)

    log = logging.getLogger('Oneka')

    # Call the working function.
    oneka(
        m.PROJECTNAME, RUNTIME,
        m.TARGET, m.NPATHS, m.DURATION, m.NREALIZATIONS,
        m.BASE, m.C_DIST, m.P_DIST, m.T_DIST,
        m.WELLS, m.OBSERVATIONS,
        m.BUFFER, m.SPACING, m.UMBRA, m.SMOOTH,
        m.CONFINED, m.TOL, m.MAXSTEP)

    # Shutdown the run.
    elapsedtime = time.time() - start_time
    log.info('\n')
    log.info('Total elapsed time = %.4f seconds' % elapsedtime)
    logging.shutdown()

    print('\n\nTotal elapsed time = %.4f seconds' % elapsedtime)


# ------------------------------------------------------------------------------
def profile_me():
    pr = cProfile.Profile()
    pr.enable()
    main(module_name)
    pr.disable()

    p = pstats.Stats(pr)
    p.strip_dirs()
    p.sort_stats(SortKey.TIME).print_stats(10)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    module_name = input('Enter the data module name (without .py, e.g. data.basic): ')
    main(module_name)

    print ('\\\\\\\\\\\\\\\\\\ DONE //////////////////')
