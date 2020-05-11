"""
A small set of visualization functions.

Functions
---------
contour_head(mo, xmin, xmax, ymin, ymax, nrows, ncols):
    Compute and plot the filled-contour map for the head, 
    as defined by model mo.

contour_potential(mo, xmin, xmax, ymin, ymax, nrows, ncols)
    Compute and plot the filled-contour map for the discharge
    potential, as defined by model mo.

create_probability_plot(target, stochastic_wells, obs, pf, smooth):
    Create the visible filled-contour plot for the stochastic capture
    zone. 

create_impact_plot(spacing, pf):
    Create the visible impact plot (area vs. probability of capture) for the 
    probability field.        

quick_capture_zone(mo, we, nrays, nyears, maxstep, fmt)
    Compute and plot a capture zone for Well we using Model mo.

Raises
------
Error
CaptureZoneError

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

import io
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import oneka.model


log = logging.getLogger('Oneka')


class Error(Exception):
    """Base class for module errors."""
    pass


class CaptureZoneError(Error):
    """Capture zone is unfilled."""
    pass


# ------------------------------------------------------------------------------
def contour_head(mo, xmin, xmax, ymin, ymax, nrows, ncols):
    """
    Create a visible filled contour map of the piezometric head, as defined 
    by model mo.

    Arguments
    ---------
    mo : oneka-style model
        The defining model.

    xmin : float
        The left edge of the map. xmin < xmax.
        
    xmax : float
        The right edge of the map. xmax > xmin.

    ymin : float
        The bottom edge of the map. ymin < ymax.

    ymax : float
        The top edge of the map. ymax > ymin.

    nrows : int
        The number of grid rows. nrow > 1.
    
    ncols : int
        The number of grid columns. ncols > 1.

    Returns
    -------
    None.
    """

    x = np.linspace(xmin, xmax, ncols)
    y = np.linspace(ymin, ymax, nrows)

    grid = np.zeros((nrows, ncols), dtype=np.double)

    for i in range(nrows):
        for j in range(ncols):
            try:
                grid[i, j] = mo.compute_head(x[j], y[i])
            except onekapy.model.AquiferError:
                grid[i, j] = np.nan

    plt.contourf(x, y, grid, cmap='bwr')
    plt.colorbar()


# ------------------------------------------------------------------------------
def contour_potential(mo, xmin, xmax, ymin, ymax, nrows, ncols):
    """
    Create a visible filled contour map of the discharge potential, as defined 
    by model mo.

    Arguments
    ---------
    mo : oneka-style model
        The defining model.

    xmin : float
        The left edge of the map. xmin < xmax.
        
    xmax : float
        The right edge of the map. xmax > xmin.

    ymin : float
        The bottom edge of the map. ymin < ymax.

    ymax : float
        The top edge of the map. ymax > ymin.

    nrows : int
        The number of grid rows. nrow > 1.
    
    ncols : int
        The number of grid columns. ncols > 1.

    Returns
    -------
    None.
    """

    x = np.linspace(xmin, xmax, ncols)
    y = np.linspace(ymin, ymax, nrows)

    grid = np.zeros((nrows, ncols), dtype=np.double)

    for i in range(nrows):
        for j in range(ncols):
            try:
                grid[i, j] = mo.compute_potential(x[j], y[i])
            except oneka.model.AquiferError:
                grid[i, j] = np.nan

    plt.contourf(x, y, grid, cmap='bwr')
    plt.colorbar()


# ------------------------------------------------------------------------------
def create_probability_plot(target, stochastic_wells, obs, pf, smooth=0):
    """
    Create the visible filled-contour plot for the stochastic capture zone.

    Arguments
    ---------
    target : int
        The index identifying the target well in the stochastic_wells.
        That is, the well for which we will compute a stochastic
        capture zone. This uses python's 0-based indexing.

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

    obs : list of observation tuples.
        An observation tuple contains four values: (x, y, z_ev, z_std), where
            x : float
                The x-coordinate of the observation [m].
            y : float
                The y-coordinate of the observation [m].
            z_ev : float
                The expected value of the observed static water level elevation [m].
            z_std : float
                The standard deviation of the observed static water level elevation [m].

    pf : ProbabilityField
        The auto-expanding, axis-aligned, grid-based probability field.

    smooth : scalar, optional
        The nominal 'standard deviation' for the Gaussian kernel smoother
        (scipy.ndimage.gaussian_filter) to be applied to the probability
        contours. The units are in grids. Default = 0.

    Returns
    -------
    None.

    Raises
    ------
    CaptureZoneError
    """

    # Make the probability contour plot.
    plt.figure()
    plt.axis('equal')

    if pf.total_weight <= 0:
        log.warning(' There were no valid realizations.')
        raise CaptureZoneError('Empty capture zone')

    X = np.linspace(pf.xmin, pf.xmax, pf.ncols)
    Y = np.linspace(pf.ymin, pf.ymax, pf.nrows)
    Z = pf.pgrid/pf.total_weight

    if smooth > 0:
        Z = scipy.ndimage.gaussian_filter(Z, smooth, mode='constant', cval=0.0)
        
    plt.contourf(X, Y, Z, np.linspace(0, 1, 11), cmap='tab10')
    plt.colorbar(ticks=np.linspace(0, 1, 11))
    plt.contour(X, Y, Z, np.linspace(0.1, 0.9, 9), colors=['black'])

    plt.xlabel('UTM Easting [m]')
    plt.ylabel('UTM Northing [m]')
    plt.grid(True)

    # Plot the stochastic_wells as o markers.
    xw = [we[0] for we in stochastic_wells]
    yw = [we[1] for we in stochastic_wells]
    plt.plot(xw, yw, 'o', markeredgecolor='k', markerfacecolor='w')

    # Plot the target well as a star marker.
    xtarget, ytarget = stochastic_wells[target][0:2]
    plt.plot(xtarget, ytarget, '*', markeredgecolor='k', markerfacecolor='w', markersize=12)

    # Plot the retained observations as fat + markers.
    xo = [ob[0] for ob in obs]
    yo = [ob[1] for ob in obs]
    plt.plot(xo, yo, 'P', markeredgecolor='k', markerfacecolor='w')


# ------------------------------------------------------------------------------
def create_impact_plot(spacing, pf):
    """
    Create the visible impact plot (area vs. probability of capture) for the 
    probability field.

    Arguments
    ---------
    spacing : float, optional
        The spacing of the rows and the columns [m] in the square
        ProbabilityField grids.

    pf : ProbabilityField
        The auto-expanding, axis-aligned, grid-based probability field.

    Returns
    -------
    (pr, area) : pair of ndarray, shape=(n,), dtype=float
        Probabilty of capture exceeds "pr" over an "area".

    Raises
    ------
    CaptureZoneError
    """

    if pf.total_weight <= 0:
        log.warning(' There were no valid realizations.')
        raise CaptureZoneError('Empty capture zone')

    pr = np.flip(np.sort(pf.pgrid/pf.total_weight, axis=None))
    n = pr.shape[0]
    area = (np.arange(n)+1)*spacing**2
    i = max(0, np.argmax(pr<1)-1)
    j = min(n, np.argmax(pr==0)+1)

    plt.figure()
    plt.semilogy(pr[i:j], area[i:j]//4046.86, '.')                  # 4046.86 m^2/acre
    plt.xticks(np.linspace(0,1,11))

    plt.xlabel('Pr(capture) exceeds')
    plt.ylabel('Area [acres]')
    plt.xlim(left=0.0, right=1.0)
    plt.grid(True, which='both', axis='both')

    return (pr, area)

# ------------------------------------------------------------------------------
def quick_capture_zone(mo, we, nrays, nyears, maxstep, fmt):
    """
    Compute and plot a capture zone for Well we using Model mo.

    Arguments
    ---------
    mo : Model
        The driving model for the capture zone.

    we : Well
        The well for which the capture zone is computed

    nrays : int
        The number of uniformly distributed rays to trace out from the well.

    nyears : double
        The number years to run the back trace.

    maxstep : float
        The solve_ivp max_step parameter.

    fmt : string
        The format string for the backtrace plot.

    Returns
    -------
    None.
    """

    radius = we.radius + 1
    xc = we.x
    yc = we.y

    for theta in np.linspace(0, 2*np.pi, nrays):
        xo = radius*np.cos(theta) + xc
        yo = radius*np.sin(theta) + yc

        try:
            sol = mo.compute_backtrack(xo, yo, nyears*365, maxstep)

            for year in np.arange(0, nyears):
                idx = np.logical_and(year*365 < sol.t, sol.t < (year+1)*365)

                if (year % 2) == 0:
                    plt.plot(sol.y[0, idx], sol.y[1, idx], fmt, linewidth=4)
                else:
                    plt.plot(sol.y[0, idx], sol.y[1, idx], '-k')

        except oneka.model.AquiferError:
            print(f"Aquifer error (e.g. dry) for theta = {theta:.3f}")
            continue
