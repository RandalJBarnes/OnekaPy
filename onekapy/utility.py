"""
Implements some simple unitility functions.

Classes
-------
    None.

Functions
---------
    isnumber(arg)
    isposnumber(arg)
    isposint(arg)
    isvalidindex(arg, n)
    isvaliddist(arg, lb, ub)

Notes
-----


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
    24 April 2020
"""


def isnumber(arg):
    return isinstance(arg, int) or isinstance(arg, float)


def isposnumber(arg):
    return isnumber(arg) and (arg > 0)


def isposint(arg):
    return isinstance(arg, int) and (arg > 0)


def isvalidindex(arg, n):
    return isinstance(arg, int) and (0 <= arg < n)


def isvaliddist(arg, lb, ub):
    if isinstance(arg, int) or isinstance(arg, float):
        return lb <= arg <= ub

    if isinstance(arg, tuple) or isinstance(arg, list):
        if len(arg) == 2:
            return lb <= arg[0] <= arg[1] <= ub
        elif len(arg) == 3:
            return lb <= arg[0] <= arg[1] <= arg[2] <= ub

    return False
