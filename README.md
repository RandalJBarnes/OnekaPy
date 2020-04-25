# OnekaPy

## Overview
Campute stochastic capture zones for water supply wells using Monte Carlo simulation.

The model is based upon steady flow in a single, homogeneous, isotropic, flat-lying, aquifer.
The aquifer can be confined, unconfined, or mixed.

## Model
The model allows for the inclusion of a pumping well of interest (the *target* well), a set of
neighboring discharge-specified wells, and regional flow.

The locations of the wells are specified deterministically. The pumping rate of the wells may
be described by probability distributions.

The regional flow component combines uniform infiltration and uniform flow. The mathematical
representation of the discharge potential for the regional flow is:

    Phi(x,y) = A*(x-xo)^2 + B*(y-yo)^2 + C*(x-xo)*(y-yo) + D*x + E*y + F

where (xo, yo) is a local origin -- typically the centroid of the observations.

## Fitting the Model
The strength of the uniform infiltration, as well as the magnitude and direction of the uniform
flow, are determined by minimizing the weighted sum of the squared modelling errors at a set
of specified observations locations (e.g. piezometers).

## Aquifer
The physical characteristics of the aquifer used in this model include:
* the base elevation of the aquifer,
* the hydraulic conductivity of the aquifer,
* the porosity of the aquifer,
* the aquifer thickness.
The hydraulic conductivity, porosity, and thickness may be described by probability
distributions.

In addition to the uncertain aquifer properties and uncertain pumping rates, the model treats
piezometric head measurements as uncertain.  Each head measurement is characterized by an
expected value and a standard deviation. Thus, the fitted infiltration and uniform flow are
also uncertain.

## Deliverables
The output from this model is a capture zone probability grid for the pumping well of interest.
The capture zone probability map is generated using a particle tracking methodology.  The
stochastic nature of this capture zone is caused by the uncertainty of the model parameters.

## How It Works
The following steps are carried out for each stochastic realization.
1. Generate a new set of random well discharges. Thelocations are fixed, but the discharges
may be random.

1. Generate a new set of random aquifer properties: conductivity, porosity, and thickness.

1. Fit the model to the observations to determine the expected value vector and covariance
    matrix of the six regional flow parameters, A - F.

1. Generate the regional flow parameters as a realization of a multivariate normal distribution
    using the expected value vector and covariance matrix generated in preceeding step.

1. Generate and backtrack npaths of particles uniformly distributed around the target well.

1. Chronicle the particle traces in the *ProbabilityField* grid.
