[![Open in Leap IDE](
  https://cdn-assets.cloud.dwavesys.com/shared/latest/badges/leapide.svg)](
  https://ide.dwavesys.io/#https://github.com/dwave-examples/airline-hubs)
[![Linux/Mac/Windows build status](
  https://circleci.com/gh/dwave-examples/airline-hubs.svg?style=shield)](
  https://circleci.com/gh/dwave-examples/airline-hubs)

# Airline Hub Locations

A challenging optimization problem in the airline industry is determining which
airports should be hub locations for an airline. In this demo, we show how to
formulate this problem as a Constrained Quadratic Model (CQM) and use a hybrid
solver to optimize and find feasible solutions.

The goal for this problem is to minimize costs for the airline, while providing
transportation for all city pairs in demand by passengers.

## Running the Demo

To run the demo, type

`python demo.py`

The program will select 3 hubs out of a list of 25 different airports. The
information in the files `flow.csv` and `cost.csv` is used to determine the
optimal selection of hub airports, based on passenger flow and airline costs
found in these files.

A GIF will be produced that illustrates the feasible results found by the
hybrid solver, as shown below.

![output](readme_imgs/airline-hubs.gif)

## Formulating the Problem

In order to minimize costs, the airlines must consider several factors.

 1. The cost to operate a non-stop route (sometimes referred to as a leg). Note
that costs between hub airports are discounted.  
 2. The passenger demand for each leg (called the flow).

The demo reads in both of these factors from provided data files (`cost.csv`
and `flow.csv`, respectively).

We have several additional constraints that must be satisfied in order for a
route map to be feasible.

 1. Every leg must connect to a hub.  
 2. Passengers only connect at hub airports.  
 3. Each airport gets assigned exactly one hub.
 4. Only p hubs total.

The first two constraints ensure that any connecting airports are hubs.

## Building the Model

For a list of `n` cities, we define `n^2` binary variables. A binary variable
`(i,j)` is equal to 1 if city `i` is assigned to a hub at city `j`, and is
equal to 0 otherwise. In the real-world scenario, this translates to a flight
route / leg from city `i` to city `j`. In particular, we assign `(i,i) = 1` if
and only if city `i` is a hub.

With these variables, we can build the quadratic model for the problem. Our
constraints translate to the following expressions in terms of the binary
variables defined.

 1. If `(i,j) = 1` and `i != j`, then `(j,j) = 1`. In other words, if city `i`
 connects to city `j`, then city `j` must be a hub.
 2. For each city `i`, exactly one city `j` exists with `(i,j) = 1`.
 3. Exactly `p` cities `i` exist with `(i,i) = 1`.

Formulating the objective with these binary variables is a bit more complex,
and the interested reader is referred to the paper referenced below.

## References

O'Kelly, Morton E. "A quadratic integer program for the location of interacting
hub facilities." European journal of operational research 32.3 (1987): 393-404.
