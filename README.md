# Gaussian Process Cross Correlation

Julia implementation of Gaussian Process Cross Correlation model.


## How to use

Apart from cloning, an easy way of using the package is the following:

1 - Add the registry [CollaborativeAstronomyJulia](https://github.com/ngiann/CollaborativeAstronomyJulia).

2 - Switch into "package mode" with ```]``` and add the package with
```
add GPCC
```

The package exposes four functions that may be of interest to the user: `gpcc`, `simulatedata`, `getprobabilities` and `performcv`.
These functions can be queried in help mode at the Julia REPL. 

In case you are installing ProbabilisticFluxVariationGradient to an existing Julia environment, there is a chance one may run into dependency problems that prevent installation. In this case, it is advisable to work in a new environment. That is

```
mkdir("myGPCC")
cd("myGPCC")
# press `]` to enter package mode:
(@v1.6) pkg> activate .
```
and use this environment for installing and working with the package.
Having exited Julia, one can enter the created environment again by simply starting Julia in the respective folder and using `activate .` in package mode.


## Experimental results

See [here](https://github.com/ngiann/GPCCExperiments) for experimental results.


## Example using a real dataset

We use the package [GPCCVirialDatasets](https://github.com/ngiann/GPCCVirialDatasets.jl) to access real observations.


```
using GPCC, GPCCVirialDatasets

# load data
tobs, yobs, σobs, _ = readdataset(source="Mrk6")

# Let's look at how data are organised. All of the three arrays have the same structure. They are all arrays of arrays.
display(type(tobs)), display(type(yobs)), display(type(σobs))

# Each array contains 2 inner arrays, one for each observed band (The number of bands is referred to as L in the paper).
length.(tobs)
length.(yobs)
length.(σobs)

# We define an array of candidate delay vectors. Without loss of generalisation, the delay that corresponds to the first light curve is fixed to 0.
delays = [[0;d] for d in 0:0.2:120]

# Check number of candidate delay vectors
length(delays)

# We want to run cross-validation for all candidate delay vectors.
# try true delays
q = performcv(tarray=tobs, yarray=yobs, stdarray=σobs, iterations=1000, numberofrestarts=7, delays = [0;2;6], kernel = GPCC.matern32);

# try perturbed delay
q2 = performcv(tarray=tobs, yarray=yobs, stdarray=σobs, iterations=1000, numberofrestarts=7, delays = [0;2.3;5.8], kernel = GPCC.matern32);

# estimate posterior probability
getprobabilities([q,q2])

# output is:
#2-element Vector{Float64}:
# 0.6356097291597814
# 0.3643902708402206

```
