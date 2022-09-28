module GPCC

    using PyPlot, BlockArrays, Random

    using Optim, Distributions, LinearAlgebra, StatsFuns

    using Printf, MiscUtil, Suppressor

    using MLBase, StatsFuns, ApproximateVI


    include("delayedCovariance.jl")

    include("simulatedata.jl")

    include("util.jl")


    include("UNUSED/gpccvi.jl")

    include("gpccfixdelay_marginaliseb.jl")

    include("performcv.jl")

    include("getprobabilities.jl")

    include("uniformpriordelay.jl")


    export simulatedata, gpcc, performcv, getprobabilities, uniformpriordelay, gpccvi


end
