using Distributed
using Plots
using CairoMakie
using AlgebraOfGraphics
using DataFramesMeta
using Chain

include("NetworkGameFuncs.jl")

pars = simulation_parameters(
    100000,          # tmax
    1,              # nreps
    500,            # N
    0.01,           # mutation rate per individual
    20,             # number of rounds
    0.1,            # payoff scale
    1.0,            # b
    0.5,            # c
    0.0,            # d
    0.0,            # discount rate
    0.0,            # param_min
    2.0,            # param_max
    0.1,            # param_step
    0.1,            # initial offer
    [0.5, 0.5],     # initial frequencies
    0.0,            # initial network weights
    2,              # network size min
    2,              # network size max
    1,              # network size step
    2,              # network size
    0.05,           # mut std for network weight 
    0.05,           # mut std for initial offer
    0.5,            # probability of mutating node or edge
    linear,         # threshold function
    1.0,            # scale for network output into threshold function    
    100,            # time step for output
    0,              # replicate id
    314,            # seed
    "test.jld2"     # output filename
)

popul = population_construction(pars)

output = simulation(popul)

draw(
    data(output) * 
    mapping(:generation, :mean_cooperation) *
    visual(Lines); 
    axis = (width = 400, height = 200))

draw(
    data(@chain output stack([:n1, :n2, :e1_2, :mean_initial_offer])) * 
    mapping(:generation, :value, color = :variable) *
    visual(Lines); 
    axis = (width = 400, height = 200)
    )

