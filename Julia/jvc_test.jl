using Distributed
using Plots
using CairoMakie
using AlgebraOfGraphics
using DataFramesMeta
using Chain
using ForwardDiff

nproc = 50
addprocs(nproc)
@everywhere begin
    include("NetworkGameFuncs.jl")
    BLAS.set_num_threads(1)
end

##

# set seeds on workers
@sync [@async remotecall_fetch(Random.seed!, w, w) for w in workers()] # set seeds on all workers

pars = simulation_parameters(
    50000,         # tmax
    nproc,          # nreps
    500,            # N
    0.01,           # mutation rate per individual
    20,             # number of rounds
    0.2,            # payoff scale
    1.0,            # b
    0.5,            # c
    0.0,            # d
    0.0,            # discount rate
    0.0,            # param_min
    2.0,            # param_max
    0.1,            # param_step
    0.5,            # initial offer
    [0.5, 0.5],     # initial frequencies
    0.0,            # initial network weights
    2,              # network size min
    2,              # network size max
    1,              # network size step
    3,              # network size
    0.05,           # mut std for network weight 
    0.05,           # mut std for initial offer
    0.5,            # probability of mutating node or edge
    linear,         # threshold function
    5.0,            # scale for network output into threshold function    
    100,            # time step for output
    0,              # replicate id
    314,            # seed
    "test.jld2"     # output filename
)
pars_reps = [copy(pars) for i in 1:pars.nreps];
[pars_reps[i].replicate_id = i for i in 1:pars.nreps];

output = vcat(pmap((x)->simulation(population_construction(x)), pars_reps)...);
mean_output = @chain output groupby(:generation) combine([:b, :c, :mean_payoff, :mean_cooperation, :n1, :n2, :n3, :e1_2, :e2_3, :mean_initial_offer] .=> mean, renamecols=false)

##
mean_output_slice = @chain mean_output @subset(:generation .< 25e3)
## Scalar that (if the population reaches an ESS) should represent the slope at each timepoint
df_dx = ForwardDiff.derivative(pars.activation_function, mean_output_slice[mean_output_slice.generation .== maximum(mean_output_slice.generation), :mean_initial_offer][1]) 

draw(
    data(@chain mean_output_slice stack([:mean_payoff, :mean_cooperation])) * 
    mapping(:generation, :value, color = :variable) *
    visual(Lines); 
    axis = (title = string(String(Symbol(pars.activation_function)), ", nnet: ", pars.nnet), width = 400, height = 200)
)

draw(
    data(@chain mean_output_slice stack([:n1, :n2, :n3, :e1_2, :e2_3, :mean_initial_offer])) * 
    mapping(:generation, :value, color = :variable) *
    visual(Lines); 
    axis = (title = string(String(Symbol(pars.activation_function)), ", nnet: ", pars.nnet), width = 400, height = 200)
)

draw(
    data(@chain mean_output_slice @transform(:bmc = @. :b * df_dx * :e1_2 - :c )) * 
    mapping(:generation, :bmc) *
    visual(Lines); 
    axis = (title = string(String(Symbol(pars.activation_function)), ", nnet: ", pars.nnet), width = 400, height = 200)
) 