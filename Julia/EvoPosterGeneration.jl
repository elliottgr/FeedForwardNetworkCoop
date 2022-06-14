## This file should be used to generate figure that is a comparison of response rules used by NetworkGameFuncs
## Mostly using the replicate calls from jvc_test.jl for convenience, but modified to iterate activation functions
using Distributed
using Plots
using CairoMakie
using AlgebraOfGraphics
using DataFramesMeta
using Chain
using ForwardDiff

nproc = 120
addprocs(nproc)
@everywhere begin
    include("NetworkGameFuncs.jl")
    BLAS.set_num_threads(1)
end


@sync [@async remotecall_fetch(Random.seed!, w, w) for w in workers()] # set seeds on all workers

## setting input range as a variable so I can adjust it later
input_range = 0:0.01:1
response_rule_df = DataFrame(Input = collect(input_range))

pars = simulation_parameters(
        5000,         # tmax
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

for activ_func in [bounded_linear, jvc_exp, lenagard_exp, ReLU, linear, ELU, heaviside]

    pars.activation_function = activ_func
    pars_reps = [copy(pars) for i in 1:pars.nreps];
    [pars_reps[i].replicate_id = i for i in 1:pars.nreps];

    output = vcat(pmap((x)->simulation(population_construction(x)), pars_reps)...);
    mean_output = @chain output groupby(:generation) combine([:b, :c, :mean_payoff, :mean_cooperation, :n1, :n2, :n3, :e1_2, :e2_3, :mean_initial_offer] .=> mean, renamecols=false)
    mean_output_slice = @chain mean_output @subset(:generation .< 25e3)


    ## Recovering mean network at the end of the simulation run 
    mean_Wm = SMatrix{pars.nnet, pars.nnet, Float64}(Matrix(UpperTriangular(fill(mean_output_slice[mean_output_slice.generation .== maximum(mean_output_slice.generation), :e1_2][1], (pars.nnet, pars.nnet)))))
    mean_Wb = SVector{pars.nnet, Float64}([mean_output_slice[mean_output_slice.generation .== maximum(mean_output_slice.generation), :n1][1], mean_output_slice[mean_output_slice.generation .== maximum(mean_output_slice.generation), :n2][1]])
    # mean_Wb = SVector{pars.nnet, Float64}([mean_output_slice[mean_output_slice.generation .== maximum(mean_output_slice.generation), :n1][1]])
    mean_init = mean_output_slice[mean_output_slice.generation .== maximum(mean_output_slice.generation), :mean_initial_offer][1] 
    mean_output_network = network(0, mean_Wm, mean_Wb, mean_init, mean_init)


    temp = []
    ## Copied from main NetworkGameFuncs.jl
    prev_out = @MVector zeros(Float64, pars.nnet) 

    for i in input_range
        push!(temp, iterateNetwork(pars.activation_function, pars.activation_scale, i, mean_output_network.Wm, mean_output_network.Wb, prev_out)[pars.nnet])
    end 

    response_rule_df[!, String(Symbol(pars.activation_function))] = temp
end

draw(
    data(@chain response_rule_df stack([:bounded_linear, :jvc_exp, :lenagard_exp, :ReLU, :linear, :ELU, :heaviside])) *
    mapping(:Input, :value, color = :variable) *
    visual(Lines);
    axis = (title = string("b = ", string(pars.b), ", c = ", string(pars.c), ", Generations = ", string(pars.tmax)), ylabel = "Output")
)