## This script plots fitness outcomes of (presumably)
## evolutionary stable strategies. The X and Y coordinates
## correspond to different Edge and Node values of a 2 node
## network. This obfuscates the actual underlying landscape,
## since we can't see whether different maxima are reachable
## from one another, but we should be able to see at least potential
## end points of the evolutionary process simulated in
## NetworkGameFuncs main simulation loop.

using Plots
include("NetworkGameFuncs.jl")


## Largely follows the same structure as EdgeLandscapes.jl

function main(activation_function = linear, b = 1.0, c = 0.5, initial_offers = 0.5, samples = 25, e_min= -1.0, e_max = 1.0, b_min = -10.0, b_max = 10.0)
    ## Constructing a dummy parameter set, initial network, and population
    params = simulation_parameters(
        1,         # tmax
        1,          # nreps
        500,            # N
        0.01,           # mutation rate per individual
        20,             # number of rounds
        0.2,            # payoff scale
        b,            # b
        c,            # c
        0.0,            # d
        0.0,            # discount rate
        0.0,            # param_min
        2.0,            # param_max
        0.1,            # param_step
        initial_offers,            # initial offer
        [0.5, 0.5],     # initial frequencies
        0.0,            # initial network weights
        2,              # network size min
        2,              # network size max
        1,              # network size step
        2,              # network size
        0.05,           # mut std for network weight 
        0.05,           # mut std for initial offer
        0.5,            # probability of mutating node or edge
        activation_function,         # threshold function
        10.0,            # scale for network output into threshold function    
        100,            # time step for output
        0,              # replicate id
        314,            # seed
        "test.jld2"     # output filename
    )
    pop = population_construction(params)
    test_net = copy(pop.networks[1])

    ## Initializing output matrix
    outputs = Matrix(undef, samples+1, samples+1)
    m = 0

    ## Outer layer of the loop is the same as EdgeLandscapes.jl
    for net_e1_2 in e_min:((e_max-e_min)/samples):e_max
        n = 0
        m += 1
        test_net.Wm = SMatrix{params.nnet, params.nnet, Float64}([0.0 net_e1_2 ; 0.0 0.0])
        for node_bias in b_min:((b_max-b_min)/samples):b_max
            n+=1
            test_net.Wb = SVector{params.nnet, Float64}([0.0, node_bias])
            pop.networks[1] = test_net
            interactionOutcome!(pop, 1, 1)
            outputs[m, n] = last(pop.temp_arrays.gamePayoffTempArray[1,1])
        end
    end
    y_ticks = ([1.0:samples/10:samples+1;], [string(i) for i in e_min:(samples/10*(e_max-e_min)/(samples)):e_max]) 
    x_ticks = ([1.0:samples/10:samples+1;], [string(i) for i in b_min:(samples/10*(b_max-b_min)/(samples)):b_max]) 
    return heatmap(outputs,
                    title = "$activation_function",
                    xticks = x_ticks, yticks = y_ticks)
end

b::Float64 = 1.0
c::Float64 = 0.5
init_offer::Float64 = 0.5
samples::Int64 = 500
e_min::Float64 = -1.0
e_max::Float64 = 1.0
b_min::Float64 = -1.0
b_max::Float64 = 1.0

plot(main(jvc_exp, b, c, init_offer, samples, e_min, e_max, b_min, b_max), main(linear,  b, c, init_offer, samples, e_min, e_max, b_min, b_max), main(gaussian,  b, c, init_offer, samples, e_min, e_max, b_min, b_max), main(softplus,  b, c, init_offer, samples, e_min, e_max, b_min, b_max),
    plot_title = "b = $b, c = $c, InitOffer = $init_offer", layout = 4,
    xlabel = "Node Bias", ylabel = "Edge Weight",
    size = (1000,1000))

## Tight zoom on the odd fitness valley/peaks off the main diagonal under a gaussian Distributions

plot(main(gaussian,  1.0, .5, .5, samples, -.5, -.2, .15, .25), 
xlabel = "Node Bias", ylabel = "Edge Weight",size = (1000,1000))

