## This should plot fitness outcomes of all possible 2 node networks (excluding biases) when played against themselves
using Plots
include("NetworkGameFuncs.jl")

function main(activation_function = linear, b = 1.0, c = 0.5, initial_offers = 0.5, samples = 25, e_min= -1.0, e_max = 1.0)
    ## Constructing a dummy parameter set and population
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
    # params.nnet = 2
    # params.b = b
    # params.c = c
    # params.activation_function = activation_function

    pop = population_construction(params)
    print(pop.parameters.activation_function)


    ## pulling out the first network to make dummy copies
    test_net_1 = copy(pop.networks[1])
    test_net_2 = copy(test_net_1)


    ## Initializing output matrix
    outputs = Matrix(undef, samples+1, samples+1)
    m = 0

    ## Running main iteration loop
    for net_1_e1_2 in e_min:((e_max-e_min)/samples):e_max
        n = 0
        m += 1
        test_net_1.Wm = SMatrix{params.nnet, params.nnet, Float64}([0.0 net_1_e1_2 ; 0.0 0.0])
        pop.networks[1] = test_net_1
        for net_2_e1_2 in e_min:((e_max-e_min)/samples):e_max
            n += 1
            test_net_2.Wm = SMatrix{params.nnet, params.nnet, Float64}([0.0 net_2_e1_2 ; 0.0 1.0])
            pop.networks[2] = test_net_2
            interactionOutcome!(pop, 1, 2)
            outputs[m, n] = last(pop.temp_arrays.gamePayoffTempArray[1,1])
        end
    end

    ## Plot stuff!
    title_str = "$activation_function, b = $b, c = $c, InitOffer = $initial_offers"
    ticks = ([1.0:samples/10:samples+1;], [string(i) for i in e_min:((e_max-e_min)/samples*10):e_max])
    heatmap(outputs,
            title = title_str,
            xlabel = "Network 1 Edge Weight", ylabel = "Network 2 Edge Weight",
            xticks =  ticks, yticks = ticks)
end

main(linear, 1.0, 0.5, 0.1, 100)
