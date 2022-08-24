## This script plots fitness outcomes of possible 
## edge weights in 2 node landscapes, where a
## focal and partner player engage in the game
## The diagonal of the output heatmaps corresponds
## to playing against a clone of oneself, and so
## the evolutionary stable strategy should like on
## the x = y diagonal of the outputs.
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
    pop = population_construction(params)

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
    ticks = ([1.0:samples/10:samples+1;], [string(i) for i in e_min:(samples/10*(e_max-e_min)/(samples)):e_max]) #Do not ask me why the step size looks like that. I got it to work for various sample sizes and didn't feel like figuring it out further
    
    return heatmap(outputs,
            title = "$activation_function",
            # xlabel = "Net 1 Edge Weight", ylabel = "Net 2 Edge Weight",
            xticks =  ticks, yticks = ticks,
            # legend = :none)
    )
end

## Params for the overall plot
b::Float64 = 1.0
c::Float64 = 0.5
init_offer::Float64 = 0.0
samples::Int64 = 500
e_min::Float64 = -.5
e_max::Float64 = .5
b_min::Float64 = -.5
b_max::Float64 = .5


####################
## Generates a main plot combining each of the others
####################

plot(main(jvc_exp, b, c, init_offer, samples, e_min, e_max), main(linear,  b, c, init_offer, samples, e_min, e_max), main(gaussian,  b, c, init_offer, samples, e_min, e_max), main(softplus,  b, c, init_offer, samples, e_min, e_max),
    plot_title = "b = $b, c = $c, InitOffer = $init_offer", layout = 4,
    ylabel = "Focal e1_2", xlabel = "Partner e1_2",
    size = (1000,1000))


####################
## Generates the other plots individually if you want to look at them in VSCode
####################

main(jvc_exp,  b, c, init_offer, samples, e_min, e_max)

## Gives three equally fit local optima,
## and one global optimum, each seperated by 
## an apparent fitness valley
main(linear, b, c, init_offer, samples, e_min, e_max)

## Gives a really cool assymetric plot where
## the focal individual is incentivized to not change
## but the opponent is
main(gaussian,  b, c, init_offer, samples, e_min, e_max)

## Gives three distinct regions of payoff behavior,
## with an optimum when not playing against oneself
main(softplus,  b, c, init_offer, samples, e_min, e_max)
