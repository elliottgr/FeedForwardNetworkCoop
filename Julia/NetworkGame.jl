 
using LinearAlgebra
using ArgParse
using Random
####################################
# Structs
####################################


## organized by part of the model modified, values set via ArgParse

struct simulation_parameters
    #popgen params
    tmax::Int64
    nreps::Int64
    N::Int64
    μ::Float64
    #game params
    rounds::Int64
    fitness_benefit_scale::Float64
    b::Float64
    c::Float64
    d::Float64
    δ::Float64
    init_freqs::Vector{Float64}
    #network params
    nnet::Int64
end

## smallest type necessary to play a complete round of the game 
mutable struct network
    Wm
    Wb::Vector{Float64}
    InitialOffer::Float64
    CurrentOffer::Float64
end

## stores information about each networks genotype, not necessary in hindsight
struct individual
    network::network
    genotypeID::Int64
end

## prevents the creation of population arrays that won't work with the shuffle reproduction method
## (those where N mod 2 != 0)
mutable struct population
    array::Vector{individual}
    function population(array)
        if mod(length(array), 2) != 0
            error("Please provide an even value of N!")
        end
        new(array)
    end
end


####################################
# Network Game Functions
####################################


##############################
## Iterates a single layer of the Feed Forward network
##############################
function calcOj(j::Int64, prev_out, Wm, Wb::Vector{Float64})
    x = dot(Wm[1:j,j][1:j], prev_out[1:j]) + Wb[j]
    return 1-(exp(x*-x))
end

##############################
## Calculates the total output of the network,
## iterating over calcOj() for each layer
##############################

function iterateNetwork(input::Float64, Wm, Wb::Vector{Float64})
    prev_out = zeros(Float64, length(Wb))
    prev_out[1] = input
    for j in 2:length(Wb)
        prev_out[j] = calcOj(j, prev_out, Wm, Wb)
    end
    return prev_out
end

##############################
## Iterates above functions over a pair of networks,
## constitutes a single game round
##############################

function networkGameRound(mutNet::network, resNet::network)
    mutOut = last(iterateNetwork(resNet.CurrentOffer, mutNet.Wm, mutNet.Wb))
    resOut = last(iterateNetwork(mutNet.CurrentOffer, resNet.Wm, resNet.Wb))
    return [mutOut, resOut]
end

##############################
## Plays multiple rounds of the network game, returns differing 
## data types depending on whether a discount needs to be calculated
##############################

function repeatedNetworkGame(parameters::simulation_parameters, mutNet::network, resNet::network)
    mutNet.CurrentOffer = mutNet.InitialOffer
    resNet.CurrentOffer = mutNet.InitialOffer
    mutHist = zeros(parameters.rounds)
    resHist = zeros(parameters.rounds)
    for i in 1:parameters.rounds
        mutNet.CurrentOffer, resNet.CurrentOffer = networkGameRound(mutNet, resNet)
        mutHist[i] = mutNet.CurrentOffer
        resHist[i] = resNet.CurrentOffer
    end
    if parameters.δ >= 0
        return [mutHist, resHist]
    elseif parameters.δ < 0
        return [mutNet.CurrentOffer, resNet.CurrentOffer]
    end
end


####################################
## calulate resident-mutant fitness matrix from contributions throughout the game history. If
## discount is availible, applied at rate delta going from present to the past in the 
## repeated cooperative investment game
##
## fitness = 1 + b * (partner) - c * (self) + d * (self) * (partner)
####################################

function fitnessOutcome(parameters::simulation_parameters,mutNet::network,resNet::network)

    ############################################################
    ## with discountretrieves the full history and passes it to the
    ##  discount formula before returning final fitness value
    ############################################################
    if parameters.δ >= 0.0
        rmOut, mrOut = repeatedNetworkGame(parameters,mutNet,resNet)
        discount = exp.(-parameters.δ.*(parameters.rounds.-1 .-range(1,parameters.rounds, step = 1)))
        td = sum(discount)
        wmr = max(0, (1 + dot((parameters.b * rmOut - parameters.c * mrOut + parameters.d * rmOut.*mrOut), discount) * parameters.fitness_benefit_scale))
        wrm = max(0, (1 + dot((parameters.b * mrOut - parameters.c * rmOut + parameters.d * rmOut.*mrOut), discount)*parameters.fitness_benefit_scale))
        return [[wmr, wrm], [dot(rmOut, discount)/td, dot(mrOut, discount)/td]]

    ############################################################
    ## without discount, retrieves only the final value after all 
    ## rounds played and returns it as w based on game parameters
    ############################################################
    elseif parameters.δ < 0.0
        rmOut, mrOut = repeatedNetworkGameHistory(parameters, mutNet, resNet)
        wmr = max(0, (1 + ((parameters.b * rmOut - parameters.c * mrOut + parameters.d * rmOut.*mrOut)*parameters.fitness_benefit_scale)))
        wrm = max(0, (1 + ((parameters.b * mrOut - parameters.c * rmOut + parameters.d * rmOut.*mrOut)*parameters.fitness_benefit_scale)))
        return [[wmr, wrm], [rmOut, mrOut]]

    end
end

###############################
# Population Simulation Funcs #
###############################


##################
# Pop Construction
##################

## Will be updated to support populations of arbitrary size/genotype frequency

# function population_construction(N::Int64, resNet::network, mutNets::Vector{individual} = individual[], initFreqs::Vector{Any} = [])
#     if length(mutNets) == 0 && length(initFreqs) == 0
#         return population(repeat([individual(resNet, 0)], N))
#     end
# end

function population_construction(N::Int64, networks::Vector{individual}, initFreqs::Vector{Float64} = [1.0, 0.0])
    population_array = Vector{individual}(undef, 0)
    for (net, p ) in zip(networks, initFreqs)
        append!(population_array, repeat([net], Int64(p*N)))
    end
    return population_array
end



#######################
# Simulation Function #
#######################

## following similar format to NetworkGame.py

function simulation(parameters::simulation_parameters, initNetworks::Vector{individual})


    population(population_construction(parameters.N, initNetworks, parameters.init_freqs))
############
# Sim init #
############

## generation of initial population from parameters

## EG 6/4/21
## WIP Note: May need to pass a vector of initial networks + corresponding weights if want this to be 
## generalizable. Trying to do this without touching anything inside the networks struct so that I can plug JVC's
## julia network code in later.


## arrays that track population statistics
## EG 6/4/21
## WIP Note: Need to decide on output format, then create an easier to modify workflow for this.
## some kind of output struct that tracks whole sim statistics, and has vectors of timepoint statistics
## as well?


## dicts for genotype lookup
## EG 6/4/21
## WIP Note: Not sure if the dict method used in the python version will be the besto option here.
## An array based method doesn't seem too difficult, but it has much more memory overhead.


    ############
    # Sim Loop #
    ############

    for t in 1:parameters.tmax
        x = 1
        # reproduction function

        # mutation function

        # update t+1 population array

        # per-timestep counters, outputs going to disk



    end

## organize replicate data into appropriate data structure to be returned to main function and saved

end

###################
#   Parameters    #
###################
function main()

    ##########
    ## Arg Parsing
    ##########

    arg_parse_settings = ArgParseSettings()
    @add_arg_table arg_parse_settings begin

        ########
        ## Population Simulation Parameters
        ########
        "--tmax"
            help = "Maximum number of timesteps"
            arg_type = Int64
            default = 1000
        "--nreps"
            help = "number of replicates to run"
            arg_type = Int64
            default = 100

        "--N"   
            help = "population size"
            arg_type = Int64
            default = 100
        "--mu"
            help = "mutation probability per birth"
            arg_type = Float64
            default = 0.0

        ########
        ## Game Parameters
        ########
        "--rounds"
            help = "number of rounds the game is played between individuals"
            arg_type = Int64
            default = 5

        "--fitness_benefit_scale"
            help = "scales the fitness payout of game rounds by this amount (payoff * scale)"
            arg_type = Float64
            default = 1.0

        "--b"
            help = "payoff benefit"
            arg_type = Float64
            default = 0.0
        "--c"
            help = "payoff cost"
            arg_type = Float64 
            default = 0.0
        "--d"
            help = "payoff synergy"
            arg_type = Float64
            default = 0.0
        "--r"
            help = "relatedness coefficient"
            arg_type = Float64
            default = 0.0
        "--delta"
            help = "payoff discount, negative values use last round"
            arg_type = Float64
            default = 0.0
        "--init_freqs"
            help = "vector of initial genotype frequencies, must sum to 1"
            arg_type = Vector{Float64}
            default = [0.25, 0.25, .25, .25]
        ########
        ## Network Parameters
        ########
        "--nnet"
            help = "network size"
            arg_type = Int64
            default = 5
    end
    
    ##passing command line arguments to simulation
    parsed_args = parse_args(ARGS, arg_parse_settings)
    parameters = simulation_parameters(parsed_args["tmax"], parsed_args["nreps"], parsed_args["N"], parsed_args["mu"],
                                        parsed_args["rounds"], parsed_args["fitness_benefit_scale"], parsed_args["b"], 
                                        parsed_args["c"], parsed_args["d"], parsed_args["delta"], parsed_args["init_freqs"], parsed_args["nnet"])

    ##############
    ## Test Values for comparing to python (or other) implementation
    ##############

    # nnet = 2
    # initWm = transpose(reshape([0.71824181,2.02987316,-0.42858626,0.6634413],2,2))
    # initWb = [-0.66332791,1.00430577]
    # init = 0.1
    # InitialNetwork = network(initWm, initWb, init, init)
    
    ##################################
    #Generation of Random Initial Network
    ##################################
    ## also generates a random mutant for testing population population_construction

    initWm = randn((parameters.nnet,parameters.nnet))
    initWb = randn(parameters.nnet)
    muWm = randn((parameters.nnet,parameters.nnet))
    muWb = randn(parameters.nnet)
    initialOffer = (1.0 + randn())/2
    muInitialOffer = (1.0 + randn())/2
    # InitialNetwork = network(initWm, initWb, initialOffer, initialOffer)
    initialnetworks = Vector{individual}(undef, length(parameters.init_freqs))
    
    for n in 1:length(parameters.init_freqs)
        initialnetworks[n] = individual(network(muWm, muWb, muInitialOffer, muInitialOffer), n)
    end
    ###################
    # Simulation call #
    ###################

    simulation(parameters, initialnetworks)

    ###################
    #   Data Output   #
    ###################

    ## sim data will be stored in previous section and saved to disk here. 


    ###################
    #     Visuals     #
    ###################
    
    ## should be it's own script eventually. Anything necessary for debugging should be done here
    ## also, test out passing the files to some graphics function at some point if it becomes an issue


end

main()

