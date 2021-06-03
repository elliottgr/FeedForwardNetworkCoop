 
using LinearAlgebra

####################################
# Structs
####################################

mutable struct simulation_parameters
    tmax::Int64
    nreps::Int64
    N::Int64
    μ::Float64
    rounds::Int64
    fitness_benefit_scale::Float64
    b::Float64
    c::Float64
    d::Float64
    δ::Float64

end

mutable struct network
    Wm
    Wb::Vector{Float64}
    InitialOffer::Float64
    CurrentOffer::Float64
end

####################################
# Functions from Python version
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

function networkGameRound(mutNet, resNet)
    mutOut = last(iterateNetwork(resNet.CurrentOffer, mutNet.Wm, mutNet.Wb))
    resOut = last(iterateNetwork(mutNet.CurrentOffer, resNet.Wm, resNet.Wb))
    return [mutOut, resOut]
end

##############################
## Plays multiple rounds of the network game, returns differing 
## data types depending on whether a discount needs to be calculated
##############################

function repeatedNetworkGame(parameters, mutNet, resNet)
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

function fitnessOutcome(parameters::simulation_parameters,mutNet,resNet)

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

###################
# Simulation Loop #
###################
function simulation(parameters::simulation_parameters, initNetwork::network)

end
###################
#   Parameters    #
###################

using ArgParse
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
    "--nnet"
        help = "network size"
        arg_type = Int64
        default = 2
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
    #Parameters for fitness payoff of game
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
end
parsed_args = parse_args(ARGS, arg_parse_settings)


nnet = 2
initWm = transpose(reshape([0.71824181,2.02987316,-0.42858626,0.6634413],2,2))
initWb = [-0.66332791,1.00430577]
init = 0.1
InitialNetwork = network(Wm, Wb, init, init)
parameters = simulation_parameters(parsed_args["tmax"], parsed_args["nreps"], parsed_args["N"], parsed_args["mu"],
                                parsed_args["rounds"], parsed_args["fitness_benefit_scale"], parsed_args["b"], 
                                parsed_args["c"], parsed_args["d"], parsed_args["delta"])

##################################
#Random Test Matrix Values
##################################



##################################
#Function Testing Scratch Space
##################################
# repeatedNetworkGameHistory(5, a, a)
b = fitnessOutcome(params, a,a)

