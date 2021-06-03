 
using LinearAlgebra

####################################
# Functions from Python version
####################################

mutable struct simulation_parameters
    #number of rounds the game is played between individuals
    rounds::Int64

    #scales the fitness payout of game rounds by this amount (payoff * scale)
    fitness_benefit_scale::Float64

    #Parameters for fitness payoff of game
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

function calcOj(j::Int64, prev_out, Wm, Wb::Vector{Float64})
    x = dot(Wm[1:j,j][1:j], prev_out[1:j]) + Wb[j]
    return 1-(exp(x*-x))
end

function iterateNetwork(input::Float64, Wm, Wb::Vector{Float64})
    prev_out = zeros(Float64, length(Wb))
    prev_out[1] = input
    for j in 2:length(Wb)
        prev_out[j] = calcOj(j, prev_out, Wm, Wb)
    end
    return prev_out
end

function networkGameRound(mutNet, resNet)
    mutOut = last(iterateNetwork(resNet.CurrentOffer, mutNet.Wm, mutNet.Wb))
    resOut = last(iterateNetwork(mutNet.CurrentOffer, resNet.Wm, resNet.Wb))
    return [mutOut, resOut]
end

function repeatedNetworkGame(rounds, mutNet, resNet)
    for i in 1:rounds
        mutInit, resInit = networkGameRound(mutNet, resNet)
    end
    return [mutInit, resInit]
end

function repeatedNetworkGameHistory(rounds::Int, mutNet, resNet)
    mutNet.CurrentOffer = mutNet.InitialOffer
    resNet.CurrentOffer = mutNet.InitialOffer
    mutHist = zeros(rounds)
    resHist = zeros(rounds)
    for i in 1:rounds
        mutNet.CurrentOffer, resNet.CurrentOffer = networkGameRound(mutNet, resNet)
        mutHist[i] = mutNet.CurrentOffer
        resHist[i] = resNet.CurrentOffer
    end
    return [mutHist, resHist]
end


####################################
## calulate resident-mutant fitness matrix from contributions throughout the game history,
## discounted at rate delta going from present to the past in the repeated cooperative
## investment game
##
## fitness = 1 + b * (partner) - c * (self) + d * (self) * (partner)
####################################

function fitnessOutcome(parameters::simulation_parameters,mutNet,resNet)
    if parameters.δ >= 0 
        rmOut, mrOut = repeatedNetworkGameHistory(parameters.rounds,mutNet,resNet)
        # print(rmOut.*mrOut)
        discount = exp.(-parameters.δ.*(parameters.rounds.-1 .-range(1,parameters.rounds, step = 1)))
        # print(discount)
        td = sum(discount)
        # print(parameters.b * rmOut )
        wmr = max(0, (1 + dot((parameters.b * rmOut - parameters.c * mrOut + parameters.d * rmOut.*mrOut), discount) * parameters.fitness_benefit_scale))
        wrm = max(0, (1 + dot((parameters.b * mrOut - parameters.c * rmOut + parameters.d * rmOut.*mrOut), discount)*parameters.fitness_benefit_scale))
        return [[wmr, wrm], [dot(rmOut, discount)/td, dot(mrOut, discount)/td]]
    elseif parameters.δ < 0
        rmOut, mrOut = repeatedNetworkGame(parameters.rounds, mutNet, resNet)
        wmr = max(0, (1 + ((parameters.b * rmOut - parameters.c * mrOut + parameters.d * rmOut.*mrOut)*parameters.fitness_benefit_scale)))
        wrm = max(0, (1 + ((parameters.b * mrOut - parameters.c * rmOut + parameters.d * rmOut.*mrOut)*parameters.fitness_benefit_scale)))
        print(parameters.b * rmOut )
        print("test")
        return [[wmr, wrm], [rmOut, mrOut]]
    end
end
##################
# Parameters
##################
nnet = 2
# Wm = randn(Float64, (nnet,nnet))
# Wb = randn(Float64, nnet)
Wm = transpose(reshape([0.71824181,2.02987316,-0.42858626,0.6634413],2,2))
Wb = [-0.66332791,1.00430577]
init = 0.1
#################
#Random Test Matrix Values
#################

a = network(Wm, Wb, init, init)
params = simulation_parameters(5, 0.0,0.0,0.0,0.0,-0.1)
repeatedNetworkGameHistory(5, a, a)
b = fitnessOutcomeEntireHist(params, a,a)

