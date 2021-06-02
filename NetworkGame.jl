 
using LinearAlgebra

####################################
# Functions from Python version
####################################

function calcOj(j::Int64, prev_out, Wm::Matrix{Float64}, Wb::Vector{Float64})
    x = dot(Wm[1:j,j][1:j], prev_out[1:j]) + Wb[j]
    return 1-(exp(x*-x))
end

function iterateNetwork(input::Float64, Wm::Matrix{Float64}, Wb::Vector{Float64})
    prev_out = zeros(Float64, length(Wb))
    prev_out[1] = input
    for j in 2:length(Wb)
        prev_out[j] = calcOj(j, prev_out, Wm, Wb)
    end
    return prev_out
end

function networkGameRound(mutWm::Matrix{Float64}, mutWb::Vector{Float64}, mutPrev::Float64, resWm::Matrix{Float64}, resWb::Vector{Float64}, resPrev)
    mutOut = last(iterateNetwork(resPrev, mutWm, mutWb))
    resOut = last(iterateNetwork(mutPrev, resWm, resWb))
    return [mutOut, resOut]
end

function repeatedNetworkGame(rounds, mutWm::Matrix{Float64}, mutWb::Vector{Float64}, mutInit::Float64, resWm::Matrix{Float64}, resWb::Vector{Float64}, resInit::Float64)
    for i in 1:rounds
        mutInit, resInit = networkGameRound(mutWm, mutWb, mutInit, resWm, resWb, resInit)
    end
    return [mutInit, resInit]
end

function repeatedNetworkGameHistory(rounds::Int, mutWm, mutWb, mutInit, resWm, resWb, resInit)
    mutHist = zeros(rounds)
    resHist = zeros(rounds)
    for i in 1:rounds
        mutInit, resInit = networkGameRound(mutWm, mutWb, mutInit, resWm, resWb, resInit)
        mutHist[i] = mutInit
        resHist[i] = resInit
    end
    return [mutHist, resHist]
end

##################
# Parameters
##################
nnet = 2
# Wm = randn(Float64, (nnet,nnet))
# Wb = randn(Float64, nnet)
Wm = transpose(reshape([0.71824181,2.02987316,-0.42858626,0.6634413],2,2))
Wb = [-0.66332791,1.00430577]

#################
#Random Test Matrix Values
#################


init = 0.1
repeatedNetworkGameHistory(5, Wm, Wb, init, Wm, Wb, init)

