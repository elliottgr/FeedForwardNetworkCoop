 
using LinearAlgebra

####################################
# Functions from Python version
####################################

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

##################
# Parameters
##################
nnet = 2
Wm = randn(Float64, (nnet,nnet))
Wb = randn(Float64, nnet)


#################
#Random Test Matrix Values
#################

Wm = reshape([0.71824181,2.02987316,-0.42858626,0.6634413],2,2)
Wb = [-0.66332791,1.00430577]
iterateNetwork(0.0, Wm, Wb)
