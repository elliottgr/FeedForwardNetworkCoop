using Strided

## organized by part of the model modified, values set via ArgParse

mutable struct simulation_parameters
    #popgen params
    tmax::Int64
    nreps::Int64
    N::Int64
    μ::Float64 
    resident_fitness_scale::Float64
    #game params
    rounds::Int64
    fitness_benefit_scale::Float64
    b::Float64
    c::Float64
    d::Float64
    δ::Float64
    game_param_min::Float64
    game_param_max::Float64
    game_param_step::Float64
    initial_offer::Float64
    init_freqs::Vector{Float64}
    #network params
    nnet_min::Int64
    nnet_max::Int64
    nnet_step::Int64
    nnet::Int64
    mutsize::Float64
    mutinitsize::Float64
    mutlink::Float64
    net_save_tick::Int64
    #file/simulation params
    seed::Int64
    filename::String
    init_freq_resolution::Float64
end

## smallest type necessary to play a complete round of the game 
mutable struct network
    genotype_id::Int64
    Wm::StridedView{Float64, 2, Vector{Float64}}
    Wb::StridedView{Float64, 1, Vector{Float64}}
    InitialOffer::Float64
    CurrentOffer::Float64
end

## copy of old network structure for saving to disk. Only difference is this has basic Julia Arrays since 
## these won't need further matrix operations

mutable struct output_network
    genotype_id::Int64
    Wm::Matrix{Float64}
    Wb::Vector{Float64}
    InitialOffer::Float64
    CurrentOffer::Float64
end

function Base.copy(net::network)
    return network(net.genotype_id, net.Wm, net.Wb, net.InitialOffer, net.CurrentOffer)
end

## need to be able to create copies of the parameters and networks structs
function Base.copy(parameters::simulation_parameters)
    return simulation_parameters(parameters.tmax, parameters.nreps,parameters.N,parameters.μ, 
                                parameters.resident_fitness_scale, parameters.rounds,
                                parameters.fitness_benefit_scale,parameters.b,parameters.c,parameters.d,
                                parameters.game_param_min, parameters.game_param_max, parameters.game_param_step,
                                parameters.δ,parameters.initial_offer, parameters.init_freqs,
                                parameters.nnet_min, parameters.nnet_max, parameters.nnet_step, parameters.nnet,
                                parameters.mutsize,parameters.mutinitsize,parameters.mutlink,parameters.net_save_tick,
                                parameters.seed,parameters.filename, parameters.init_freq_resolution)
end

function Base.copy(networks::Vector{network})
    out = Vector{network}(undef, length(networks))
    for n in 1:length(networks)
        out[n] = copy(networks[n])
    end
    return out
end

## prevents the creation of population arrays that won't work with the shuffle reproduction method
## (those where N mod 2 != 0)
mutable struct population
    parameters::simulation_parameters
    networks::Vector{network}
    genotypes::Vector{Int64}
    fit_dict::Dict{Int64, Dict{Int64, Float64}}
    shuffled_indices::Vector{Int64}
    n_genotypes::Int64
    payoffs::Vector{Float64}
    mean_w::Float64
    gamePayoffTempArray::Vector{Vector{Float64}}
end

mutable struct simulation_output
    ## Fixation Stats ##
    ## time points where an allele becomes 100% of the population
    fixations::Vector{Int64}
    ## number of genotypes at each time point
    n_genotypes::Vector{Int64}
    ## simulation results ##
    ## mean values of fitness and initial offer over the sim
    payoff_mean_history::Vector{Float64}
    w_mean_history::Vector{Float64}
    init_mean_history::Vector{Float64}
    mean_net_history::Vector{output_network}
    ## Output copy of parameters
    parameters::simulation_parameters

end
