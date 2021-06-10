

## organized by part of the model modified, values set via ArgParse

mutable struct simulation_parameters
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
    mutsize::Float64
    mutinitsize::Float64
    mutlink::Float64
    #file params
    filename::String
end

## smallest type necessary to play a complete round of the game 
mutable struct network
    genotype_id::Int64
    Wm
    Wb::Vector{Float64}
    InitialOffer::Float64
    CurrentOffer::Float64
end

## prevents the creation of population arrays that won't work with the shuffle reproduction method
## (those where N mod 2 != 0)
mutable struct population
    parameters::simulation_parameters
    networks::Vector{network}
    genotypes::Vector{Int64}
    fit_dict::Dict{Int64, Dict{Int64, Vector{Float64}}}
    shuffled_indices::Vector{Int64}
end

mutable struct simulation_output
    ## Fixation Stats ##
    ## time points where an allele becomes 100% of the population
    fixations::Vector{Int64}
    ## number of genotypes at each time point
    n_genotypes::Vector{Int64}

    ## simulation results ##
    ## mean values of fitness and initial offer over the sim
    w_mean_history::Vector{Float64}
    init_mean_history::Vector{Float64}

    ## Output copy of parameters
    parameters::simulation_parameters

end
