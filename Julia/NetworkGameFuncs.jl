using LinearAlgebra, Random, Distributions

####################################
# Network Game Functions
####################################

include("NetworkGameStructs.jl")

function calcOj(j::Int64, prev_out, Wm, Wb::Vector{Float64})
    ##############################
    ## Iterates a single layer of the Feed Forward network
    ##############################
    x = dot(Wm[1:j,j][1:j], prev_out[1:j]) + Wb[j]
    return 1-(exp(x*-x))
end

function iterateNetwork(input::Float64, Wm::Matrix{Float64}, Wb::Vector{Float64})
    ##############################
    ## Calculates the total output of the network,
    ## iterating over calcOj() for each layer
    ##############################

    prev_out = zeros(Float64, length(Wb))
    prev_out[1] = input
    for j in 2:length(Wb)
        prev_out[j] = calcOj(j, prev_out, Wm, Wb)
    end
    return prev_out
end

function networkGameRound(mutNet::network, resNet::network)
    ##############################
    ## Iterates above functions over a pair of networks,
    ## constitutes a single game round
    ##############################
    mutOut = last(iterateNetwork(resNet.CurrentOffer, mutNet.Wm, mutNet.Wb))
    resOut = last(iterateNetwork(mutNet.CurrentOffer, resNet.Wm, resNet.Wb))
    return [mutOut, resOut]
end

function repeatedNetworkGame(parameters::simulation_parameters, mutNet::network, resNet::network)
    ##############################
    ## Plays multiple rounds of the network game, returns differing 
    ## data types depending on whether a discount needs to be calculated
    ##############################

    mutNet.CurrentOffer = mutNet.InitialOffer
    resNet.CurrentOffer = resNet.InitialOffer

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

function fitnessOutcome(parameters::simulation_parameters,mutNet::network,resNet::network)

    ####################################
    ## calulate resident-mutant fitness matrix from contributions throughout the game history. If
    ## discount is availible, applied at rate delta going from present to the past in the 
    ## repeated cooperative investment game
    ##
    ## fitness = 1 + b * (partner) - c * (self) + d * (self) * (partner)
    ####################################
    ############################################################
    ## with discount, retrieves the full history and passes it to the
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

function update_population!(pop::population)
    ## runs functions necessary at every timestep of the simulation
    ## updates pop struct with new partner indices and genotype ID arrays
    pop.genotypes = return_genotype_id_array(pop.networks)
    shuffle!(pop.shuffled_indices)
    update_fit_dict!(pop)
end

function return_genotype_id_array(population_array::Vector{network})
    ## WIP returns an array of the genotype inside
    genotype_array = Vector{Int64}(undef, 0)
    for individual in population_array
        append!(genotype_array, individual.genotype_id)
    end
    return genotype_array
end

function output!(t::Int64, pop::population, outputs::simulation_output)
    ## Updates output arrays
    if length(Set(pop.genotypes)) == 1
        outputs.fixations[t] = maximum(Set(pop.genotypes))
    else
        outputs.fixations[t] = 0
    end
    ## Maximum or length of the set of keys should return the largest genotype index ever present because
    ## each iteration will guarantee it shows up in fit_dict via the shuffle method
    outputs.n_genotypes[t] = length(Set(keys(pop.fit_dict)))
end
function population_construction(parameters::simulation_parameters)
    ## constructs a population array when supplied with parameters and a list of networks
    ## should default to a full array of a randomly chosen resident genotype unless
    ## instructed otherwise in params
    initialnetworks = Vector{network}(undef, length(parameters.init_freqs))
    population_array = Vector{network}(undef, 0)
    for n in 1:length(parameters.init_freqs)
        Wm = randn((parameters.nnet,parameters.nnet))

        Wb = randn(parameters.nnet)
        initOffer = (1.0 + randn())/2
        initialnetworks[n] = network(n, Wm, Wb, initOffer, initOffer)
    end
    for (net, p, gen) in zip(initialnetworks, parameters.init_freqs, 1:length(parameters.init_freqs))
        append!(population_array, repeat([net], Int64(trunc(p*parameters.N))))
    end
    return population(parameters, population_array, return_genotype_id_array(population_array), Dict{Int64, Dict{Int64, Vector{Float64}}}(), shuffle(1:parameters.N))
end

##################
# Pairwise fitness
##################
function update_fit_dict!(pop::population)
    g(x) = pop.genotypes[x]
    for (n1, n2) in zip(1:pop.parameters.N, pop.shuffled_indices)
        if g(n1) ∉ keys(pop.fit_dict)
            pop.fit_dict[g(n1)] = Dict{Int64, Vector{Float64}}()
        end
        if g(n2) ∉ keys(pop.fit_dict[g(n1)])
            fitness_outcome, raw_payoffs = fitnessOutcome(pop.parameters, pop.networks[n2], pop.networks[n1])
            pop.fit_dict[g(n1)][g(n2)] = fitness_outcome
        end
    end
end
##################
# Pairwise fitness
##################


function pairwise_fitness_calc!(pop::population)
    ## shuffles the population array, returns an array of fitness values calculated by
    ## running the fitness outcome function along both the original and shuffled array
    repro_array = Vector{Float64}(undef, pop.parameters.N)
    g(x) = pop.genotypes[x]
    for (n1,n2) in zip(1:pop.parameters.N, pop.shuffled_indices)
        repro_array[n1] = pop.fit_dict[g(n1)][g(n2)][1]./ sum(pop.fit_dict[g(n1)][g(n2)])
    end
    return repro_array
end

##################
# Reproduction function
##################

function reproduce!(pop::population)
    ## working with new arrays rather than copy of old pop to avoid in-place weirdness with shuffle()

    repro_array = pairwise_fitness_calc!(pop)
    new_genotypes = Vector{Int64}(undef, pop.parameters.N)
    new_networks = Vector{network}(undef, pop.parameters.N)
    g(x) = pop.genotypes[x]
    n(x) = pop.networks[x]
    for (res_i, mut_i) in zip(1:length(pop.networks), pop.shuffled_indices)
        if rand() <= repro_array[res_i]
            new_genotypes[res_i] = g(res_i)
            new_networks[res_i] = n(res_i)
        else
            new_genotypes[res_i] = g(mut_i)
            new_networks[res_i] = n(mut_i)
        end
    end
    pop.genotypes = new_genotypes
    pop.networks = new_networks
end

##################
#  Mutation Function 
##################

function mutate!(pop::population)
    for i in 1:length(pop.networks)
        if rand() <= pop.parameters.μ
            pop.genotypes[i] = maximum(return_genotype_id_array(pop.networks)) + 1
            mutWm = UpperTriangular(rand(Binomial(1, pop.parameters.mutlink), (pop.parameters.nnet,pop.parameters.nnet)) 
                                    .* rand(Normal(0, pop.parameters.mutsize), (pop.parameters.nnet,pop.parameters.nnet)))
            mutWb = rand(Binomial(1, pop.parameters.mutlink), pop.parameters.nnet) .* rand(Normal(0, pop.parameters.mutsize),pop.parameters.nnet)
            mutInit = rand(Normal(0, pop.parameters.mutsize))
            pop.networks[i] = network((maximum(return_genotype_id_array(pop.networks))+1),
                                        (pop.networks[i].Wm + mutWm),
                                        (pop.networks[i].Wb + mutWb),
                                        (pop.networks[i].InitialOffer + mutInit),
                                        (pop.networks[i].InitialOffer + mutInit))
        end
    end
end



#######################
# Simulation Function #
#######################

## following similar format to NetworkGame.py

function simulation(pop::population)

############
# Sim init #
############


## EG 6/4/21
## WIP Note: May need to pass a vector of initial networks + corresponding weights if want this to be 
## generalizable. Trying to do this without touching anything inside the networks struct so that I can plug JVC's
## julia network code in later.


## arrays that track population statistics
## EG 6/4/21
## WIP Note: Need to decide on output format, then create an easier to modify workflow for this.
## some kind of output struct that tracks whole sim statistics, and has vectors of timepoint statistics
## as well?

outputs = simulation_output(zeros(Int64, pop.parameters.tmax),
                            zeros(Int64, pop.parameters.tmax),
                            zeros(Float64, pop.parameters.tmax),
                            zeros(Float64, pop.parameters.tmax),
                            pop.parameters)

    ############
    # Sim Loop #
    ############
    for t in 1:pop.parameters.tmax
        # update population struct 
        update_population!(pop)

        # reproduction function / produce and save t+1 population array
        reproduce!(pop)

        # mutation function / iterates over population and mutates at chance probability μ
        if pop.parameters.μ > 0
            mutate!(pop)
        end
        # per-timestep counters, outputs going to disk
        output!(t, pop, outputs)

        ## should detect an error in genotype tracking. Will trip if there is <2 genotypes initially
        if pop.parameters.init_freqs[1] != 0.0
            if length(Set(keys(pop.fit_dict))) != maximum(Set(keys(pop.fit_dict)))
                print("Length: ", length(Set(keys(pop.fit_dict))))
                print("Max: ", maximum(Set(keys(pop.fit_dict))))
                print("Error in genotype tracking, dictionary of fitness values has missing genotypes")
                break
            end
        end
    end
## organize replicate data into appropriate data structure to be returned to main function and saved
return outputs
end

