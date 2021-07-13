using LinearAlgebra, Random, Distributions, ArgParse, StatsBase

####################################
# Network Game Functions
####################################

include("NetworkGameStructs.jl")

function calcOj(j::Int64, prev_out::Vector{Float64}, Wm::Matrix{Float64}, Wb::Vector{Float64} )
    ##############################
    ## Iterates a single layer of the Feed Forward network
    ##############################
    x = dot(Wm[1:j,j], prev_out[1:j]) + Wb[j]
    return (1/(1+exp(-x)))
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

    mutHist = zeros(Float64, parameters.rounds)
    resHist = zeros(Float64, parameters.rounds)
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

function calc_discount(δ::Float64, rounds::Int64)
    return exp.(-δ.*(rounds.-1 .-range(1,rounds, step = 1)))
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

    ## 6/16 Note: Previous versions of this script returned an array of arrays, it now returns a single array
    if parameters.δ >= 0.0
        rmOut, mrOut = repeatedNetworkGame(parameters,mutNet,resNet)
        discount = calc_discount(parameters.δ, parameters.rounds)
        # x::Vector{Float64} = (parameters.b * rmOut - parameters.c * mrOut + parameters.d * rmOut.*mrOut)
        # y::Vector{Float64} = (parameters.b * mrOut - parameters.c * rmOut + parameters.d * rmOut.*mrOut)

        # wmr = 1 + (dot((parameters.b * rmOut - parameters.c * mrOut + parameters.d * rmOut.*mrOut), discount) * parameters.fitness_benefit_scale)
        wrm = 1 + (dot((parameters.b * mrOut - parameters.c * rmOut + parameters.d * rmOut.*mrOut), discount) * parameters.fitness_benefit_scale)

        ## this will return the frequency of competitions in which
        ## the the resident will outcompete the mutant in the reproduction game
        ## P(mutant) + P(resident) = 1
        return wrm
        ## Legacy code, changed 6/18/21
        #return [wmr, wrm]
    ############################################################
    ## without discount, retrieves only the final value after all 
    ## rounds played and returns it as w based on game parameters
    ############################################################
    elseif parameters.δ < 0.0
        rmOut, mrOut = repeatedNetworkGameHistory(parameters, mutNet, resNet)
        wmr = max(0.0, (1 + ((parameters.b * rmOut - parameters.c * mrOut + parameters.d * rmOut.*mrOut)*parameters.fitness_benefit_scale)))
        wrm = max(0.0, (1 + ((parameters.b * mrOut - parameters.c * rmOut + parameters.d * rmOut.*mrOut)*parameters.fitness_benefit_scale)))
        return wrm

    end
end

###############################
# Population Simulation Funcs #
###############################

function update_population!(pop::population)
    ## runs functions necessary at every timestep of the simulation
    ## updates pop struct with new partner indices and genotype ID arrays
    pop.genotypes = return_genotype_id_array(pop.networks)
    pop.shuffled_indices = shuffle(pop.shuffled_indices)
    update_fit_dict!(pop)
end

function return_initial_offer_array(pop::population)
    init_offers = zeros(Float64, pop.parameters.N)
    for i in 1:pop.parameters.N
        init_offers[i] = pop.networks[i].InitialOffer
    end
    return init_offers
end


function return_genotype_id_array(population_array::Vector{network})
    ## Returns an array of the genotype inside
    genotype_array = zeros(Int64, length(population_array))
    for i in 1:length(population_array)
        genotype_array[i] = population_array[i].genotype_id
    end
    return genotype_array
end

## Will iterate over each network and save the mean value of each vertex/edge
function return_mean_network(pop::population)
    Wm_out = zeros((pop.parameters.nnet, pop.parameters.nnet))
    Wb_out = zeros(pop.parameters.nnet)
    init_out = Vector{Float64}(undef, length(pop.networks))
    ## summing the value of each network, dividing by pop size at end to find mean
    for i in 1:length(pop.networks)
        for ij in eachindex(pop.networks[i].Wm)
            Wm_out[ij] += pop.networks[i].Wm[ij]
        end
        for j in 1:pop.parameters.nnet
            Wb_out[j] += pop.networks[i].Wb[j]
        end

    end
    wm_mean = Wm_out./pop.parameters.N
    wb_mean = Wb_out./pop.parameters.N
    # wm_mean = mean!(zeros((pop.parameters.nnet, pop.parameters.nnet)),Wm_out)
    # wb_mean = mean!(zeros(pop.parameters.nnet), Wb_out)

    return network(-1, wm_mean, wb_mean, mean(init_out), mean(init_out))
end


function output!(t::Int64, pop::population, outputs::simulation_output)
    ## Updates output arrays
    if count(i->(i==pop.genotypes[1]), pop.genotypes) == pop.parameters.N
        outputs.fixations[t] = pop.genotypes[1]
    else
        outputs.fixations[t] = 0
    end
    ## Maximum or length of the set of keys should return the largest genotype index ever present because
    ## each iteration will guarantee it shows up in fit_dict via the shuffle method

    ## only calculates mean net on tick intervals to save calculations
    if pop.parameters.net_save_tick != 0
        if mod(pop.parameters.net_save_tick, t) == 0
            outputs.mean_net_history[t] = return_mean_network(pop)
        else
            outputs.mean_net_history[t] = outputs.mean_net_history[t-1]
        end
    end

    outputs.n_genotypes[t] = pop.n_genotypes
    outputs.init_mean_history[t] = mean(return_initial_offer_array(pop))
    outputs.w_mean_history[t] = pop.mean_w
end


## returns number if 0 < number < 1, else returns 0, 1
function range_check(number::Float64)
    if number < 0
        return 0
    elseif number > 1
        return 1
    else
        return number
    end
end

function range_check(vector::Vector{Float64})
    for i in 1:length(vector)
        vector[i] = range_check(vector[i])
    end
    return vector
end

function range_check(matrix::Matrix{Float64})
    for i in eachindex(matrix)
        matrix[i] = range_check(matrix[i])
    end
    return matrix
end
        

function population_construction(parameters::simulation_parameters)
    ## constructs a population array when supplied with parameters and a list of networks
    ## should default to a full array of a randomly chosen resident genotype unless
    ## instructed otherwise in params
    initialnetworks = Vector{network}(undef, length(parameters.init_freqs))
    population_array = Vector{network}(undef, parameters.N)
    scale_freq(p, N) = convert(Int64, round((p*N), digits=0))
    for n::Int64 in 1:length(parameters.init_freqs)
        Wm = UpperTriangular(randn((parameters.nnet,parameters.nnet)))
        Wb = randn(parameters.nnet)
        # initOffer = range_check((1.0 + randn())/2)
        initOffer = copy(parameters.initial_offer)
        initialnetworks[n] = network(n, Wm, Wb, initOffer, initOffer)
    end
    pop_iterator = 0
    for init_freq_i in 1:length(parameters.init_freqs)
        for n_i in 1:scale_freq(parameters.init_freqs[init_freq_i], parameters.N)
            pop_iterator += 1
            population_array[pop_iterator] = initialnetworks[init_freq_i]
        end
    end
    if length(population_array) != parameters.N
        return error("population array failed to generate $N networks")
    end
    return population(parameters, shuffle!(population_array), return_genotype_id_array(population_array), Dict{Int64, Dict{Int64, Float64}}(), shuffle(1:parameters.N), length(parameters.init_freqs), 0)
end

##################
# Pairwise fitness
##################
function update_fit_dict!(pop::population)
    for (n1::Int64, n2::Int64) in zip(1:pop.parameters.N, pop.shuffled_indices)
        if pop.genotypes[n1] ∉ keys(pop.fit_dict)
            pop.fit_dict[pop.genotypes[n1]] = Dict{Int64, Vector{Float64}}()
        end
        if pop.genotypes[n2] ∉ keys(pop.fit_dict[pop.genotypes[n1]])
            if n1 != 1
                pop.fit_dict[pop.genotypes[n1]][pop.genotypes[n2]] = fitnessOutcome(pop.parameters, pop.networks[n2], pop.networks[n1])
            else 
                pop.fit_dict[pop.genotypes[n1]][pop.genotypes[n2]] = fitnessOutcome(pop.parameters, pop.networks[n2], pop.networks[n1])
            end
        end
    end
end

##################
# Pairwise fitness
##################


function pairwise_fitness_calc!(pop::population)
    ## shuffles the population array, returns the fitness of the resident at each point calculated by
    ## running the fitness outcome function along both the original and shuffled array
    
    repro_array = zeros(Float64, pop.parameters.N)
    for (n1,n2) in zip(1:pop.parameters.N, pop.shuffled_indices)
        if pop.genotypes[n1] == 1
            repro_array[n1] = pop.fit_dict[pop.genotypes[n1]][pop.genotypes[n2]]*pop.parameters.resident_fitness_scale
        else
            repro_array[n1] = pop.fit_dict[pop.genotypes[n1]][pop.genotypes[n2]]
        end
    end
    pop.mean_w = mean(repro_array)
    return repro_array./sum(repro_array)
end

##################
# Reproduction function
##################

function reproduce!(pop::population)
    repro_array = pairwise_fitness_calc!(pop)
    genotype_i_array = sample(1:pop.parameters.N, ProbabilityWeights(repro_array), pop.parameters.N, replace=true)
    old_networks = copy(pop.networks)
    for (res_i, offspring_i) in zip(1:pop.parameters.N, genotype_i_array)
        pop.networks[res_i] = old_networks[offspring_i]
    end
end

##################
#  Mutation Function 
##################

function mutate!(pop::population)
    for i in 1:length(pop.networks)
        if rand() <= pop.parameters.μ
            pop.n_genotypes += 1
            pop.genotypes[i] = pop.n_genotypes
            mutWm = UpperTriangular(rand(Binomial(1, pop.parameters.mutlink), (pop.parameters.nnet,pop.parameters.nnet)) 
                                    .* rand(Normal(0, pop.parameters.mutsize), (pop.parameters.nnet,pop.parameters.nnet)))
            mutWb = rand(Binomial(1, pop.parameters.mutlink), pop.parameters.nnet) .* rand(Normal(0, pop.parameters.mutsize),pop.parameters.nnet)
            mutInit = rand(Normal(0, pop.parameters.mutinitsize))

            pop.networks[i] = network(pop.n_genotypes,
                                        (pop.networks[i].Wm + mutWm),
                                        (pop.networks[i].Wb + mutWb),
                                        range_check(pop.networks[i].InitialOffer + mutInit),
                                        range_check(pop.networks[i].InitialOffer + mutInit))
        end
    end
end

# \:raised_hands: 
#######################
# Simulation Function #
#######################


function initial_arg_parsing()
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
            default = 0.01
        "--resident_fitness_scale"
            help = "scales the initial resident fitness for debugging pop gen funcs"
            arg_type = Float64
            default = 1.0
        ########
        ## Game Parameters
        ########
        "--rounds"
            help = "number of rounds the game is played between individuals"
            arg_type = Int64
            default = 16

        "--fitness_benefit_scale"
            help = "scales the fitness payout of game rounds by this amount (payoff * scale)"
            arg_type = Float64
            default = 0.01

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
            help = "relatedness coefficient (does nothing as of 7/1/21)"
            arg_type = Float64
            default = 0.0
        "--delta"
            help = "payoff discount, negative values use last round"
            arg_type = Float64
            default = 0.0
        "--game_param_min"
            help = "initial value of b/c for replicates"
            arg_type = Float64
            default = 0.0
        "--game_param_max"
            help = "final value of b/c for replicates"
            arg_type = Float64
            default = 2.0
        "--game_param_step"
            help = "step size of iterations for b/c"
            arg_type = Float64
            default = 0.5
        "--initial_offer"
            help = "the default value of initial offers for the initial residents."
            arg_type = Float64
            default = 0.1
        "--init_freqs"
            help = "vector of initial genotype frequencies, must sum to 1"
            arg_type = Vector{Float64}
            default = [.99, 0.01]
        ########
        ## Network Parameters
        ########
        "--nnet"
            help = "network size"
            arg_type = Int64
            default = 5
        "--mutsize"
            help = "Size of mutant effects on network in Normal Dist. StdDevs"
            arg_type = Float64
            default = 0.1
        "--mutinitsize"
            help = "Size of mutant effects on initial offers in Normal Dist. StdDevs"
            arg_type = Float64
            default = 0.1
        "--mutlink"
            help = "Probability that a random edge or node be altered in a mutation event"
            arg_type = Float64
            default = 0.5
        "--net_save_tick"
            help = "Computes and saves the mean network values ever [x] timesteps. x = 0 does not save"
            arg_type = Int64
            default = 1000
        ########
        ## File/Simulation Parameters
        ########
        "--seed"
            help = "seed number for RNG"
            arg_type = Int64
            default = 666
        "--filename"
            help = "Filename to save outputs to (please include .jld2 extension)"
            arg_type = String
            default = "NetworkGameTests.jld2"
        "--init_freq_resolution"
            help = "Step-size between initial frequencies if iterating over them"
            arg_type = Float64
            default = 0.05
    end

    ##passing command line arguments to simulation
    parsed_args = parse_args(ARGS, arg_parse_settings)
    parameters = simulation_parameters(parsed_args["tmax"], parsed_args["nreps"], parsed_args["N"], parsed_args["mu"], parsed_args["resident_fitness_scale"],
                                        parsed_args["rounds"], parsed_args["fitness_benefit_scale"], parsed_args["b"], 
                                        parsed_args["c"], parsed_args["d"], parsed_args["delta"],
                                        parsed_args["game_param_min"], parsed_args["game_param_max"], parsed_args["game_param_step"],
                                         parsed_args["initial_offer"], parsed_args["init_freqs"], 
                                        parsed_args["nnet"], parsed_args["mutsize"], parsed_args["mutinitsize"], parsed_args["mutlink"],
                                        parsed_args["net_save_tick"], parsed_args["seed"], parsed_args["filename"], parsed_args["init_freq_resolution"])

    ## Necessary sanity checks for params
    if mod(parameters.N, 2) != 0
        print("Please supply an even value of N!")
    end
    return parameters
end


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
                            Vector{network}(undef, pop.parameters.tmax),
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


        ## ends the loop if only one genotype exists AND mutation is not enabled
        if pop.parameters.μ ==  0.0
            if outputs.fixations[t] != 0
                return outputs
            end
        end
        ## should detect an error in genotype tracking. Will trip if there is <2 genotypes initially
        if pop.parameters.init_freqs[1] != 0.0
            if length(Set(keys(pop.fit_dict))) != maximum(Set(keys(pop.fit_dict)))
                print("Length: ", length(Set(keys(pop.fit_dict))), "\n")
                print("Max: ", maximum(Set(keys(pop.fit_dict))), "\n")
                print("Error in genotype tracking, dictionary of fitness values has missing genotypes", "\n")
                break
            end
        end
    end
## organize replicate data into appropriate data structure to be returned to main function and saved
return outputs
end

