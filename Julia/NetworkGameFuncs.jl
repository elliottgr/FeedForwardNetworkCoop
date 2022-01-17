using LinearAlgebra, Random, Distributions, ArgParse, StatsBase, DataFrames

####################################
# Network Game Functions
####################################

include("NetworkGameStructs.jl")


## no activation function

function calcOj(activation_scale::Float64, j::Int64, prev_out, Wm::SMatrix, Wb::SVector)

## JVC Activation function
# function calcOj(activation_scale::Float64, j::Int64, prev_out, Wm::SMatrix, Wb::SVector)
#     ##############################
#     ## Iterates a single layer of the Feed Forward network
#     ##############################

#     ## dot product of Wm and prev_out, + node weights. Equivalent to x = dot(Wm[1:j,j], prev_out[1:j]) + Wb[j]
    ## doing it this way allows scalar indexing of the static arrays, which is significantly faster and avoids unnecessary array invocation
    x = 0
    for i in 1:j-1
        x += (Wm[i, j] * prev_out[i]) 
    end
    x += Wb[j]

    return (1/(1+exp(-x * activation_scale))) 
end

function iterateNetwork(activation_scale::Float64, input::Float64, Wm::SMatrix, Wb::SVector, prev_out::MVector)
    ##############################
    ## Calculates the total output of the network,
    ## iterating over calcOj() for each layer
    ##############################

    # prev_out = zeros(Float64, length(Wb))
    # prev_out = @MVector zeros(Float64, length(Wb))
    prev_out[1] = input
    for j in 2:length(Wb)
        prev_out[j] = calcOj(activation_scale, j, prev_out, Wm, Wb)
    end
    
    return prev_out
end

function networkGameRound!(pop, mutI, resI)
    input_mut = pop.networks[resI].CurrentOffer
    input_res = pop.networks[mutI].CurrentOffer

    pop.networks[mutI].CurrentOffer = iterateNetwork(pop.parameters.activation_scale, input_mut, pop.networks[mutI].Wm, pop.networks[mutI].Wb, pop.temp_arrays.prev_out)[pop.parameters.nnet]
    pop.networks[resI].CurrentOffer = iterateNetwork(pop.parameters.activation_scale, input_res, pop.networks[resI].Wm, pop.networks[resI].Wb, pop.temp_arrays.prev_out)[pop.parameters.nnet]
end

function repeatedNetworkGame(pop, mutI, resI)
    ##############################
    ## Plays multiple rounds of the network game, returns differing 
    ## data types depending on whether a discount needs to be calculated
    ##############################
    mutHist = zeros(Float64, pop.parameters.rounds)
    resHist = zeros(Float64, pop.parameters.rounds)

    # reset current offer to initial offer
    pop.networks[mutI].CurrentOffer = pop.networks[mutI].InitialOffer
    pop.networks[resI].CurrentOffer = pop.networks[resI].InitialOffer

    for i in 1:pop.parameters.rounds
        networkGameRound!(pop, mutI, resI)
        mutHist[i] = copy(pop.networks[mutI].CurrentOffer)
        resHist[i] = copy(pop.networks[resI].CurrentOffer)
    end
    if pop.parameters.δ >= 0.0 
        return [mutHist, resHist]
    end
end


function calc_discount(δ::Float64, rounds::Int64)
    return exp.(-δ.*(rounds.-1 .-range(1,rounds, step = 1)))
end



function calc_payoff(parameters::simulation_parameters, rmOut, mrOut, discount)
    output = 0.0
    # wmr = 1 + (dot((parameters.b * rmOut - parameters.c * mrOut + parameters.d * rmOut.*mrOut), discount) * parameters.fitness_benefit_scale)
    for i in 1:parameters.rounds
        output += (parameters.b * rmOut[i] - parameters.c * mrOut[i] + (parameters.d * rmOut[i] * mrOut[i])) * discount[i]
    end
    return 1 + (output * parameters.fitness_benefit_scale)
end

function fitnessOutcome!(pop, mutI, resI)
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

    if pop.parameters.δ >= 0.0
        rmOut, mrOut = repeatedNetworkGame(pop, mutI, resI)
        pop.temp_arrays.gamePayoffTempArray[1][1] = calc_payoff(pop.parameters, mrOut, rmOut, discount)
        pop.temp_arrays.gamePayoffTempArray[1][2] = calc_payoff(pop.parameters, rmOut, mrOut, discount)
        pop.temp_arrays.gamePayoffTempArray[2][1] = dot(rmOut, discount)
        pop.temp_arrays.gamePayoffTempArray[2][2] = dot(mrOut, discount)
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
    Wm_out = Matrix{Float64}(undef, (pop.parameters.nnet,pop.parameters.nnet))
    Wb_out = Vector{Float64}(undef, pop.parameters.nnet)

    init_out = Vector{Float64}(undef, length(pop.networks))

    for net_i in 1:length(pop.networks)
        for i in 1:pop.parameters.nnet
            @inbounds Wb_out[i] += pop.networks[net_i].Wb[i]
            for j in 1:pop.parameters.nnet
                @inbounds Wm_out[i, j] += pop.networks[net_i].Wm[i,j]
            end
        end
    end
    wm_mean = Wm_out./pop.parameters.N
    wb_mean = Wb_out./pop.parameters.N

    return output_network(-1, wm_mean, wb_mean, mean(init_out), mean(init_out))
end


function output!(t::Int64, pop::population, outputs::DataFrame)
    ## Updates output dataframe
    output_row = Int64(t/pop.parameters.output_save_tick)
    
    outputs.generation[output_row] = t
    outputs.n_genotypes[output_row] = pop.n_genotypes
    outputs.mean_payoff[output_row] = mean(pop.payoffs)
    outputs.mean_cooperation[output_row] = mean(pop.cooperation_vals)
    outputs.mean_initial_offer[output_row] = mean(return_initial_offer_array(pop))
    outputs.var_payoff[output_row] = var(pop.payoffs)
    outputs.var_cooperation[output_row] = var(pop.cooperation_vals)
    outputs.var_initial_offer[output_row] = var(return_initial_offer_array(pop))

    mean_network = return_mean_network(pop)

    if pop.parameters.nnet >= 1
       outputs.n1[output_row] = mean_network.Wb[1]
    end 

    if pop.parameters.nnet >= 2
        outputs.n2[output_row] = mean_network.Wb[2]
        outputs.e1_2[output_row] = mean_network.Wm[1,2]
    end

    if pop.parameters.nnet >= 3

        outputs.n3[output_row] = mean_network.Wb[3]
        outputs.e1_3[output_row] = mean_network.Wm[1,3]
        outputs.e2_3[output_row] = mean_network.Wm[2,3]
    end

    if pop.parameters.nnet >= 4

        outputs.n4[output_row] = mean_network.Wb[4]
        outputs.e1_4[output_row] = mean_network.Wm[1,4]
        outputs.e2_4[output_row] = mean_network.Wm[2,4]
        outputs.e3_4[output_row] = mean_network.Wm[3,4]
    end

    if pop.parameters.nnet >= 5
        outputs.n5[output_row] = mean_network.Wb[5]
        outputs.e1_5[output_row] = mean_network.Wm[1,5]
        outputs.e2_5[output_row] = mean_network.Wm[2,5]
        outputs.e3_5[output_row] = mean_network.Wm[3,5]
        outputs.e4_5[output_row] = mean_network.Wm[4,5]
    end
end


        

function population_construction(parameters::simulation_parameters)
    ## constructs a population array when supplied with parameters and a list of networks
    ## should default to a full array of a randomly chosen resident genotype unless
    ## instructed otherwise in params
    initialnetworks = Vector{network}(undef, length(parameters.init_freqs))
    population_array = Vector{network}(undef, parameters.N)
    scale_freq(p, N) = convert(Int64, round((p*N), digits=0))
    for n::Int64 in 1:length(parameters.init_freqs)
        if parameters.init_net_weights == 0.0
            Wm = SMatrix{parameters.nnet, parameters.nnet, Float64}(Matrix(UpperTriangular(randn((parameters.nnet,parameters.nnet)))))
            Wb =  SVector{parameters.nnet, Float64}(randn(parameters.nnet))
        else
            Wm = SMatrix{parameters.nnet, parameters.nnet, Float64}(Matrix(UpperTriangular(fill(parameters.init_net_weights, (parameters.nnet,parameters.nnet)))))
            Wb =  SVector{parameters.nnet, Float64}(fill(parameters.init_net_weights, parameters.nnet))
        end

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
    payoff_temp_array = [[0.0, 0.0], [0.0,0.0]]
    prev_out = @MVector zeros(Float64, parameters.nnet) 
    NetworkGameRound = @MVector zeros(Float64, 2)
    temp_arrays = sim_temp_array(payoff_temp_array, prev_out, NetworkGameRound)
    return population(parameters, shuffle!(population_array), return_genotype_id_array(population_array), Dict{Int64, Dict{Int64, Float64}}(), Dict{Int64, Dict{Int64, Float64}}(), shuffle(1:parameters.N), length(parameters.init_freqs), zeros(Float64, parameters.N), zeros(Float64, parameters.N), 0, temp_arrays)
end

##################
# Pairwise fitness
##################
function update_fit_dict!(pop::population)
    for (n1::Int64, n2::Int64) in zip(1:pop.parameters.N, pop.shuffled_indices)
        if [pop.genotypes[n1], pop.genotypes[n2]] ∉ keys(pop.fit_dict)
          fitnessOutcome!(pop, n2, n1)
          pop.fit_dict[[pop.genotypes[n1], pop.genotypes[n2]]] = pop.temp_arrays.gamePayoffTempArray[1][1]
          pop.coop_dict[[pop.genotypes[n1], pop.genotypes[n2]]] = pop.temp_arrays.gamePayoffTempArray[2][1]
          if [pop.genotypes[n2], pop.genotypes[n1]] ∉ keys(pop.fit_dict)
            pop.fit_dict[[pop.genotypes[n2], pop.genotypes[n1]]] = pop.temp_arrays.gamePayoffTempArray[1][2]
            pop.coop_dict[[pop.genotypes[n2], pop.genotypes[n1]]] = pop.temp_arrays.gamePayoffTempArray[2][2]
          end
        end
    pop.payoffs[n1] = pop.fit_dict[[pop.genotypes[n1], pop.genotypes[n2]]]
    pop.payoffs[n2] = pop.fit_dict[[pop.genotypes[n2], pop.genotypes[n1]]]
    pop.cooperation_vals[n1] = pop.coop_dict[[pop.genotypes[n1], pop.genotypes[n2]]]
    pop.cooperation_vals[n2] = pop.coop_dict[[pop.genotypes[n2], pop.genotypes[n1]]]
    end
end
# function update_fit_dict!(pop::population)

#     for (n1::Int64, n2::Int64) in zip(1:pop.parameters.N, pop.shuffled_indices)
#         if pop.genotypes[n1] ∉ keys(pop.fit_dict)
#             pop.fit_dict[pop.genotypes[n1]] = Dict{Int64, Vector{Float64}}()
#             pop.coop_dict[pop.genotypes[n1]] = Dict{Int64, Vector{Float64}}()
#         end
#         if pop.genotypes[n2] ∉ keys(pop.fit_dict[pop.genotypes[n1]])
#                 # pop.temp_arrays.gamePayoffTempArray = fitnessOutcome!(pop.parameters, pop.networks[n2], pop.networks[n1], pop.temp_arrays)
#                 fitnessOutcome!(pop, n2, n1)
#                 pop.fit_dict[pop.genotypes[n1]][pop.genotypes[n2]] = pop.temp_arrays.gamePayoffTempArray[1][1]
#                 pop.coop_dict[pop.genotypes[n1]][pop.genotypes[n2]] = pop.temp_arrays.gamePayoffTempArray[2][1]
#         end
#         pop.payoffs[n1] = pop.fit_dict[pop.genotypes[n1]][pop.genotypes[n2]]
#         pop.cooperation_vals[n1] = pop.coop_dict[pop.genotypes[n1]][pop.genotypes[n2]]
#     end
# end

##################
# Pairwise fitness
##################


function pairwise_fitness_calc!(pop::population)
    ## shuffles the population array, returns the fitness of the resident at each point calculated by
    ## running the fitness outcome function along both the original and shuffled array
    
    repro_array = zeros(Float64, pop.parameters.N)
    for (n1,n2) in zip(1:pop.parameters.N, pop.shuffled_indices)
        
        if pop.genotypes[n1] == 1
            repro_array[n1] = pop.fit_dict[[pop.genotypes[n1], pop.genotypes[n2]]]*pop.parameters.resident_fitness_scale
        else
            repro_array[n1] = pop.fit_dict[[pop.genotypes[n1], pop.genotypes[n2]]]
        end
    end
    pop.mean_w = mean(repro_array)
    # return repro_array./sum(repro_array)
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
            outWm = pop.networks[i].Wm + Matrix(mutWm)
            outWb = pop.networks[i].Wb + mutWb
            pop.networks[i] = network(pop.n_genotypes,
                                        (outWm),
                                        (outWb),
                                        (pop.networks[i].InitialOffer + mutInit),
                                        (pop.networks[i].InitialOffer + mutInit),
                                        )
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
            default = 50000
        "--nreps"
            help = "number of replicates to run"
            arg_type = Int64
            default = 100
        "--N"   
            help = "population size"
            arg_type = Int64
            default = 500
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
            help = "scales the fitness payout of game rounds by this amount (payoff * scale). setting to 1.0 results in a crash, values between 0.01 and 0.1 work"
            arg_type = Float64
            default = 0.1
        "--b"
            help = "payoff benefit"
            arg_type = Float64
            default = 1.0
        "--c"
            help = "payoff cost"
            arg_type = Float64 
            default = 0.5
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
            default = 1.0
        "--initial_offer"
            help = "the default value of initial offers for the initial residents."
            arg_type = Float64
            default = 0.5
        "--init_freqs"
            help = "vector of initial genotype frequencies, must sum to 1"
            arg_type = Vector{Float64}
            default = [0.5, 0.5]
        "--init_net_weights"
            help = "Initial weight of network nodes, from 0.0 to 1.0. Set to 0.0 to randomly sample"
            arg_type = Float64
            default = 0.00
        ########
        ## Network Parameters
        ########
        "--nnet_min"
            help = "smallest n x n network size of replicates. def = 1"
            arg_type = Int64
            default = 1
        "--nnet_max"
            help = "largest n x n network size of replicates. def = 15"
            arg_type = Int64
            default = 5
        "--nnet_step"
            help = "step size of network iterations. def = 1"
            arg_type = Int64
            default = 1
        "--nnet"
            help = "network size (deprecated)"
            arg_type = Int64
            default = 5
        "--mutsize"
            help = "Size of mutant effects on network in Normal Dist. StdDevs"
            arg_type = Float64
            default = 0.01
        "--mutinitsize"
            help = "Size of mutant effects on initial offers in Normal Dist. StdDevs"
            arg_type = Float64
            default = 0.01
        "--mutlink"
            help = "Probability that a random edge or node be altered in a mutation event"
            arg_type = Float64
            default = 0.5
        "--net_save_tick"
            help = "Computes and saves the mean network values ever [x] timesteps. x = 0 does not save"
            arg_type = Int64
            default = 1000
        "--activation_scale"
            help = "Adjust layer activation function. Formula is (1/(1+exp(-x * activation_scale))). Higher values will filter more activation noise, lower values will allow intermediate activation to propogate through layers."
            arg_type = Float64
            default = 1.0
        ########
        ## File/Simulation Parameters
        ########
        "--output_save_tick"
            help = "Number of timesteps to wait between saving of all simulation results EXCEPT mean network state"
            arg_type = Int64
            default = 1000
        "--replicate_id"
            help = "Internal variable set to track each replicate in the final dataset"
            arg_type = Int64
            default = 0
        "--seed"
            help = "seed number for RNG"
            arg_type = Int64
            default = 1453
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
                                        parsed_args["initial_offer"], parsed_args["init_freqs"], parsed_args["init_net_weights"],
                                        parsed_args["nnet_min"], parsed_args["nnet_max"], parsed_args["nnet_step"],
                                        parsed_args["nnet"], parsed_args["mutsize"], parsed_args["mutinitsize"], parsed_args["mutlink"],
                                        parsed_args["net_save_tick"], parsed_args["replicate_id"], parsed_args["activation_scale"],
                                         parsed_args["output_save_tick"], parsed_args["seed"], parsed_args["filename"], parsed_args["init_freq_resolution"])

    ## 1/13/22
    ## For some reason, the above simulation_parameters() decleration isn't importing parsed_args properly
    ## resetting the arguments seems to do the trick
    parameters.activation_scale = parsed_args["activation_scale"]
    parameters.output_save_tick = parsed_args["output_save_tick"]

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




output_length = Int64(pop.parameters.tmax/pop.parameters.output_save_tick)

    # ## Fixation Stats ##
    # ## time points where an allele becomes 100% of the population
    # fixations::Vector{Int64}
    # ## number of genotypes at each time point
    # n_genotypes::Vector{Int64}
    # ## simulation results ##
    # ## mean values of fitness and initial offer over the sim
    # payoff_mean_history::Vector{Float64}
    # coop_mean_history::Vector{Float64}
    # w_mean_history::Vector{Float64}
    # init_mean_history::Vector{Float64}
    # mean_net_history::Vector{output_network}
    # ## Output copy of parameters
    # parameters::simulation_parameters

 

outputs = DataFrame(b = fill(pop.parameters.b, output_length),
                    c = fill(pop.parameters.c, output_length),
                    nnet = fill(pop.parameters.nnet, output_length),
                    replicate_id = fill(pop.parameters.replicate_id, output_length),
                    generation = zeros(Int64, output_length),
                    n_genotypes = zeros(Int64, output_length),
                    mean_payoff = zeros(Float64, output_length),
                    mean_cooperation = zeros(Float64, output_length), 
                    mean_initial_offer = zeros(Float64, output_length), 
                    var_payoff = zeros(Float64, output_length),
                    var_cooperation = zeros(Float64, output_length),  
                    var_initial_offer = zeros(Float64, output_length), 
                    n1 = fill(NaN, output_length),
                    n2 = fill(NaN, output_length),
                    n3 = fill(NaN, output_length),
                    n4 = fill(NaN, output_length),
                    n5 = fill(NaN, output_length),
                    e1_2 = fill(NaN, output_length),
                    e1_3 = fill(NaN, output_length),
                    e1_4 = fill(NaN, output_length),
                    e1_5 = fill(NaN, output_length),
                    e2_3 = fill(NaN, output_length),
                    e2_4 = fill(NaN, output_length),
                    e2_5 = fill(NaN, output_length),
                    e3_4 = fill(NaN, output_length),
                    e3_5 = fill(NaN, output_length),
                    e4_5 = fill(NaN, output_length),)

    ## pre allocating this array so it doesn't get reallocated each time a game is played
    global discount = calc_discount(pop.parameters.δ, pop.parameters.rounds)
    global discount = SVector{pop.parameters.rounds}(discount/sum(discount))
    ############
    # Sim Loop #
    ############
            # 1/11/22 Testing a thing
        # for μ in 1:500
        #     mutate!(pop)
        #     update_population!(pop)
        # end
    # update_population!(pop)
    # initial_payoffs = mean(copy(pop.payoffs))
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
        if mod(t, pop.parameters.output_save_tick) == 0
            output!(t, copy(pop), outputs)
        end

    end
# print("Final Payoff: "*string(mean(pop.payoffs)))
# if mean(pop.payoffs) == initial_payoffs
#     print("Replicate did not evolve new payoffs")
# end
## organize replicate data into appropriate data structure to be returned to main function and saved
return outputs
end

