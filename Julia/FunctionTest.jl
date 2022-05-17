## This is a demo file to make sure that the population genetics functions in NetworkGameFuncs.jl are behaving as 
## expected. This may also prove useful if someone needs to see how the functions are being used
## in the main simulation

## include necessary files

include("NetworkGameFuncs.jl")
include("ActivationFuncs.jl")
using Plots

function main(b = 1.0, c = 0.5, activation_function = "linear", nnet = 5, activation_scale = 1.0, sample_generations = 1000)

    ## Making a dictionary for extracting data to plot
    
    out_dict = Dict()
    out_dict["activation_function"] = activation_function
    ## Import dummy parameters, defining global variables used in main files
    parameters = initial_arg_parsing() ## Can also use this to quickly check new params by altering NetworkGame defaults!

    ## Setting the network size of the testing populations
    parameters.nnet = nnet
    parameters.activation_function = getfield(Main, Symbol(activation_function))
    ## Create a demo population

    pop = population_construction(parameters)

    ## Verifying the population is being constructed with the correct # of individuals
    print("# of individuals requested: ")
    print(parameters.N)
    print("\n")
    print("# of individuals actually constructed: ")
    print(length(pop.payoffs))
    print("\n")

    ################################
    ## Reproduction testing, making sure that results from gameplay actually result in selection
    ################################

    ## Testing population updating (first step of main loop)
    initial_population = copy(pop) ## saved version of the population as initialized via parameters
    update_population!(pop)
    social_interactions!(pop)
    if pop.shuffled_indices == initial_population.shuffled_indices
        print("Population shuffling failed! Something may be broken! \n")
    end


    ## Testing reproduce!()
    ## reproduction shouldn't change the population if there are no mutants
    gen0 = copy(pop)
    reproduce!(pop)

    if pop.genotypes != gen0.genotypes
        print("\n Reproduction without mutation is creating new populations, something went wrong!")
    end

    ## testing reproduce!() on a mutated population
    mutagenic_events = 5 # number of times the population experiences the mutation function, any value >3 should be fine
    for μ in 1:mutagenic_events
        mutate!(pop)
        update_population!(pop)
    end

    if pop.genotypes == gen0.genotypes
        print("\n Reproduction with mutation is NOT mutating into a new population, something went wrong!")
    else
        print("\nGood news! The mutate() function is changing the population!\n")
    end

    ## Resetting the population to check if game results actually impact reproduction
    pop = copy(initial_population)

    ## Setting b = 1.0, leaving c = 0.0 to encourage cooperation
    pop.parameters.b = b
    pop.parameters.c = c

    ## Setting this to 1.0 makes most networks return an activation of b/c
    pop.parameters.activation_scale = activation_scale

    ## Hyper-mutating the population a lot to differentiate genotypes
    mutagenic_events = 500
    for μ in 1:mutagenic_events
        mutate!(pop)
        update_population!(pop)
    end

    ## Testing the fitness results of the first genotype and its shuffled partner
    update_population!(pop)

    g1 = pop.genotypes[pop.shuffled_indices[1]]
    n1 = pop.networks[pop.shuffled_indices[1]]
    g2 = pop.genotypes[pop.shuffled_indices[2]]
    n2 = pop.networks[pop.shuffled_indices[2]]
    g3 = pop.genotypes[pop.shuffled_indices[3]]
    n3 = pop.networks[pop.shuffled_indices[3]]
    print("\n\n ########################################################################\n")
    print("## Randomly Sampled Network Stats ## \n ")
    print("########################################################################\n\n")
    print("Genotype 1: $g1 \n Network Weights: $n1 \n\n")
    print("Genotype 2: $g2 \n Network Weights: $n2 \n\n")
    print("Genotype 3: $g3 \n Network Weights: $n3 \n\n")

    if pop.networks[1] == pop.networks[pop.shuffled_indices[1]]
        print("\n ################ ERROR ################\n You're comparing the same network here :(\n################################\n")
    end


    ## Using G1 and G2 to show the difference of payoffs over multiple rounds

    pop.temp_arrays.prev_out = @MVector zeros(Float64, parameters.nnet) 
    outcome = repeatedNetworkGame(pop, pop.shuffled_indices[1], 1)
    g1_post_round = outcome[1]
    g2_post_round = outcome[2]
    n_rounds = pop.parameters.rounds
    print("########################################################################\n")
    print("You can see here the results of G1 (Genotype # $g1) and G2 (Genotype # $g2) over $n_rounds rounds:\n")
    print("########################################################################\n\n")
    print("G1 Results: $g1_post_round \n\n")
    print("G2 Results: $g2_post_round \n\n")

    print("Here's these arrays as a table: \n\n")
    print(" r |       G1       |       G2       \n")
    for x in 1:length(g1_post_round)
        g1_round = g1_post_round[x]
        g2_round = g2_post_round[x]
        print(" $x | $g1_round | $g2_round \n")
    end

    all_p1 = []
    all_p2 = []

    
    
    ## Making a plot of every pair in the population
    for n in 1:pop.parameters.N
        push!(all_p1, repeatedNetworkGame(pop, pop.shuffled_indices[n], n)[1])
        push!(all_p2, repeatedNetworkGame(pop, pop.shuffled_indices[n], n)[2])
    end

    out_dict["rounds"] = pop.parameters.rounds
    out_dict["p1_rounds"] = all_p1
    out_dict["p2_rounds"] =all_p2
    out_dict["N"] = pop.parameters.N
  
    print("########################################################################\n")
    print("Testing the reproduction() function on our hyper-mutated population\n")
    print("########################################################################\n\n")

    # repro_array = pairwise_fitness_calc!(pop)
    repro_array = pop.payoffs
    parent_ids = pop.genotypes
    offspring_indices = sample(1:pop.parameters.N, ProbabilityWeights(repro_array), pop.parameters.N, replace=true)
    offspring_genotype_ids = []
    offspring_networks = []
    for n in offspring_indices
        push!(offspring_genotype_ids, pop.genotypes[n])
        push!(offspring_networks, pop.networks[n])
    end
    n_parent_genotypes = length(unique(parent_ids))
    n_offspring_genotypes = length(unique(offspring_indices))
    offspring_pop = copy(pop)
    offspring_pop.networks = offspring_networks
    offspring_pop.genotypes = offspring_genotype_ids
    print("$n_offspring_genotypes out of $n_parent_genotypes genotypes were kept after one round of reproduction!\n\n")

    mean_parent_payoffs = mean(pop.payoffs)
    offspring_payoffs = []

    ## estimating using the parent shuffled_indices because it'd be tedious to recode that for this demo
    ## need to calculate all possible fitness values real quick

    for n in 1:pop.parameters.N
        interactionOutcome!(offspring_pop, offspring_pop.shuffled_indices[n], n)
        push!(offspring_payoffs, offspring_pop.temp_arrays.gamePayoffTempArray[1][1])
    end

    mean_offspring_payoffs = mean(offspring_payoffs)
    ## This should show that the underlying reproduction functions are selecting new genotypes with differing payoffs
    print("After one round of reproduction, the mean payoff changed from $mean_parent_payoffs to $mean_offspring_payoffs \n\n")

    ########################################################################################
    ## Testing the main simulation loop for a change in population payoffs
    ########################################################################################
    print("########################################################################################\n")
    print("Testing the main NetworkGameFuncs.jl simulation() loop on a hyper-mutated population\n")
    print("########################################################################################\n\n")
    ## Now we'll show that the reproduction function is raising the average payoff
    sample_generations = sample_generations
    pop = population_construction(pop.parameters)
    pop.parameters.b = b
    pop.parameters.c = c
    pop.parameters.activation_scale = activation_scale
    update_population!(pop)
    social_interactions!(pop)
    initial_payoff = round(copy(mean(pop.payoffs)), digits = 5)
    initial_coop = round(copy(mean(pop.cooperation_vals)), digits = 5)
    print("_t_|_w_\n")


    for μ in 1:mutagenic_events
        mutate!(pop)
        update_population!(pop)
    end

    for t in 1:sample_generations
        update_population!(pop)
        social_interactions!(pop)
        reproduce!(pop)
        mutate!(pop)
        if mod(t, sample_generations/10) == 0
            temp_payoff = round(mean(pop.payoffs), digits = 3)
            temp_coop = round(mean(pop.cooperation_vals), digits = 3)
            print(" $t | $temp_payoff | $temp_coop \n")
        end
        if t == 1
            temp_payoff = round(mean(pop.payoffs), digits = 3)
            temp_coop = round(mean(pop.cooperation_vals), digits = 3)
            print(" $t | $temp_payoff | $temp_coop \n")
        end
    end
    final_payoff = round(copy(mean(pop.payoffs)), digits = 5)
    final_cooperation = round(copy(mean(pop.cooperation_vals)), digits = 5)
    print("\nUsing the functions from NetworkGameCoop.jl's simulation() loop, \n 
    the mean payoff changed from $initial_payoff to $final_payoff \n
    the mean cooperation changed from $initial_coop to $final_cooperation \n
    over $sample_generations generations \n
    \n")

    ## Rerunning the above in the main simulation loop

    pop = copy(initial_population)
    pop.parameters.tmax = sample_generations
    pop.parameters.nnet = nnet
    pop.parameters.b = b
    pop.parameters.c = c
    pop.parameters.activation_scale = activation_scale
    pop.parameters.output_save_tick = 1
    initial_payoff = mean(pop.payoffs)


    outputs = simulation(pop)

    output_payoff = outputs[sample_generations, :mean_payoff]

    print("\n Using the main simulation() function, the mean payoff changed from $initial_payoff to $output_payoff")
    
    ## Generating response curve for all individuals

    network_iteration_outputs = []
    for n in 1:pop.parameters.N
        temp = []
        for i in 0.0:0.01:1.0
            push!(temp, iterateNetwork(pop.parameters.activation_function, pop.parameters.activation_scale, i, pop.networks[n].Wm, pop.networks[n].Wb, pop.temp_arrays.prev_out)[2])
        end
        push!(network_iteration_outputs, temp)
    end
    out_dict["network_iteration_outputs"] = network_iteration_outputs
    return out_dict
end ## end main FunctionTest.jl file

out = main(1.0, 0.0, "linear", 2, 1.0, 10000)

## Quick plot of the results of each round
plt = plot(title = out["activation_function"], ylabel = "Cooperation", xlabel = "Round", legend = false)
for n in 1:out["N"]
    plt= plot!(1:out["rounds"], out["p1_rounds"][n])
end
plt

net_iter_plot = plot(title = out["activation_function"], ylabel = "Output", xlabel = "Input", legend = false)
for n in 1:out["N"]
    net_iter_plot = plot!(0.0:0.01:1.0, out["network_iteration_outputs"][n])
end
net_iter_plot