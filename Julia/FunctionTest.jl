## This is a demo file to make sure that the population genetics functions in NetworkGameFuncs.jl are behaving as 
## expected. This may also prove useful if someone needs to see how the functions are being used
## in the main simulation



## include necessary files

include("NetworkGameFuncs.jl")




function main()

## Import dummy parameters, defining global variables used in main files
parameters = initial_arg_parsing()
global discount = calc_discount(parameters.δ, parameters.rounds)
global discount = SVector{parameters.rounds}(discount/sum(discount))

## Setting the network size of the testing populations
parameters.nnet = 5

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
end

## Resetting the population to check if game results actually impact reproduction
pop = copy(initial_population)

## Setting b = 1.0, leaving c = 0.0 to encourage cooperation
pop.parameters.b = 1.0
pop.parameters.c = .5
pop.parameters.activation_scale = 100.0
## Mutating the population a lot to differentiate genotypes

mutagenic_events = 500
for μ in 1:mutagenic_events
    mutate!(pop)
    update_population!(pop)
end

## Testing the fitness results of the first genotype and its shuffled partner
update_population!(pop)
g1 = pop.genotypes[1]
n1 = pop.networks[1]
g2 = pop.genotypes[pop.shuffled_indices[1]]
n2 = pop.networks[pop.shuffled_indices[1]]
g3 = pop.genotypes[pop.shuffled_indices[pop.shuffled_indices[1]]]
n3 = pop.networks[pop.shuffled_indices[pop.shuffled_indices[1]]]
print("\n\n ########################################################################\n")
print("## Randomly Sampled Network Stats ## \n ")
print("########################################################################\n\n")
print("Genotype 1: $g1 \n Network Weights: $n1 \n\n")
print("Genotype 2: $g2 \n Network Weights: $n2 \n\n")
print("Genotype 3: $g3 \n Network Weights: $n3 \n\n")

if pop.networks[1] == pop.networks[pop.shuffled_indices[1]]
    print("You're comparing the same network here :(")
end

## fitnessOutcome as calculated in main simulation
## This section shows that the fitness of G2 is not determined by the fitness of the G1 and G2 matchup
## Instead, G2s fitness is determined by 
fitness_test1 = fitnessOutcome(pop.parameters,  pop.networks[pop.shuffled_indices[1]], pop.networks[1], pop.temp_arrays)
print(fitness_test1)
print("\n\n########################################################################\n")
print(" ## Testing fitness outcomes and their storage in the population array ## \n")
print("########################################################################\n\n")

if fitness_test1[1][1] != pop.fit_dict[g1][g2]
    dictionary_value = pop.fit_dict[g1][g2]
    expected_value = fitness_test1[1][1]
    diff = dictionary_value - expected_value
    percent = round(100*(diff/dictionary_value), digits = 3)
    print("G1 (Genotype #$g1) is not saving the exact fitness in the fitness dictionary \n")
    print("Dictionary Value = $dictionary_value \n Expected Value = $expected_value \n")
    print("Total Difference = $diff, a $percent% difference!")
end

fitness_test2 = fitnessOutcome(pop.parameters, pop.networks[1], pop.networks[pop.shuffled_indices[1]], pop.temp_arrays)
 print(fitness_test2)
print("\n")



## This part does not work like you'd expect under Wright-Fisher because of how the fitness dictionary is constructed
## partners in the simulation are only one way, rather than each partner calculating its fitness based on each
## partner. Not sure if this is an issue or explains why the evolution of networks isn't occurring 
if fitness_test2[1][2] != pop.fit_dict[g2][g3]
    dictionary_value = pop.fit_dict[g2][g3]
    expected_value = fitness_test1[1][2]
    diff = dictionary_value - expected_value
    print("\n\n G2's fitness is NOT determined by the fitness derived from the G1 vs G2 game \n")
    print("Dictionary Value = $dictionary_value \n Expected Value = $expected_value \n\n Difference = $diff \n\n")
end

## Calculating the mean difference caused by this effect over the entire population

difference_array = []
for x in 1:pop.parameters.N
    expected_outcome = fitnessOutcome(pop.parameters, pop.networks[pop.shuffled_indices[x]], pop.networks[x], pop.temp_arrays)[1][2]
    realized_outcome = fitnessOutcome(pop.parameters, pop.networks[pop.shuffled_indices[pop.shuffled_indices[x]]], pop.networks[pop.shuffled_indices[x]], pop.temp_arrays)[1][1]
    push!(difference_array, (expected_outcome-realized_outcome))
end

mean_error = mean(difference_array)
max_error = maximum(difference_array)
print("This programming choice has resulted in a mean calculated fitness error of $mean_error, \n 
with a maximum error of $max_error  \n\n")

## Using G1 and G2 to show the difference of payoffs over multiple rounds

example_prev_out = @MVector zeros(Float64, parameters.nnet) 
outcome = repeatedNetworkGame(pop.parameters, pop.networks[pop.shuffled_indices[1]], pop.networks[1], example_prev_out)
g1_post_round = outcome[1]
g2_post_round = outcome[2]
n_rounds = pop.parameters.rounds
print("########################################################################\n")
print("You can see here the results of G1 (Genotype # $g1) and G2 (Genotype # $g2) over $n_rounds round:\n")
print("########################################################################\n\n")
print("G1 Results: $g1_post_round \n\n")
print("G2 Results: $g2_post_round \n\n")

end ## end main FunctionTest.jl file

main()