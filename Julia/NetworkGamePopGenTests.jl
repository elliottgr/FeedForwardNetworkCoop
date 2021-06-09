## This file runs the Feed Forward Network Game functions across initial
## frequencies without selection to compare the population simulation with theory.
## For a given parameter set, it iterates the desired number of replicates/timesteps
## over different initial frequencies. 
using ArgParse, JLD2

include(NetworkGameFuncs.jl)

###################
#      main       #
###################
function main()

    ##########
    ## Arg Parsing
    ##########

    arg_parse_settings = ArgParseSettings()
    @add_arg_table arg_parse_settings begin

        ########
        ## Population Simulation Parameters
        ########
        "--tmax"
            help = "Maximum number of timesteps"
            arg_type = Int64
            default = 100
        "--nreps"
            help = "number of replicates to run"
            arg_type = Int64
            default = 10

        "--N"   
            help = "population size"
            arg_type = Int64
            default = 100
        "--mu"
            help = "mutation probability per birth"
            arg_type = Float64
            default = 0.0

        ########
        ## Game Parameters
        ########
        "--rounds"
            help = "number of rounds the game is played between individuals"
            arg_type = Int64
            default = 5

        "--fitness_benefit_scale"
            help = "scales the fitness payout of game rounds by this amount (payoff * scale)"
            arg_type = Float64
            default = 1.0

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
            help = "relatedness coefficient"
            arg_type = Float64
            default = 0.0
        "--delta"
            help = "payoff discount, negative values use last round"
            arg_type = Float64
            default = 0.0
        "--init_freqs"
            help = "vector of initial genotype frequencies, must sum to 1"
            arg_type = Vector{Float64}
            default = [0.50,0.5]
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
            default = 0.01
        "--mutlink"
            help = "Probability that a random edge or node be altered in a mutation event"
            arg_type = Float64
            default = 0.5
        ########
        ## File Parameters
        ########
        "--filename"
            help = "File to save outputs to"
            arg_type = String
            default = "NetworkGameOutput.jld2"
    end
    
    ##passing command line arguments to simulation
    parsed_args = parse_args(ARGS, arg_parse_settings)
    parameters = simulation_parameters(parsed_args["tmax"], parsed_args["nreps"], parsed_args["N"], parsed_args["mu"],
                                        parsed_args["rounds"], parsed_args["fitness_benefit_scale"], parsed_args["b"], 
                                        parsed_args["c"], parsed_args["d"], parsed_args["delta"], parsed_args["init_freqs"], 
                                        parsed_args["nnet"], parsed_args["mutsize"], parsed_args["mutinitsize"], parsed_args["mutlink"],
                                        parsed_args["filename"])

    ## Necessary sanity checks for params
    if mod(parameters.N, 2) != 0
        print("Please supply an even value of N!")
    end


    ##################################
    #Generation of Random Initial Network
    ##################################
    ## generates random networks based on simulation parameters
    init_pop = population_construction(parameters)

    ###################
    # Simulation call #
    ###################

    ## need to define vectors over sim outputs so I can preallocate some memory :)
    sim_outputs = Vector(undef, parameters.nreps)
    for rep in 1:parameters.nreps
        sim_outputs[rep] = simulation(init_pop)
    end


    ###################
    #   Data Output   #
    ###################

    ## sim data will be stored in previous section and saved to disk here. 
    jldsave(parameters.filename; sim_outputs)


end


main()

