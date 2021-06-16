## This file runs the Feed Forward Network Game functions across initial
## frequencies without selection to compare the population simulation with theory.
## For a given parameter set, it iterates the desired number of replicates/timesteps
## over different initial frequencies. 


## Starting distributed computing

using Distributed

addprocs(8, topology=:master_worker, exeflags="--project=$(Base.active_project())")
@everywhere using ArgParse, JLD2
@everywhere begin
    #instantiating environment
    using Pkg; Pkg.instantiate()
    Pkg.activate(@__DIR__)
    #loading dependencies
    using ArgParse, JLD2


    include("NetworkGameFuncs.jl")
    function parameter_output_map(parameters::simulation_parameters)
        replicate_parameters = copy(parameters)
        return simulation(population_construction(replicate_parameters))
    end

    function RunReplicates(parameters::simulation_parameters)
        ## generation of initial network

        # rep_outputs = repeat([parameters], parameters.nreps)
        # for rep in 1:parameters.nreps
        #     init_pop = population_construction(parameters)
        #     rep_outputs[rep] = simulation(init_pop)
        # end
        # pmap(x->parameter_output_map(x) : x, rep_output)
        rep_outputs = pmap(parameter_output_map, repeat([parameters], parameters.nreps))
        nreps = length(rep_outputs)
        print("$nreps replicates completed in: ")
        return rep_outputs
    end

    ###################
    #      main       #
    ###################
    function main()

        ##########
        ## Arg Parsing
        ##########

 
        
        parameters = initial_arg_parsing()

        ## setting iterator of population frequency
        ps = collect(0.0:parameters.init_freq_resolution:1.0)
        print("Starting replicates", "\n")

        ## initializing output array
        sim_outputs = Vector(undef, 0)

        for (p, q) in zip(ps, reverse(ps))

            print("p = ", p, "\n")

            ## creating savable copy of the parameters
            parameters.init_freqs = [p, q] 
            replicate_parameters = copy(parameters)
            # ###################
            # # Simulation call #
            # ###################
            @time current_reps = RunReplicates(replicate_parameters)

            push!(sim_outputs, current_reps)
            
        ## end of init_freq iteration loop
        end

    jldsave(parameters.filename; sim_outputs)
    ###################
    #   Data Output   #     
    ###################

    # sim data will be stored in previous section and saved to disk here. 
    print("Simulation done in ")
    end

end
@time main()

