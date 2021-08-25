## This file will use the functions and structures in the accompanying files to
## evolve networks under Wright-Fisher selection. It replicates this accross different network sizes, as well



using Distributed, Random


addprocs(20, topology=:master_worker, exeflags="--project=$(Base.active_project())")
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
        print("Network Size = $(parameters.nnet)", "\n")
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

        
            ## initializing output array

        parameters = initial_arg_parsing() 
        # sim_outputs = Vector(undef, 0)
        # sim_outputs = 
        Random.seed!(parameters.seed)
        n_workers = nworkers()
        print("Starting replicates with $n_workers processes", "\n")
        for b in collect(parameters.game_param_min:parameters.game_param_step:parameters.game_param_max)
            replicate_parameters = copy(parameters)
            replicate_parameters.b = b
            for c in collect(parameters.game_param_min:parameters.game_param_step:parameters.game_param_max)
                replicate_parameters.c = c
                for net_size in collect(parameters.nnet_min:parameters.nnet_step:parameters.nnet_max)
                    replicate_parameters.nnet = net_size
                    @time current_reps = RunReplicates(replicate_parameters)
                    push!(sim_outputs, current_reps)
                end
            end
        end
    output_filename = replace(parameters.filename, ".jld2"=>"")
    parameters.filename = string(output_filename, "b_c_min_", replace(string(parameters.game_param_min), "." => "0"), "b_c_max", replace(string(parameters.game_param_max), "." => "0"), "_nreps_", parameters.nreps, "_tmax_", parameters.tmax, ".jld2")
        # parameters.filename = "NetworkGameCoop.jld2"

    jldsave(parameters.filename; sim_outputs)
    ###################
    #   Data Output   #     
    ###################

    # sim data will be stored in previous section and saved to disk here. 
    print("Simulation done in ")
    end

end
@time main()

