## This file will use the functions and structures in the accompanying files to
## evolve networks under Wright-Fisher selection. It replicates this accross different network sizes, as well



using Distributed


addprocs(4, topology=:master_worker, exeflags="--project=$(Base.active_project())")
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
        sim_outputs = Vector(undef, 0)
        parameters = initial_arg_parsing() 
        n_workers = nworkers()
        print("Starting replicates with $n_workers processes", "\n")
        
        for net_size in collect(1:2:18)

            replicate_parameters = copy(parameters)
            replicate_parameters.nnet = net_size
            ## setting iterator of population frequency


            @time current_reps = RunReplicates(replicate_parameters)
            push!(sim_outputs, current_reps)
        end
    
    output_filename = replace(parameters.filename, ".jld2"=>"")
    # b_val = replace(string(parameters.b), "." => "0")
    # c_val = replace(string(parameters.c), "."=>"0")
    parameters.filename = string(output_filename, "_b_", replace(string(parameters.b), "." => "0"), "_c_", replace(string(parameters.c), "."=>"0"), ".jld2")
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

