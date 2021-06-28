## This file will use the functions and structures in the accompanying files to
## evolve networks under Wright-Fisher selection. 



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
        n_workers = nworkers()
        print("Starting replicates with $n_workers processes", "\n")

        ## initializing output array
        sim_outputs = Vector(undef, 0)
        @time current_reps = RunReplicates(replicate_parameters)
        push!(sim_outputs, current_reps)
            
    parameters.filename = "NetworkGameCoop.jld2"
    jldsave(parameters.filename; sim_outputs)
    ###################
    #   Data Output   #     
    ###################

    # sim data will be stored in previous section and saved to disk here. 
    print("Simulation done in ")
    end

end
@time main()

