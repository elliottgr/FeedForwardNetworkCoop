## This file will use the functions and structures in the accompanying files to
## evolve networks under Wright-Fisher selection. It replicates this accross different network sizes, as well



using Distributed, Random, InteractiveUtils

addprocs(1, topology=:master_worker, exeflags="--project=$(Base.active_project())")

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
        return DataFrames.vcat(rep_outputs..., cols = :union)
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

        b_vals = collect(parameters.game_param_min:parameters.game_param_step:parameters.game_param_max)
        c_vals = collect(parameters.game_param_min:parameters.game_param_step:parameters.game_param_max)
        nnet_vals = collect(parameters.nnet_min:parameters.nnet_step:parameters.nnet_max)
            
        sim_outputs = Vector{DataFrame}(undef, 0)
        Random.seed!(parameters.seed)


        n_workers = nworkers()
        print("Starting replicates with $n_workers processes", "\n")
        i = 1

        ## variables to handle larger datasets
        multi_file_flag = false
        n_files = 0
        n_replicate_sets = length(b_vals) * length(c_vals) * length(nnet_vals)
        rep_set_i = 0
        for b in b_vals
            replicate_parameters = copy(parameters)
            replicate_parameters.b = b
            for c in c_vals
                replicate_parameters.c = c
                for net_size in nnet_vals
                    rep_set_i += 1
                    print("\n Starting replicate set $rep_set_i of $n_replicate_sets \n")
                    replicate_parameters.nnet = net_size
                    @time push!(sim_outputs, RunReplicates(replicate_parameters))

                    ## 25 gigabytes as max filesize
                    ## should be safe for dispatched workers since they're called on replicate sets anyway
                    if Base.summarysize(sim_outputs) >= 25 * (1000)^3 
                        multi_file_flag = true
                        n_files += 1
                        parameters.filename = string(output_filename, "b_c_min_", replace(string(parameters.game_param_min), "." => "0"), "b_c_max", replace(string(parameters.game_param_max), "." => "0"), "_nreps_", parameters.nreps, "_tmax_", parameters.tmax, "part_", n_files, ".jld2")
                        jldsave(parameters.filename; sim_outputs)
                        sim_outputs = Vector{DataFrame}(undef, 0)
                    end
                    i+=1
                end
            end
        end
    output_filename = replace(parameters.filename, ".jld2"=>"")
    parameters.filename = string(output_filename, "b_c_min_", replace(string(parameters.game_param_min), "." => "0"), "b_c_max", replace(string(parameters.game_param_max), "." => "0"), "_nreps_", parameters.nreps, "_tmax_", parameters.tmax, ".jld2")
    output_df = vcat(sim_outputs..., cols = :union)
    if multi_file_flag == false
        jldsave(parameters.filename; output_df)
    end
    ###################
    #   Data Output   #     
    ###################

    # sim data will be stored in previous section and saved to disk here. 
    print("Simulation done in ")
    end

end
@time main()

