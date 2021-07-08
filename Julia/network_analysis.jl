## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

using DataFrames

include("NetworkGameStructs.jl")

function get_experiment(file::JLD2.JLDFile, experiment=nothing, replicate=nothing) 
    if experiment == nothing
        return file["sim_outputs"]
    elseif typeof(experiment) == Int64
        if replicate == nothing
            return file["sim_outputs"][experiment]
        ## not sure the best way to do the error handling here
        elseif typeof(replicate) == Int64
            return get_experiment(file, experiment)[replicate]
        end
    end
end


## finds the names of all dataframe columns for pre-allocation,
## since it just uses the type definition, it doesn't need input
function get_column_names()
    param_column_names = Vector{Symbol}(undef, 0)
    output_column_names = Vector{Symbol}(undef, 0)
    for parameter in fieldnames(simulation_parameters)
        push!(param_column_names, parameter)
    end
    for parameter in fieldnames(simulation_output)
        push!(output_column_names, parameter)
    end
    return [param_column_names, output_column_names]
end


## creates a dataframe containing each timestep of all replicates
function create_df(files::Vector{JLD2.JLDFile})
    ## need starting variables for preallocation
    ## max_nnet will break if network structures are changed!!!!!!!!!!!!!
    nrows = Int64(0)
    max_nnet = Int64(0)

    for file in files
        data = get_experiment(file)
        param_column_names, output_column_names = get_column_names()

        for experiment in data
            nrows += (experiment[1].parameters.nreps*experiment[1].parameters.tmax)
            max_nnet = maximum([max_nnet, experiment[1].parameters.nnet])
        end 

        ## creating column names for all nodes and node-node (edge ID) permutations 
        for i in 1:max_nnet
        
        end
        for experiment_i in 1:length(data)
            for replicate_i in 1:length(data[experiment_i])

            end
        end
    end
    print(nrows)
    print(max_nnet)
end


function main()
    files = Vector{JLD2.JLDFile}(undef, 0)
    for file in readdir()
        if last(splitext(file)) == ".jld2"
            push!(files, jldopen(file))
        end
    end
    filestr = "NetworkGameTests_b_200_c_100_tmax_100000.jld2"

    # output = jldopen(file)
    create_df(files)
end

main()