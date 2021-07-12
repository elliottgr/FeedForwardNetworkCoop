## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

using DataFrames, JLD2

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
    output_column_names = Vector{Symbol}(undef, 0)
    for output in fieldnames(simulation_output)
        push!(output_column_names, output)
    end
    return output_column_names
end

function find_n_rows(files::Vector{JLD2.JLDFile})
    n_replicates = Int64(0)
    for file in files
        data = get_experiment(file)
        for experiment in data
            if experiment[1].parameters.nreps == length(experiment)
                n_replicates += experiment[1].parameters.nreps
            else
                print("Missing data: dataframe is missing some replicates")
            end
        end
    end
    return n_replicates
end

function get_df_dict(files::Vector{JLD2.JLDFile})
    
    fixations = Vector{Vector{Int64}}(undef, 0)
    n_genotypes = Vector{Vector{Int64}}(undef, 0)
    w_mean_history = Vector{Vector{Float64}}(undef, 0)
    init_mean_history = Vector{Vector{Float64}}(undef, 0)
    mean_net_history = Vector{Vector{network}}(undef, 0)
    parameters = Vector{simulation_parameters}(undef, 0)
    df_dict = Dict([
                (Symbol(:fixations), fixations),
                (Symbol(:n_genotypes), n_genotypes),
                (Symbol(:w_mean_history), w_mean_history),
                (Symbol(:init_mean_history), init_mean_history),
                (Symbol(:mean_net_history), mean_net_history),
                (Symbol(:parameters), parameters)])
    for file in files
        for experiment in get_experiment(file)
            for rep in experiment
                for col in get_column_names()
                        push!(df_dict[col], getproperty(rep, col))
                end
            end
        end
    end
    return df_dict
end

## 7/9/21 2nd try at creating a dataframe of values. 
## going to be 1 row = 1 replicate, try to extract from there for plotting
function create_df(files::Vector{JLD2.JLDFile})


    ## preallocating arrays for DF, unused
    # n_replicates = find_n_rows(files)

    ## wasn't sure how to define a preallocation on these, leaving it undef for append
    
    df_dict = get_df_dict(files)

    
end


function main()
    files = Vector{JLD2.JLDFile}(undef, 0)
    for file in readdir()
        if last(splitext(file)) == ".jld2"
            print(file)
            push!(files, jldopen(file))
        end
    end
    filestr = "NetworkGameTests_b_200_c_100_tmax_100000.jld2"

    # output = jldopen(file)
    create_df(files)
end

main()