## functions necessary to create and analyze results from network game sims

using DataFrames, JLD2, StatsPlots, Statistics, Plots

include("NetworkGameStructs.jl")

###############
## File Manip Functions
###############
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
    split_parameters(df_dict)
    return df_dict
end

## need to turn parameter values into columns
function split_parameters(df_dict::Dict)
    # param_names = Vector{Symbol}(undef, 0)
    # print(keys(df_dict))
    # parameters = df_dict[:parameters][1]
    for param in fieldnames(simulation_parameters)
        temp_array = Vector(undef, 0)
        for param_entry in df_dict[:parameters]
            push!(temp_array, getproperty(param_entry, param))
        end
        df_dict[param] = temp_array
        # push!(param_names, param)
    end
end

## 7/9/21 2nd try at creating a dataframe of values. 
## going to be 1 row = 1 replicate, try to extract from there for plotting
function create_df(files::Vector{JLD2.JLDFile})


    ## preallocating arrays for DF, unused
    df = DataFrame()

    # n_replicates = find_n_rows(files)

    ## wasn't sure how to define a preallocation on these, leaving it undef for append
    
    for (key, value) in get_df_dict(files)
        df[!, Symbol(key)] = value
    end

    return df
end


## for a DataFrame group, returns figures similar to those created in cooperateion_analysis.jl
## and JVC's original grant proposal. Implementation creates two dataframes because I copied it from
## cooperation analysis rather than directly applying it to the gdf created elsewhere.
function create_mean_w_violin_plots(group::SubDataFrame)
    nnets = zeros(Int64, 0)
    w_means = zeros(Float64, 0)
    for replicate in eachrow(group)
        push!(nnets, replicate[:nnet])
        push!(w_means, last(replicate[:w_mean_history]))
    end
    temp_df = DataFrame(nnet=nnets, w_mean = w_means)
    @df temp_df violin(:nnet, :w_mean, title = "Mean Fitness", legend = :none)
    @df temp_df boxplot!(:nnet, :w_mean, fillalpha=.6)
end


##similar to above, creates plots for a given subdataframe.
function create_mean_init_violin_plots(group::SubDataFrame)
    nnets = zeros(Int64, 0)
    inits = zeros(Float64,0)
    for replicate in eachrow(group)
        push!(nnets, replicate[:nnet])
        push!(inits, last(replicate[:init_mean_history]))
    end
    temp_df = DataFrame(nnet=nnets, inits = inits)
    @df temp_df violin(:nnet, :inits, title = "Mean Initial Offer", legend = :none)
    @df temp_df boxplot!(:nnet, :inits, fillalpha=.6)
end
