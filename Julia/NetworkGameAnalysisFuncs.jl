## functions necessary to create and analyze results from network game sims

using DataFrames, JLD2, StatsPlots, Statistics, Plots, ColorSchemes

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


## finds the names of all columns extracted from simulation_output
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

## constructs a dictionary of column vectors extracted from an instance of
## the simulation_output struct to be passed to the dataframe
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

## need to turn parameter values into columns. This function should be robust to updates of the 
## simulation_parameters struct, but will introduce issues if different versions are analyzed together.
## place files in seperate directories if this is necessary
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

## wasn't sure how to define a preallocation on these, leaving it undef for append
## speed isn't essential when working with <1800 replicates. Unless future sims
## are several orders of magnitude higher, the push! vector construction is fine.
function create_df(files::Vector{JLD2.JLDFile})

    ## preallocating  DF
    df = DataFrame()

    # n_replicates = find_n_rows(files)


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

## creates violin plots from a grouped dataframe
function create_all_violin_plots(gdf)
    for group in gdf
        plt = plot(create_mean_w_violin_plots(group), create_mean_init_violin_plots(group), layout=(2,1))
        b = replace(string(group[!, :b][1]), "."=>"0")
        c = replace(string(group[!, :c][1]), "."=>"0")
        tmax = replace(string(group[!, :tmax][1]), "."=>"0")
        filename = string("mean_init_and_fitness", "_b_", b, "_c_", c, "_tmax_", tmax)
        savefig(plt, filename)
    end
end

## returns a heightmap of all edge->edge connections at the end of each replicate in a given group 
## I'm not putting an EXPLICIT control for what happens when the group contains
## differing values of nnet, but passing something like that will likely break your analysis!!

## This plots the node weights on the diagonal of the edge weights!
function network_heatmap(group::SubDataFrame)
    # max_net = ma
    output_wm = zeros(Float64, (group[1, :nnet], group[1, :nnet]))
    output_wb = zeros(Float64, group[1, :nnet])
    reps = 0
    for replicate in eachrow(group)
        output_wm .+= last(replicate.mean_net_history).Wm
        output_wb .+= last(replicate.mean_net_history).Wb
        reps += 1
    end
    output_wm ./= reps
    output_wb ./= reps
    for wb_i in 1:length(output_wb)
        output_wm[wb_i,wb_i] = output_wb[wb_i]
    end
    gr()
    title = string("b = ", string(group[1, :b]), ", c = ", string(group[1, :c]))
    # plt1 = 
    # plt2 = 
    return heatmap(output_wm, c = :RdBu_11, yflip = true,  clim = (-1,1), title = title)
end



