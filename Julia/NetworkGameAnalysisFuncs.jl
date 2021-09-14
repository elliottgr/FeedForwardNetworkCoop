## functions necessary to create and analyze results from network game sims

using DataFrames, JLD2, StatsPlots, Statistics, Plots, ColorSchemes, LinearAlgebra, Random, ArgParse, Dates

include("NetworkGameStructs.jl")

mutable struct analysis_parameters
    k::Int64
    max_rows::Int64
    use_random::Bool
    t_start::Int64
    t_end::Int64
    output_folder::String
    filepath::String
end


###############
## File Manip Functions
###############


## creates a log file to save parameters from specific runs of the analysis file

function create_directories(analysis_params::analysis_parameters, folder_name)
    analysis_params.filepath = string(analysis_params.output_folder, "/", folder_name )
    if isdir(string(pwd(), "/", analysis_params.output_folder)) == false
        print("Output directory not found, creating a new one at ", string(pwd(), "/", analysis_params.output_folder))
        mkdir(string(pwd(), "/", analysis_params.output_folder))
    end
    if isdir(string(pwd(), "/", analysis_params.filepath, "/")) == false
        mkdir(string(pwd(), "/", analysis_params.filepath))
    end
    if isdir(string(pwd()*"/"*analysis_params.filepath*"/b_c_coop_heatmaps/")) == false
        mkdir(string(pwd()*"/"*analysis_params.filepath*"/b_c_coop_heatmaps/"))
    end
    if isdir(string(pwd()*"/"*analysis_params.filepath*"/edge_weight_w_heatmaps/")) == false
        mkdir(string(pwd()*"/"*analysis_params.filepath*"/edge_weight_w_heatmaps/"))
    end
    if isdir(string(pwd()*"/"*analysis_params.filepath*"/violin_plots/")) ==false
        mkdir(string(pwd()*"/"*analysis_params.filepath*"/violin_plots/"))
    end
    if isdir(string(pwd()*"/"*analysis_params.filepath*"/time_series/")) ==false
        mkdir(string(pwd()*"/"*analysis_params.filepath*"/time_series/"))
    end
end

function create_log_file(df::DataFrame, analysis_params::analysis_parameters)
    logfilestr = string(analysis_params.filepath, "/parameter_info_", string(now()), ".txt")
    io = open(logfilestr, "w")
    println(io, "log file for network_analysis.jl")
    println(io, now())
    println(io, "simulation_parameters (NetworkGameCoop.jl):")
    println(io, "#####################")
    for param in fieldnames(simulation_parameters)
        println(io, string(param, " = ", unique(df[!, param])))
    end
    println(io,"")
    println(io, "analysis_parameters (network_analysis.jl):")
    println(io, "#####################")
    for param in fieldnames(analysis_parameters)
        println(io, string(param, " = ", getproperty(analysis_params, param)))
    end
    close(io)
end

function load_files()
    files = Vector{JLD2.JLDFile}(undef, 0)
    for file in readdir()
        if last(splitext(file)) == ".jld2"
            print(file)
            push!(files, jldopen(file))
            folder_name = first(splitext(file))
        end
    end
    if length(files) > 1
        folder_name = "multiple_datasets"
    end
    return [files, folder_name]
end

function generate_output_directory(analysis_params::analysis_parameters, files::Vector{JLD2.JLDFile})
    if length(files) > 1
        folder_name = "multiple_datasets"
    else
        folder_name = splitext(files[1])[1]
    end
    return folder_name
end

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


## going to try a more optimized "get_df_dict"
function get_df_dict(files::Vector{JLD2.JLDFile}, analysis_params::analysis_parameters)

    ## iterating over all replicates to get the number of rows to generate
    n_rows = 0
    for file in files
        for file in files
            for experiment in get_experiment(file)
                for rep in experiment
                    n_rows += 1
                end
            end
        end
    end
    
    ## creating columns to be filled
    fixations = fill(zeros(Int64,0), n_rows)
    n_genotypes = fill(zeros(Int64,0), n_rows)
    w_mean_history =  fill(zeros(Float64,0), n_rows)
    init_mean_history =  fill(zeros(Float64,0), n_rows)
    coop_mean_history = fill(zeros(Float64, 0), n_rows)
    mean_net_history = fill(Vector{output_network}(undef, 0), n_rows)
    parameters = Vector{simulation_parameters}(undef, n_rows)
    payoff_mean_history = fill(zeros(Float64, 0), n_rows)
    timestep = fill(zeros(Int64,0), n_rows)
    rep_id = zeros(Int64, n_rows)
    df_dict = Dict([
                (Symbol(:fixations), fixations),
                (Symbol(:n_genotypes), n_genotypes),
                (Symbol(:w_mean_history), w_mean_history),
                (Symbol(:coop_mean_history), coop_mean_history),
                (Symbol(:init_mean_history), init_mean_history),
                (Symbol(:mean_net_history), mean_net_history),
                (Symbol(:parameters), parameters),
                (Symbol(:timestep), timestep),
                (Symbol(:rep_id), rep_id),
                (Symbol(:payoff_mean_history), payoff_mean_history)])
    
    rep_i = 0
    column_names = get_column_names()
    for file in files
        for experiment in get_experiment(file)
            for rep in experiment
                rep_i += 1
                df_dict[:timestep][rep_i] = collect(Int64, analysis_params.t_start:1:length(rep.fixations))
                ## not necessary in this implementation but don't want to break other stuff
                df_dict[:rep_id][rep_i] = rep_i

                for col in column_names
                    ## need to filter things where t_start:end doesnt work
                    if typeof(getproperty(rep,col)) == Vector
                        df_dict[col][rep_i] = getproperty(rep,col)[analysis_params.t_start:analysis_params.t_end]
                    else
                        df_dict[col][rep_i] = getproperty(rep,col)
                    end
                end
            end
        end
    end
    split_parameters(df_dict, analysis_params)
    return df_dict
end

## need to turn parameter values into columns. This function should be robust to updates of the 
## simulation_parameters struct, but will introduce issues if different versions are analyzed together.
## place files in seperate directories if this is necessary
function split_parameters(df_dict::Dict, analysis_params::analysis_parameters)
    for param in fieldnames(simulation_parameters)
        temp_array = Vector(undef, length(df_dict[:parameters]))
        i = 0
        for param_entry in df_dict[:parameters]
            i+=1
            ## need to filter things where t_start:end doesnt work
            if typeof(getproperty(param_entry,param)) == Vector
                temp_array[i] = getproperty(param_entry, param)[analysis_params.t_start:analysis_params.t_end]
            else
                temp_array[i] = getproperty(param_entry, param)
            end
        end
        df_dict[param] = temp_array

    end
end



## 7/9/21 2nd try at creating a dataframe of values. 
## going to be 1 row = 1 replicate, try to extract from there for plotting

## wasn't sure how to define a preallocation on these, leaving it undef for append
## speed isn't essential when working with <1800 replicates. Unless future sims
## are several orders of magnitude higher, the push! vector construction is fine.
function create_df(files::Vector{JLD2.JLDFile}, analysis_params::analysis_parameters)

    ## creating  DF
    df = DataFrame()

    for (key, value) in get_df_dict(files, analysis_params)
        df[!, Symbol(key)] = value
    end

    return df
end

## takes a subgroup from the existing dataframe created by create_df()
## iterates over that array, creating a second dataframe from that (inefficient but didn't feel like rewriting the whole thing)
## one entry corresponds to a single network weight
## edges represented by two columns. e1 is starting column, e2 is end. e2 = 0 means e1 is a node weight
## the "use_random" flag determines whether networks are sampled randomly or in the sorted order of the original DF
function create_edge_df(df_dict::SubDataFrame, analysis_params::analysis_parameters)


    ## df_dict returns a dictionary that is not properly setup
    ## for creating a column of edge weights
    
    
    ## initializing column vectors, preallocating to a set length improves performance here substantially
    rep_id = zeros(Int64, analysis_params.max_rows)
    timestep = zeros(Int64, analysis_params.max_rows)
    b_col = zeros(Float64, analysis_params.max_rows)
    c_col = zeros(Float64, analysis_params.max_rows)
    nnet = zeros(Int64, analysis_params.max_rows)
    edge_weight = zeros(Float64, analysis_params.max_rows)
    e1 = zeros(Int64, analysis_params.max_rows)
    e2 = zeros(Int64, analysis_params.max_rows)
    fitness = zeros(Float64, analysis_params.max_rows)

    ##pre allocating this array to prevent iterating over unused timepoints in the main loop
    # tmax_array = collect(1:df_dict[!, :net_save_tick][1][1]:(maximum(df_dict[!, :timestep])[end])-(analysis_params.t_start+analysis_params.t_end))
    tmax_array = collect(1:df_dict[!, :net_save_tick][1][1]:(analysis_params.t_end - analysis_params.t_start))


    row = 0
    if analysis_params.use_random == true
        network_indices = shuffle(1:length(df_dict[!, :nnet]))
    else
        network_indices = 1:length(df_dict[!, :nnet])
    end
    
    ##iterates over all replicate networks
    for i in network_indices

        ## stops the loop when all rows are used
        if row == analysis_params.max_rows
            break
        end

        ## iterates over all nodes of a network
        for n1 in 1:df_dict[!, :nnet][i]
            for t in tmax_array
                if row < analysis_params.max_rows
                    row += 1
                    e1[row] = n1
                    e2[row] = 0
                    ## this is actually the node weight :)
                    edge_weight[row] = df_dict[!, :mean_net_history][i][t].Wb[n1]
                    rep_id[row] = df_dict[!, :rep_id][i]
                    timestep[row] = df_dict[!, :timestep][i][t]
                    b_col[row] = df_dict[!, :b][i]
                    c_col[row] = df_dict[!, :c][i]
                    nnet[row] = df_dict[!, :nnet][i]
                    fitness[row] = df_dict[!, :w_mean_history][i][t]
                end
                ##Iterates over all edges
                for n2 in n1:df_dict[!, :nnet][i]
                    for t in tmax_array
                        if row < analysis_params.max_rows
                            row+=1
                            e1[row] = n1
                            e2[row] = n2
                            edge_weight[row] = df_dict[!, :mean_net_history][i][t].Wm[n1,n2]
                            rep_id[row] = df_dict[!, :rep_id][i]
                            timestep[row] = df_dict[!, :timestep][i][t]
                            b_col[row] = df_dict[!, :b][i]
                            c_col[row] = df_dict[!, :c][i]
                            nnet[row] = df_dict[!, :nnet][i]
                            fitness[row] = df_dict[!, :w_mean_history][i][t]
                        end
                    end
                end
            end
        end
    end
    return DataFrame(rep_id = rep_id, 
                    timestep = timestep,
                    b = b_col,
                    c = c_col, 
                    nnet = nnet,
                    edge_weight = edge_weight,
                    fitness = fitness,
                    e1 = e1,
                    e2 = e2)
end




## takes a parameter set and creates a nested heatmap of edge-fitness correlations
## main diagonal is node weights, above main diag is node-node (edge) correlations
function correlation_heatmaps(b_c_nnet_group::DataFrame)
    weight_fitness_corr_matrix = zeros(Float64, (b_c_nnet_group[!, :nnet][1],b_c_nnet_group[!, :nnet][1]))
    for edge_group in groupby(b_c_nnet_group, [:e1, :e2])
        if edge_group[!, :e2][1] == 0
            if edge_group[1, :e1] != 0
                weight_fitness_corr_matrix[edge_group[!, :e1][1] , edge_group[!, :e1][1]] = cor(edge_group[!,:edge_weight], edge_group[!,:fitness])
            end
        elseif edge_group[!, :e2][1] < edge_group[!, :e1][1]
            weight_fitness_corr_matrix[edge_group[!, :e1][1] , edge_group[!, :e2][1]] = NaN
        else
            weight_fitness_corr_matrix[edge_group[!, :e1][1] , edge_group[!, :e2][1]] = cor(edge_group[!,:edge_weight], edge_group[!,:fitness])
        end
    end
    return heatmap(weight_fitness_corr_matrix, 
        xlabel = "Node 2", 
        ylabel = "Node 1", 
        title = string("b: " , b_c_nnet_group[1, :b], " c: ", b_c_nnet_group[1, :c]),
        c = :RdBu_9,
        clims = (-0.5, 0.5),
        xticks = (1:1:(b_c_nnet_group[1, :nnet]), (1:1:b_c_nnet_group[1,:nnet])),
        yticks = (1:1:(b_c_nnet_group[1, :nnet]), (1:1:b_c_nnet_group[1,:nnet])),
        yflip = true)
end


## pass the main_df (timepoint rows) to create a n x n heatmap composed of
## edge-fitness corrrelation heatmaps. the larger heatmap axes are b, c params
function create_b_c_heatmap_plot(df, nnet::Int64, analysis_params)
    b_c_vals = unique(df[!,:c])
    heatmaps = Vector{Plots.Plot}(undef,0)
    main_df = df[df.nnet .== nnet, :]

    for group in groupby(main_df, [:b,:c])
        push!(heatmaps,correlation_heatmaps(create_edge_df(group, analysis_params)))
    end
    filestr = pwd()*"/"*analysis_params.filepath*"/edge_weight_w_heatmaps/"*string("fitness_edge_weight_heatmap_nnet_", nnet, "_tstart_", analysis_params.t_start, "_tend_", analysis_params.t_end, ".png")
    savefig(plot(heatmaps..., layout = (length(b_c_vals), length(b_c_vals))), filestr)
end

## for a DataFrame group, returns figures similar to those created in cooperateion_analysis.jl
## and JVC's original grant proposal. Implementation creates two dataframes because I copied it from
## cooperation analysis rather than directly applying it to the gdf created elsewhere.
function create_mean_w_violin_plots(group::SubDataFrame, analysis_params::analysis_parameters)
    nnets = zeros(Int64, 0)
    w_means = zeros(Float64, 0)
    for replicate in eachrow(group)
        push!(nnets, replicate[:nnet])
        push!(w_means, last(rolling_mean(replicate[:w_mean_history][analysis_params.t_start:analysis_params.t_end], analysis_params.k)))
    end
    temp_df = DataFrame(nnet=nnets, w_mean = w_means)
    @df temp_df violin(:nnet, :w_mean, title = "Mean Fitness", legend = :none)
    @df temp_df boxplot!(:nnet, :w_mean, fillalpha=.6)
end


##similar to above, creates plots for a given subdataframe.
function create_mean_init_violin_plots(group::SubDataFrame, analysis_params::analysis_parameters)
    nnets = zeros(Int64, 0)
    inits = zeros(Float64,0)
    for replicate in eachrow(group)
        push!(nnets, replicate[:nnet])
        push!(inits, last(rolling_mean(replicate[:init_mean_history][analysis_params.t_start:analysis_params.t_end], analysis_params.k)))
    end
    temp_df = DataFrame(nnet=nnets, inits = inits)
    @df temp_df violin(:nnet, :inits, title = "Mean Initial Offer", legend = :none)
    @df temp_df boxplot!(:nnet, :inits, fillalpha=.6)
end

## creates violin plots from a grouped dataframe
function create_all_violin_plots(gdf, analysis_params::analysis_parameters)
    for group in gdf
        plt = plot(create_mean_w_violin_plots(group, analysis_params), create_mean_init_violin_plots(group, analysis_params), layout=(2,1))
        b = replace(string(group[!, :b][1]), "."=>"0")
        c = replace(string(group[!, :c][1]), "."=>"0")
        filestr = pwd()*"/"*analysis_params.filepath*"/violin_plots/"*string("mean_init_and_fitness", "_b_", b, "_c_", c, "_tstart_", analysis_params.t_start, "_tend_", analysis_params.t_end, "_k_", string(analysis_params.k))
        savefig(plt, filestr)
    end
end

## returns a heightmap of all edge->edge connections at the end of each replicate in a given group 
## I'm not putting an EXPLICIT control for what happens when the group contains
## differing values of nnet, but passing something like that will likely break your analysis!!

## This plots the node weights on the diagonal of the edge weights!
function network_heatmap(group::SubDataFrame)
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

    return heatmap(output_wm, c = :RdBu_11, yflip = true,  clim = (-1,1), title = title)
end


## from Satvik Beri (https://stackoverflow.com/questions/59562325/moving-average-in-julia)
## by default, only computes the mean after k datapoints, modified here to fill the array with intermediates

function rolling_mean(arr, n)
    so_far = sum(arr[1:n])
    pre_out = zero(arr[1:(n-1)])
    for i in 1:(n-1)
        pre_out[i] = mean(arr[1:i])
    end
    out = zero(arr[n:end])

    ## modification from Beri's script: this doesn't get overwrriten in the below loop
    out[1] = so_far / n
    for (i, (start, stop)) in enumerate(zip(arr, arr[n+1:end]))
        so_far += stop - start
        out[i+1] = so_far / n
    end

    return append!(pre_out, out)
end


## setting this up as the mean of rolling averages sampling "k" timepoints
function create_mean_init_payoff_and_fitness_plots(group::DataFrame, analysis_params::analysis_parameters)
    print("Creating time series plots...", "\n")
    for b in unique(group[!,:b])
        for c in unique(group[!,:c])
            plt_payoff = plot()
            plt_init = plot()
            plt_coop = plot()
            plt_out = plot( plt_init, plt_payoff, plt_coop, layout = (3,1))
            for nnet in unique(group[!,:nnet])
                i = 0
                # tmax = maximum(group[!, :tmax])
                
                fitness_array = zeros(Float64, (analysis_params.t_end - analysis_params.t_start + 1))
                init_array = zeros(Float64, (analysis_params.t_end - analysis_params.t_start + 1))
                coop_array = zeros(Float64, (analysis_params.t_end - analysis_params.t_start + 1))

                ## finding the element wise mean for the conditions
                for replicate in eachrow(subset(group, :b => ByRow(==(b)), :c => ByRow(==(c)), :nnet => ByRow(==(nnet))))
                    ## summing the rolling mean of each replicate
                    fitness_array .+= rolling_mean(replicate.w_mean_history[analysis_params.t_start:analysis_params.t_end], analysis_params.k)
                    init_array .+= rolling_mean(replicate.init_mean_history[analysis_params.t_start:analysis_params.t_end], analysis_params.k)
                    coop_array .+= rolling_mean(replicate.coop_mean_history[analysis_params.t_start:analysis_params.t_end], analysis_params.k)
                    i+=1
                end
                ## dividing sum of replicates by # of reps
                fitness_array ./= i
                init_array ./= i
                coop_array ./= i
                plt_init = plot!(plt_out[1], init_array, label = nnet, title = "Initial Offer")
                plt_payoff = plot!(plt_out[2], fitness_array, label = nnet, title = "Payoff (fitness)")
                plt_coop = plot!(plt_out[3], coop_array, label = nnet, title = "Cooperation")
            end
            filestr = pwd()*"/"*analysis_params.filepath*"/time_series/"*string("mean_w_b_", replace(string(b), "."=>"0"), "_c_", replace(string(c),"."=>"0"), "_tstart_", analysis_params.t_start, "_tend_", analysis_params.t_end, "_k_", string(analysis_params.k))
            savefig(plt_out, filestr)
        end
    end
end

function analysis_arg_parsing()
    global arg_parse_settings = ArgParseSettings()
    @add_arg_table arg_parse_settings begin
        "--k"
            help = "number of datapoints for running mean in time series data"
            arg_type = Int64
            default = 50
        "--max_rows"
            help = "maximum number of edges to sample // setting too high will crash"
            arg_type = Int64
            default = 10000000
        "--use_random"
            help = "Boolean // True = Edges are sampled randomly, False = Edges are sampled in order from simulation"
            arg_type = Bool
            default = true
        "--t_start"
            help = "timestep to begin tracking data for analysis // def = 1"
            arg_type = Int64
            default = 1
        "--t_end"
            help = "timestep to stop tracking data for analysis // def = 100k, do not set higher than tmax from networkgamecoop.jl runs"
            arg_type = Int64
            default = 100000
        "--output_folder"
            help = "folder name to save output plots"
            arg_type = String
            default = "output_figures"
    end

    parsed_args = parse_args(ARGS, arg_parse_settings)
    ## I'm sure there's a way to do this iteratively 
    analysis_params = analysis_parameters(parsed_args["k"], parsed_args["max_rows"],
                                        parsed_args["use_random"], parsed_args["t_start"],
                                        parsed_args["t_end"], parsed_args["output_folder"], "filepath")
    return analysis_params
end


