## functions necessary to create and analyze results from network game sims

using DataFrames, JLD2, StatsPlots, Statistics, Plots, ColorSchemes

include("NetworkGameStructs.jl")

struct analysis_parameters
    k::Int64
    max_rows::Int64
    use_random::Bool
    t_start::Int64
    t_end::Int64
end


###############
## File Manip Functions
###############

function load_files()
    files = Vector{JLD2.JLDFile}(undef, 0)
    for file in readdir()
        if last(splitext(file)) == ".jld2"
            print(file)
            push!(files, jldopen(file))
        end
        
    end
    
    return files
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
    
    ## filling networks and parameters
    fixations = fill(zeros(Int64,0), n_rows)
    n_genotypes = fill(zeros(Int64,0), n_rows)
    w_mean_history =  fill(zeros(Float64,0), n_rows)
    init_mean_history =  fill(zeros(Float64,0), n_rows)
    mean_net_history = fill(Vector{network}(undef, 0), n_rows)
    parameters = Vector{simulation_parameters}(undef, n_rows)
    timestep = fill(zeros(Int64,0), n_rows)
    rep_id = zeros(Int64, n_rows)
    df_dict = Dict([
                (Symbol(:fixations), fixations),
                (Symbol(:n_genotypes), n_genotypes),
                (Symbol(:w_mean_history), w_mean_history),
                (Symbol(:init_mean_history), init_mean_history),
                (Symbol(:mean_net_history), mean_net_history),
                (Symbol(:parameters), parameters),
                (Symbol(:timestep), timestep),
                (Symbol(:rep_id), rep_id)])
    
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
function split_parameters(df_dict::Dict, analysis_params)
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
function create_df(files::Vector{JLD2.JLDFile}, analysis_params)

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
    print("t_max ",length(tmax_array))
    ## less elegant for loop than the regular create_df :(
    ## seperating the edge matrix, populating the other columns
    row = 0
    if analysis_params.use_random == true
        network_indices = shuffle(1:length(df_dict[!, :nnet]))
    else
        network_indices = 1:length(df_dict[!, :nnet])
    end
    print("network_indices ", length(network_indices))
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



## for a DataFrame group, returns figures similar to those created in cooperateion_analysis.jl
## and JVC's original grant proposal. Implementation creates two dataframes because I copied it from
## cooperation analysis rather than directly applying it to the gdf created elsewhere.
function create_mean_w_violin_plots(group::SubDataFrame, k)
    nnets = zeros(Int64, 0)
    w_means = zeros(Float64, 0)
    for replicate in eachrow(group)
        push!(nnets, replicate[:nnet])
        push!(w_means, last(rolling_mean3(replicate[:w_mean_history], k)))
    end
    temp_df = DataFrame(nnet=nnets, w_mean = w_means)
    @df temp_df violin(:nnet, :w_mean, title = "Mean Fitness", legend = :none)
    @df temp_df boxplot!(:nnet, :w_mean, fillalpha=.6)
end


##similar to above, creates plots for a given subdataframe.
function create_mean_init_violin_plots(group::SubDataFrame, k)
    nnets = zeros(Int64, 0)
    inits = zeros(Float64,0)
    for replicate in eachrow(group)
        push!(nnets, replicate[:nnet])
        push!(inits, last(rolling_mean3(replicate[:init_mean_history], k)))
    end
    temp_df = DataFrame(nnet=nnets, inits = inits)
    @df temp_df violin(:nnet, :inits, title = "Mean Initial Offer", legend = :none)
    @df temp_df boxplot!(:nnet, :inits, fillalpha=.6)
end

## creates violin plots from a grouped dataframe
function create_all_violin_plots(gdf, k)
    for group in gdf
        plt = plot(create_mean_w_violin_plots(group, k), create_mean_init_violin_plots(group, k), layout=(2,1))
        b = replace(string(group[!, :b][1]), "."=>"0")
        c = replace(string(group[!, :c][1]), "."=>"0")
        tmax = replace(string(group[!, :tmax][1]), "."=>"0")
        filename = string("mean_init_and_fitness", "_b_", b, "_c_", c, "_tmax_", tmax, "_k_", string(k))
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


## from Satvik Beri (https://stackoverflow.com/questions/59562325/moving-average-in-julia)
## by default, only computes the mean after k datapoints, modified here to fill the array with intermediates

function rolling_mean3(arr, n)
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
function create_mean_init_and_fitness_plots(group::DataFrame, k::Int64)
    test_var = 1
    for b in unique(group[!,:b])
        print(string("b: ", string(b)))
        print("\n")
        for c in unique(group[!,:c])
            print(string("c: ", string(c)))
            print("\n")
            plt_w = plot()
            plt_init = plot()
            plt_out = plot(plt_w, plt_init, layout = (2,1))
            for nnet in unique(group[!,:nnet])
                i = 0
                tmax = maximum(group[!, :tmax])
                fitness_array = zeros(Float64, tmax)
                init_array = zeros(Float64, tmax)
                ## finding the element wise mean for the conditions
                for replicate in eachrow(subset(group, :b => ByRow(==(b)), :c => ByRow(==(c)), :nnet => ByRow(==(nnet))))
                    ## summing the rolling mean of each replicate
                    fitness_array .+= rolling_mean3(replicate.w_mean_history, k)
                    init_array .+= rolling_mean3(replicate.init_mean_history, k)
                    i+=1
                end
                ## dividing sum of replicates by # of reps
                fitness_array ./= i
                init_array ./= i
                plt_init = plot!(plt_out[1], init_array, label = nnet, title = "InitOffer")
                plt_w = plot!(plt_out[2], fitness_array, label = nnet, title = "W")
            end
            filestr = string("mean_w_b_", replace(string(b), "."=>"0"), "_c_", replace(string(c),"."=>"0"), "_k_", string(k))
            savefig(plt_out, filestr)
        end
    end
end


