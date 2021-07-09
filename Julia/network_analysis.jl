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
## 7/9/21 2nd try at creating a dataframe of values. 
## going to be 1 row = 1 replicate, try to extract from there for plotting
function create_df(files::Vector{JLD2.JLDFile})


    ## preallocating arrays for DF
    n_replicates = find_n_rows(files)
    fixations = zeros(Int64, n_replicates)
    n_genotypes = zeros(Int64, n_replicates)
    w_mean_history = zeros(Float64, n_replicates)
    init_mean_history = zeros(Float64, n_replicates)
    ## not sure how to define a zeros on these, leaving it undef for append
    mean_net_history = Vector{network}(undef, 0)
    parameters = Vector{simulation_parameters}(undef, 0)
    df_dict = Dict([(Symbol(n_replicates), n_replicates),
                (Symbol(fixations), fixations),
                (Symbol(n_genotypes), n_genotypes),
                (Symbol(w_mean_history), w_mean_history),
                (Symbol(init_mean_history), init_mean_history),
                (Symbol(mean_net_history), mean_net_history),
                (Symbol(parameters), parameters)])

    row_i = Int64(1)
    for file in files
        for experiment in get_experiment(file)
            for rep in experiment
                for col in get_column_names
                    df_dict[col][row_i] = getproperty(rep, col)
                end
            end
        end
    end
end

## creates a dataframe containing each timestep of all replicates
# function create_df(files::Vector{JLD2.JLDFile})
#     ## need starting variables for preallocation
#     ## max_nnet will break if network structures are changed!!!!!!!!!!!!!
#     nrows = Int64(0)
#     max_nnet = Int64(0)
#     param_column_names, output_column_names = get_column_names()

#     for file in files
#         data = get_experiment(file)
#         for experiment in data
#             nrows += (experiment[1].parameters.nreps*experiment[1].parameters.tmax)
#             max_nnet = maximum([max_nnet, experiment[1].parameters.nnet])
#         end 
#     end

#     # creating column names for all nodes and node-node (edge) permutations 
#     node_column_names = Vector{Int64}(undef, 0)
#     edge_column_names = Vector{Tuple}(undef, 0)
#     for i in 1:max_nnet
#         push!(node_column_names, i)
#         # for j in 1:max_nnet
#         #     push!(edge_column_names, tuple(i,j))
#         # end
#     end

#     timesteps = zeros(Int64, nrows)
#     t_n = Int64(1)
#     print(length(edge_column_names))
#     param_array = zeros((length(param_column_names), nrows))
#     output_array = zeros((length(output_column_names), nrows))
#     node_array = zeros((length(node_column_names), nrows))
#     edge_array = zeros((1, nrows))
#     print(size(output_array))
#     file_i = 1
#     for file in files
#         data = get_experiment(file)
#         for experiment_i in 1:length(data)
#             for replicate_i in 1:length(data[experiment_i])
#                 for t_n_replicate in 1:data[experiment_i][replicate_i].parameters.tmax
#                     timesteps[t_n] = t_n_replicate
#                     ##parameter loop
#                     for param_i in 1:length(param_column_names)
#                         val = getproperty(data[experiment_i][replicate_i].parameters, param_column_names[param_i])
#                         if param_column_names[param_i] == Symbol("init_freqs")
#                             val = val[1]
#                         end
#                         if typeof(val) == String
#                             val = file_i
#                         end
#                         param_array[param_i, t_n] = val
#                     end

#                     #sim_output loop
#                     for output_i in 1:length(output_column_names)
#                         # print(output_i)
#                         if output_column_names[output_i] != Symbol("parameters")

#                             val = getproperty(data[experiment_i][replicate_i], output_column_names[output_i])
#                             # if typeof(val) == String
#                             #     print(val, "\n")
#                             #     print(output_column_names[output_i], "\n")
#                             # end
#                             # if isdefined(val, t_n_replicate) == false
#                             #     print(output_column_names[output_i], "\n")
#                             #     if output_column_names[output_i] == Symbol("mean_network_history")
#                             #         ## creating dummy network
#                             #         val[t_n_replicate] = network(0, zeros(Float64, (1,1)), zeros(Float64, 1), 0.0, 0.0)
#                             #     else
#                             #         val[t_n_replicate] = data[experiment_i][replicate_i].parameters.tmax + 1
#                             #     end
#                             #         # print(val[t_n_replicate])
                                
#                             # end
#                             output_array[output_i, t_n] = val[t_n_replicate]
#                         end
#                     end
#                     t_n += Int64(1)
#                 end
#             end
#         end
#         file_i += 1
#     end
# end


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