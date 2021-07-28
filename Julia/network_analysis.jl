## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

# using DataFrames, JLD2
using LinearAlgebra
# include("NetworkGameStructs.jl")
include("NetworkGameAnalysisFuncs.jl")



## using the existing get_df_dict function, iterating over that array, creating a dataframe from that
## one entry corresponds to a single network weight
## edges represented by two columns. e1 is starting column, e2 is end. e2 = 0 means e1 is a node weight
function create_edge_df(files::Vector{JLD2.JLDFile}, net_save_tick_multiplier, max_rows)
    
    ## df_dict returns a dictionary that is not properly setup
    ## for creating a column of edge weights
    df_dict = get_df_dict(files)
    print("df_dict created!")
    rep_id = zeros(Int64, max_rows)
    timestep = zeros(Int64, max_rows)
    b_col = zeros(Float64, max_rows)
    c_col = zeros(Float64, max_rows)
    nnet = zeros(Int64, max_rows)
    edge_weight = zeros(Float64, max_rows)
    e1 = zeros(Int64, max_rows)
    e2 = zeros(Int64, max_rows)
    fitness = zeros(Float64, max_rows)

    ## less elegant for loop than the regular create_df :(
    ## seperating the edge matrix, populating the other columns
    row = 0
    # for i in 1:length(df_dict[:mean_net_history])
    for i in 1:max_rows

        ## debug checker
        if mod(i, 50) == 0
            print(i)
        end
        for n1 in 1:df_dict[:nnet][i]
            for t in 1:length(df_dict[:mean_net_history][i])
                if mod((df_dict[:net_save_tick][i][1]*net_save_tick_multiplier), t) == 0
                    if row < max_rows
                        row += 1
                        e1[row] = n1
                        e2[row] = 0
                        edge_weight[row] = df_dict[:mean_net_history][i][t].Wb[n1]
                        rep_id[row] = df_dict[:rep_id][i]
                        timestep[row] = df_dict[:timestep][i][t]
                        b_col[row] = df_dict[:b][i]
                        c_col[row] = df_dict[:c][i]
                        nnet[row] = df_dict[:nnet][i]
                        fitness[row] = df_dict[:w_mean_history][i][t]

                    end
                end
                for n2 in 1:df_dict[:nnet][i]
                    for t in 1:length(df_dict[:mean_net_history][i])
                        if mod(df_dict[:net_save_tick][i][1]*net_save_tick_multiplier, t) == 0
                            if row < max_rows
                                row+=1
                                e1[row] = n1
                                e2[row] = 0
                                edge_weight[row] = df_dict[:mean_net_history][i][t].Wb[n1]
                                rep_id[row] = df_dict[:rep_id][i]
                                timestep[row] = df_dict[:timestep][i][t]
                                b_col[row] = df_dict[:b][i]
                                c_col[row] = df_dict[:c][i]
                                nnet[row] = df_dict[:nnet][i]
                                fitness[row] = df_dict[:w_mean_history][i][t]
                            end
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
                    e1 = e1,
                    e2 = e2)
end

## function accepts a group of e1, returns correlations to all e2
function edge_correlations(group::SubDataFrame)

end

## takes a parameter set and creates a nested heatmap of edge-fitness correlations
function correlation_heatmaps(b_c_group::SubDataFrame)
    edge_df = groupby(b_c_group, [:e1, :e2])
    for i in 1:nnet
        for j in 1:nnet
            output_matrix[i,j] = correlation()
        end
    end
end

function main(k=50 , net_save_tick_multiplier = 1, max_rows = 1000000)
    include("NetworkGameAnalysisFuncs.jl")
    files = Vector{JLD2.JLDFile}(undef, 0)
    for file in readdir()
        if last(splitext(file)) == ".jld2"
            print(file)
            push!(files, jldopen(file))
        end
    end

    print("creating edge_df...")
    edge_df = create_edge_df(files, net_save_tick_multiplier, max_rows)
    print("edge_df done")
    
    # creating and passing grouped dataframes of parameter sets
    # b_c_grouped_edge_df = groupby(edge_df, [:b, :c])

    # for b_c_group in b_c_grouped_edge_df
    #     correlation_heatmaps(b_c_group)
    # end
    
    # selection = groupby(create_df(files), [:b, :c])


    #############################   
    ## Working plots, 
    ## disabled for now
    #############################

        #############################
        ## this one does not allow for different groups 
        ## because it produces a multi-line plot that 
        ## will not slice nicely with arbitrary groupings
        #############################
    # create_mean_init_and_fitness_plots(create_df(files), k)

    # create_all_violin_plots(selection, k)
    # plt = Vector(undef, 0)



    # for group in selection

    #     if group[1, :nnet] == 7
    #         push!(plt, network_heatmap(group))
    #     end
    # end
    # plt2 = plot(plt...)
    # savefig(plt2, "heatmap_test.png")

    ## will probably crash if memory isn't cleared between runs
    for file in files
        close(file)
    end
end

@time main(1, 1, 1000)  