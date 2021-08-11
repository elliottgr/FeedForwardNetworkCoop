## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

using LinearAlgebra, Random
include("NetworkGameAnalysisFuncs.jl")




## takes a parameter set and creates a nested heatmap of edge-fitness correlations
## main diagonal is node weights, above main diag is node-node (edge) correlations
function correlation_heatmaps(b_c_nnet_group::DataFrame)
    shifted_nnet = b_c_nnet_group[!, :nnet][1] + 1
    weight_fitness_corr_matrix = zeros(Float64, (b_c_nnet_group[!, :nnet][1],b_c_nnet_group[!, :nnet][1]))
    for edge_group in groupby(b_c_nnet_group, [:e1, :e2])
        shifted_e1 = edge_group[!, :e1][1] + 1
        shifted_e2 = edge_group[!, :e2][1] + 1

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
function create_b_c_heatmap_plot(df, nnet::Int64, max_rows, use_random)
    b_c_vals = unique(df[!,:c])
    heatmaps = Vector{Plots.Plot}(undef,0)
    main_df = df[df.nnet .== nnet, :]

    for group in groupby(main_df, [:b,:c])
        push!(heatmaps,correlation_heatmaps(create_edge_df(group, max_rows, use_random)))
    end
    filestr = string("fitness_edge_weight_heatmap_nnet_", nnet, ".png")
    savefig(plot(heatmaps..., layout = (length(b_c_vals), length(b_c_vals))), filestr)
end


function main(k=50 , max_rows = 1000000, use_random = false)
    include("NetworkGameAnalysisFuncs.jl")
    files = load_files()
    print("\n")
    print( "creating main DataFrame...", "\n")
    print("\n")
    main_df = create_df(files)
    print("Done!")
    ## will probably crash if memory isn't cleared between runs
    for file in files
        close(file)
    end
    create_b_c_heatmap_plot(main_df, 5, max_rows, use_random)
    # for nnet in 1:2:maximum(main_df[!,:nnet])
    #     create_b_c_heatmap_plot(main_df, nnet, max_rows, use_random)
    # end
        # test_group = groupby(main_df, [:b,:c,:nnet])
    i = 0

   
 # catch MethodError
        # print("not enough datapoints for b: ", group[!, :b][1], " c: ", group[!, :c][1], " :nnet ", group[!, :nnet][1])

    
    # end
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



end


## crashes on PC with 8gb of RAM if max_rows >= around 40 million
## paramters are
## k = number of datapoints for running mean in time series data
## max_rows = number of edges to sample for edge analysis
## use_random = boolean for whether edges are sampled randomly or sequentially 
@time main(1, 1000000, true)  