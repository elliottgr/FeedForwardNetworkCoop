## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

using LinearAlgebra, Random
include("NetworkGameAnalysisFuncs.jl")




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
    filestr = string("fitness_edge_weight_heatmap_nnet_", nnet, "_tstart_", analysis_params.t_start, "_tend_", analysis_params.t_end, ".png")
    savefig(plot(heatmaps..., layout = (length(b_c_vals), length(b_c_vals))), filestr)
end


function main(k=50 , max_rows = 1000000, use_random = false, t_start = 1, t_end = 1000)
    include("NetworkGameAnalysisFuncs.jl")
    files = load_files()
    print("\n")
    print( "creating main DataFrame...", "\n")
    print("\n")

    ## see NetworkGameAnalysisfunc.jl for explanation of create_df() and create_edge_df()
    analysis_params = analysis_parameters(k, max_rows, use_random, t_start, t_end)
    main_df = create_df(files, analysis_params)

    print("Done!")
    ## will probably crash if memory isn't cleared between runs
    for file in files
        close(file)
    end

    ## main loop
    for nnet in 1:2:maximum(main_df[!,:nnet])
        create_b_c_heatmap_plot(main_df, nnet, analysis_params)
    end
        

    #############################   
    ## Working plots, 
    ## disabled for now
    #############################

        #############################
        ## this one does not allow for different groups 
        ## because it produces a multi-line plot that 
        ## will not slice nicely with arbitrary groupings
        #############################
    # create_mean_init_and_fitness_plots(main_df, k)
    # create_all_violin_plots(groupby(main_df, [:b, :c]), k)



end


## crashes on PC with 8gb of RAM if max_rows >= around 40 million
## paramters are
## k = number of datapoints for running mean in time series data
## max_rows = number of edges to sample for edge analysis
## use_random = boolean for whether edges are sampled randomly or sequentially 
## t_start = timestep to begin tracking data for analysis
@time main(1, 1000000, true, 10000, 100000) 