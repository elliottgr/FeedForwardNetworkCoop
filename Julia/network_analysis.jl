## takes outputs from networkgamecoop.jl simulations and creates visualizations 
include("NetworkGameAnalysisFuncs.jl")


## creates a plot of b/c ratio versus the evolved edge weight of node 1 to node 2
function w12_heatmap(edge_df, analysis_params)
    subset = edge_df[(edge_df.e1 .== 1) .& (edge_df.e2 .== 2) .& (edge_df.timestep .== maximum(edge_df.timestep)), :]
    filestr = pwd()*"/"*analysis_params.filepath*"/scatter_plots/"*string("bc_ratio_scatterplot_t_start_", analysis_params.t_start, "_tend_", analysis_params.t_end, ".png")
    savefig(scatter(((subset.b)./(subset.c)), subset.edge_weight), filestr)
end

function time_edgeweight_scatterplot(edge_df, analysis_params)
    subset = edge_df[(edge_df.e1 .== 1) .& (edge_df.e2 .== 2) , :]
    filestr = pwd()*"/"*analysis_params.filepath*"/scatter_plots/"*string("time_edgeweight_scatterplot_t_start_", analysis_params.t_start, "_tend_", analysis_params.t_end, ".png")
    savefig(scatter(subset.timestep, subset.edge_weight), filestr)
end   
function main()
    include("NetworkGameAnalysisFuncs.jl")
    files, folder_name = load_files()
    print("\n")
    print( "creating main DataFrame...", "\n")
    print("\n")

    ## see NetworkGameAnalysisfunc.jl for explanation of create_df() and create_edge_df()
    # analysis_params = analysis_parameters(k, max_rows, use_random, t_start, t_end, "figure_outputs")
    analysis_params = analysis_arg_parsing()
    analysis_params.t_start = 1
    analysis_params.t_end = 50000
    main_df = create_df(files, analysis_params)

    print("Done!" , "\n")

    ## there's no built-in exception to stop you analyzing multiple files at once

    ## creating output directories
    create_directories(analysis_params, folder_name)
    create_log_file(main_df, analysis_params)


    for file in files
        close(file)
    end

    # create_b_c_cooperation_heatmap(main_df, analysis_params)

    ##creates heatmaps of b/c vals and network weight / fitness correlations
    # for nnet in 1:2:maximum(main_df[!,:nnet])
    #     create_b_c_heatmap_plot(main_df, nnet, analysis_params)
    # end
        

    #############################   
    ## Working plots, 
    ## disabled for now
    #############################

        #############################
        ## this one does not allow for different groups 
        ## because it produces a multi-line plot that 
        ## will not slice nicely with arbitrary groupings
        #############################
    # create_mean_init_payoff_and_fitness_plots(main_df, analysis_params)
    # create_all_violin_plots(groupby(main_df, [:b, :c]), analysis_params)
    test_edges = create_edge_df(groupby(main_df, [:nnet])[1], analysis_params)
    w12_heatmap(test_edges, analysis_params)
    time_edgeweight_scatterplot(test_edges, analysis_params)

end


## crashes on PC with 8gb of RAM if max_rows >= around 40 million
## paramters are
## k = number of datapoints for running mean in time series data
## max_rows = number of edges to sample for edge analysis
## use_random = boolean for whether edges are sampled randomly or sequentially 
## t_start = timestep to begin tracking data for analysis
@time main() 