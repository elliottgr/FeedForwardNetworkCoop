## takes outputs from networkgamecoop.jl simulations and creates visualizations 
include("NetworkGameAnalysisFuncs.jl")

function create_b_c_cooperation_heatmap(main_df, analysis_params)
    b_vals = sort(unique(main_df[!, :b]))
    c_vals = sort(unique(main_df[!, :c]))
    i = 0
    b_i_map = Dict{Float64, Int64}()
    c_i_map = Dict{Float64, Int64}()
    for b in b_vals
        i += 1
        b_i_map[b] = i
    end
    i = 0
    for c in c_vals 
        i += 1
        c_i_map[c] = i
    end
    for nnet_group in groupby(main_df, [:nnet])

        ## data here is the mean cooperation level at each timepoint for a replicate
        ## rolling mean of last k datapoints
        ## median of final mean cooperation level among replicates
        mean_output_matrix = zeros(Float64, (length(b_vals), length(c_vals)))
        median_output_matrix = zeros(Float64, (length(b_vals), length(c_vals)))
        # for group in b_c_groups

        for b_c_group in groupby(nnet_group, [:b, :c])
            replicates = 0
            b_i = b_i_map[b_c_group[!, :b][1]]
            c_i = c_i_map[b_c_group[!, :c][1]]
            # print(b_i, c_i)
            for replicate in eachrow(b_c_group)
                replicates += 1
                mean_output_matrix[b_i, c_i] += last(rolling_mean(replicate[:coop_mean_history][analysis_params.t_start:analysis_params.t_end], analysis_params.k))
                
            end
            mean_output_matrix[b_i, c_i] /= replicates
            median_output_matrix[b_i, c_i] = median([x[analysis_params.t_end] for x in b_c_group[!, :coop_mean_history]])
        end
         print(mean_output_matrix)
        filestr_mean = pwd()*"/"*analysis_params.filepath*"/b_c_coop_heatmaps/"*string("b_c_mean_coop_heatmap_nnet_", nnet_group[!, :nnet][1], "_tstart_", analysis_params.t_start, "_tend_", analysis_params.t_end, ".png")
        filestr_median = pwd()*"/"*analysis_params.filepath*"/b_c_coop_heatmaps/"*string("b_c_median_coop_heatmap_nnet_", nnet_group[!, :nnet][1], "_tstart_", analysis_params.t_start, "_tend_", analysis_params.t_end, ".png")

        savefig(heatmap(mean_output_matrix,
            xlabel = "c",
            ylabel = "b",
            xticks = c_vals,
            yticks = b_vals,
            title = "Mean Cooperation (k = $(analysis_params.k) )",
            clims = (.45, .55)), filestr_mean)
        savefig(heatmap(median_output_matrix,
            xlabel = "c",
            ylabel = "b",
            xticks = c_vals,
            yticks = b_vals,
            title = "Median Cooperation)",
            clims = (.45, .55)), filestr_median)
    end

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
    analysis_params.t_end = 1000
    main_df = create_df(files, analysis_params)

    print("Done!" , "\n")

    ## there's no built-in exception to stop you analyzing multiple files at once


    ## will probably crash if memory isn't cleared between runs


    ## creating output directories
    create_directories(analysis_params, folder_name)
    create_log_file(main_df, analysis_params)


    for file in files
        close(file)
    end

    create_b_c_cooperation_heatmap(main_df, analysis_params)

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



end


## crashes on PC with 8gb of RAM if max_rows >= around 40 million
## paramters are
## k = number of datapoints for running mean in time series data
## max_rows = number of edges to sample for edge analysis
## use_random = boolean for whether edges are sampled randomly or sequentially 
## t_start = timestep to begin tracking data for analysis
@time main() 