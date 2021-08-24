## takes outputs from networkgamecoop.jl simulations and creates visualizations 
include("NetworkGameAnalysisFuncs.jl")

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

    analysis_params.filepath = string(analysis_params.output_folder, "/", folder_name )
    ## will probably crash if memory isn't cleared between runs


    ## creating output directories
    if isdir(string(pwd(), "/", analysis_params.output_folder)) == false
        print("Output directory not found, creating a new one at ", string(pwd(), "/", analysis_params.output_folder))
        mkdir(string(pwd(), "/", analysis_params.output_folder))
    end
    if isdir(string(pwd(), "/", analysis_params.filepath, "/")) == false
        mkdir(string(pwd(), "/", analysis_params.filepath))
    end
    create_log_file(main_df, analysis_params)


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
    create_mean_init_payoff_and_fitness_plots(main_df, analysis_params)
    create_all_violin_plots(groupby(main_df, [:b, :c]), analysis_params)



end


## crashes on PC with 8gb of RAM if max_rows >= around 40 million
## paramters are
## k = number of datapoints for running mean in time series data
## max_rows = number of edges to sample for edge analysis
## use_random = boolean for whether edges are sampled randomly or sequentially 
## t_start = timestep to begin tracking data for analysis
@time main() 