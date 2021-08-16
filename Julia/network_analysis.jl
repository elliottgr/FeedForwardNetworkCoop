## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

# using LinearAlgebra, Random
include("NetworkGameAnalysisFuncs.jl")

function main(k=50 , max_rows = 1000000, use_random = false, t_start = 1, t_end = 1000)
    include("NetworkGameAnalysisFuncs.jl")
    files = load_files()
    print("\n")
    print( "creating main DataFrame...", "\n")
    print("\n")

    ## see NetworkGameAnalysisfunc.jl for explanation of create_df() and create_edge_df()
    analysis_params = analysis_parameters(k, max_rows, use_random, t_start, t_end, "figure_outputs")
    main_df = create_df(files, analysis_params)

    print("Done!")
    ## will probably crash if memory isn't cleared between runs
    for file in files
        close(file)
    end

    if isdir(string(pwd(), analysis_params.output_folder)) == false
        print("Output directory not found, creating a new one at ", string(pwd(), "/", analysis_params.output_folder))
        mkdir(string(pwd(), "/", analysis_params.output_folder))
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
@time main(50, 10000000, true, 1, 10000) 