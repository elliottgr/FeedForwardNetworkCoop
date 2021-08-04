## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

using LinearAlgebra, Random
include("NetworkGameAnalysisFuncs.jl")




## function accepts a group of e1, returns correlations to all e2
function edge_correlations(group::SubDataFrame)

end

## takes a parameter set and creates a nested heatmap of edge-fitness correlations
function correlation_heatmaps(edge_df::DataFrame)

end



function main(k=50 , max_rows = 1000000, use_random = false)
    include("NetworkGameAnalysisFuncs.jl")
    files = load_files()

    print("creating edge_df...")
    edge_df = create_edge_df(files, max_rows, use_random)

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


    ## will probably crash if memory isn't cleared between runs
    for file in files
        close(file)
    end
end

@time main(1, 100000, true)  