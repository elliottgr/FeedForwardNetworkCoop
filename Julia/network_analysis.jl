## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

# using DataFrames, JLD2
using LinearAlgebra, Query
# include("NetworkGameStructs.jl")
include("NetworkGameAnalysisFuncs.jl")

function fitness_edge_cross_corr(replicate)

end


function find_fitness_edge_corrs(group::SubDataFrame)
    for replicate in eachrow(group)
        print(typeof(replicate))
    end
end

function main(k=50)
    include("NetworkGameAnalysisFuncs.jl")
    files = Vector{JLD2.JLDFile}(undef, 0)
    for file in readdir()
        if last(splitext(file)) == ".jld2"
            print(file)
            push!(files, jldopen(file))
        end
    end
    
    selection = groupby(create_df(files), [:b, :c])
    find_fitness_edge_corrs(selection)


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
    for file in files
        close(file)
    end
end

@time main(5000)  