## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

# using DataFrames, JLD2
using LinearAlgebra
# include("NetworkGameStructs.jl")
include("NetworkGameAnalysisFuncs.jl")


## returns a heightmap of all edge->edge connections at the end of each replicate in a given group 
## I'm not putting an EXPLICIT control for what happens when the group contains
## differing values of nnet, but passing something like that will likely break your analysis!!
function network_heatmap(group::SubDataFrame)
    # max_net = ma
    output_wm = zeros(Float64, (group[1, :nnet], group[1, :nnet]))
    output_wb = zeros(Float64, group[1, :nnet])
    reps = 0
    for replicate in eachrow(group)
        output_wm .+= last(replicate.mean_net_history).Wm
        output_wb .+= last(replicate.mean_net_history).Wb
        reps += 1
    end
    output_wm ./= reps
    output_wb ./= reps
    gr()
    title = string("b = ", string(group[1, :b]), ", c = ", string(group[1, :c]))
    return heatmap(output_wm, clim = (-0.7,0.7), title = title) 
end

function main()
    include("NetworkGameAnalysisFuncs.jl")
    files = Vector{JLD2.JLDFile}(undef, 0)
    for file in readdir()
        if last(splitext(file)) == ".jld2"
            print(file)
            push!(files, jldopen(file))
        end
    end
    gdf = groupby(create_df(files), [:b, :c])

    # create_all_violin_plots(gdf)
    selection = groupby(create_df(files), [:b, :c, :nnet])
    i = 0
    plt = Vector(undef, 0)
    for group in selection
        # i+=1
        if group[1, :nnet] == 3
            push!(plt, network_heatmap(group))
        end
    end
    plt2 = plot(plt...)
    savefig(plt2, "heatmap_test.png")
    for file in files
        close(file)
    end
end

main()