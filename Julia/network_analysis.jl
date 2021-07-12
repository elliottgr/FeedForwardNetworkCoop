## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

using DataFrames, JLD2

include("NetworkGameStructs.jl")
include("NetworkGameAnalysisFuncs.jl")



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
    for group in gdf
        plt = plot(create_mean_w_violin_plots(group), create_mean_init_violin_plots(group), layout=(2,1))
        b = replace(string(group[!, :b][1]), "."=>"0")
        c = replace(string(group[!, :c][1]), "."=>"0")
        tmax = replace(string(group[!, :tmax][1]), "."=>"0")
        filename = string("mean_init_and_fitness", "_b_", b, "_c_", c, "_tmax_", tmax)
        savefig(plt, filename,)
    end
end

main()