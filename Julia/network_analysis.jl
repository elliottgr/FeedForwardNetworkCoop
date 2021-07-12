## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

using DataFrames, JLD2

include("NetworkGameStructs.jl")
include("NetworkGameAnalysisFuncs.jl")


function main()
    files = Vector{JLD2.JLDFile}(undef, 0)
    for file in readdir()
        if last(splitext(file)) == ".jld2"
            print(file)
            push!(files, jldopen(file))
        end
    end
    # filestr = "NetworkGameTests_b_200_c_100_tmax_100000.jld2"

    # output = jldopen(file)
    create_df(files)
end

main()