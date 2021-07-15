## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

# using DataFrames, JLD2
using LinearAlgebra, Query
# include("NetworkGameStructs.jl")
include("NetworkGameAnalysisFuncs.jl")


## setting this up as the mean of rolling averages
function create_mean_init_and_fitness_plots(group::DataFrame, k::Int64)
    test_var = 1
    for b in unique(group[!,:b])
        for c in unique(group[!,:c])
            plt_w = plot()
            plt_init = plot()
            plt_out = plot(plt_w, plt_init, layout = (2,1))
            for nnet in unique(group[!,:nnet])
                i = 0
                tmax = maximum(group[!, :tmax])
                fitness_array = zeros(Float64, tmax)
                init_array = zeros(Float64, tmax)
                ## finding the element wise mean for the conditions
                # for replicate in group[(group.b .== b).&(group[!,:c] .== c).&(group[!,:nnet] .== nnet)]
                # for replicate in eachrow(group[(isequal.(group.b, b), :).&(isequal.(group.c, c),:).&(isequal.(group.nnet, nnet),:)])
                for replicate in eachrow(subset(group, :b => ByRow(==(b)), :c => ByRow(==(c)), :nnet => ByRow(==(nnet))))

                    for timepoint in 1:tmax
                        t_min = maximum([(timepoint-k), 1])
                        fitness_array[timepoint] += sum(replicate.w_mean_history[t_min:timepoint])/k
                        init_array[timepoint] += sum(replicate.init_mean_history[t_min:timepoint])/k
                    end
                    i+=1
                end
                ## dividing sum of replicates by # of reps
                fitness_array ./= i
                init_array ./= i
                plt_init = plot!(plt_out[1], init_array, label = nnet)
                plt_w = plot!(plt_out[2], fitness_array, label = nnet)
                # plt_out = plot!(init_array, fitness_array, layout=(2,1), label = nnet)

            end
            # plt_out = plot(plt_init, plt_w, layout=(1,2))
            filestr = string("mean_w_b_", replace(string(b), "."=>"0"), "_c_", replace(string(c),"."=>"0"))
            savefig(plt_out, filestr)
        end
    end
    # for replicate in group
        
    #     fitness_array .+= replicate.init_mean_history
    #     i+=1
    # end
    # fitness_array ./= i

    # return plot(fitness_array)
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
    create_mean_init_and_fitness_plots(create_df(files), 5000)
    # gdf = groupby(, [:b, :c])
    # for g in gdf
    #     create_mean_fitness_plots(g)
    # end
    # create_all_violin_plots(gdf)
    selection = groupby(create_df(files), [:b, :c, :nnet])
    i = 0
    plt = Vector(undef, 0)
    for group in selection
        # i+=1
        if group[1, :nnet] == 7
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