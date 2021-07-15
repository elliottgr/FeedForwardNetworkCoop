## this file takes outputs from networkgame*.jl simulations and
## creates visualizations of mean network weights over time

# using DataFrames, JLD2
using LinearAlgebra, Query
# include("NetworkGameStructs.jl")
include("NetworkGameAnalysisFuncs.jl")

## from Satvik Beri (https://stackoverflow.com/questions/59562325/moving-average-in-julia)
## by default, only computes the mean after k datapoints, modified here to fill the array with intermediates

function rolling_mean3(arr, n)
    so_far = sum(arr[1:n])
    pre_out = zero(arr[1:(n-1)])
    for i in 1:(n-1)
        pre_out[i] = mean(arr[1:i])
    end
    out = zero(arr[n:end])

    ## modification from Beri's script: this doesn't get overwrriten in the below loop
    out[1] = so_far / n
    for (i, (start, stop)) in enumerate(zip(arr, arr[n+1:end]))
        so_far += stop - start
        out[i+1] = so_far / n
    end

    return append!(pre_out, out)
end


## setting this up as the mean of rolling averages sampling "k" timepoints
function create_mean_init_and_fitness_plots(group::DataFrame, k::Int64)
    test_var = 1
    for b in unique(group[!,:b])
        print(string("b: ", string(b)))
        print("\n")
        for c in unique(group[!,:c])
            print(string("c: ", string(c)))
            print("\n")
            plt_w = plot()
            plt_init = plot()
            plt_out = plot(plt_w, plt_init, layout = (2,1))
            for nnet in unique(group[!,:nnet])
                i = 0
                tmax = maximum(group[!, :tmax])
                fitness_array = zeros(Float64, tmax)
                init_array = zeros(Float64, tmax)
                ## finding the element wise mean for the conditions
                for replicate in eachrow(subset(group, :b => ByRow(==(b)), :c => ByRow(==(c)), :nnet => ByRow(==(nnet))))
                    ## summing the rolling mean of each replicate
                    fitness_array .+= rolling_mean3(replicate.w_mean_history, k)
                    init_array .+= rolling_mean3(replicate.init_mean_history, k)
                    i+=1
                end
                ## dividing sum of replicates by # of reps
                fitness_array ./= i
                init_array ./= i
                plt_init = plot!(plt_out[1], init_array, label = nnet, title = "InitOffer")
                plt_w = plot!(plt_out[2], fitness_array, label = nnet, title = "W")
            end
            filestr = string("mean_w_b_", replace(string(b), "."=>"0"), "_c_", replace(string(c),"."=>"0"), "_k_", string(k))
            savefig(plt_out, filestr)
        end
    end
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
    create_mean_init_and_fitness_plots(create_df(files), 50)

    selection = groupby(create_df(files), [:b, :c, :nnet])
    plt = Vector(undef, 0)
    for group in selection

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

@time main()  