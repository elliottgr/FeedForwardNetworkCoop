using JLD2, Plots, DataFrames, StatsPlots


## need to write generalized import at some point
output = jldopen("NetworkGamePopGenTests.jld2")


## Functions
## retrieves the invidividual experiments from the larger data structure
## each element in the output["sim_outputs"] vector is a full experiment + replicates
## within each experiment vector, elements correspond to simulation_output objects from each replicate
## simulation_output will contain parameters + saved data from each experiment


function get_experiment(file::JLD2.JLDFile, experiment=nothing, replicate=nothing) 
    if experiment == nothing
        return file["sim_outputs"]
    elseif typeof(experiment) == Int64
        if replicate == nothing
            return file["sim_outputs"][experiment]

        ## not sure the best way to do the error handling here
        elseif typeof(replicate) == Int64
            return get_experiment(file, experiment)[replicate]
        end
    end
end

## find time until fixation.
## Returns first genotype that fixes, 
## unless a specific genotypeID is provided
function find_t_fix(rep, genotype=nothing)
    for t in 1:rep.parameters.tmax
        if typeof(genotype) == Int64
            if rep.fixations[t] == genotype
                return [t, rep.fixations[t]]
            end
        else
            if rep.fixations[t] != 0
                return [t, rep.fixations[t]]
            end
        end
    end

    ## -1 used as for debug. Could switch to nan later depending on plotting
    return [NaN, NaN]
end

function get_t_fix(file::JLD2.JLDFile, genotype=nothing)

    ps = Vector{Float64}(undef, 0)
    genotypes = Vector{Int64}(undef, 0)
    t_fix = Vector{Int64}(undef, 0)
    for file in get_experiment(file)
        for replicate in file
            if isnan.(find_t_fix(replicate,1)) == [false, false]
                push!(ps, replicate.parameters.init_freqs[1])
                push!(genotypes, find_t_fix(replicate,genotype)[2])
                push!(t_fix, find_t_fix(replicate,genotype)[1])
        
            end
        end
    end
    return ps, genotypes, t_fix
end 

function plot_t_fix(file, genotype=nothing)
    ps, genotypes, t_fix = get_t_fix(file, genotype)
    df = DataFrame(init_freqs = ps, genotypes = genotypes, t_fix = t_fix)
    # df[df.genotypes .== 1, :]
    gr()
    plot(ps, seriestype=:scatter, t_fix)
    # print(nrow(df))
    # @df df scatter(
    #                 :init_freqs,
    #                 :t_fix
    #                 )
end
plot_t_fix(output, 1)