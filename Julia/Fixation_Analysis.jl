using JLD2, Plots, DataFrames, StatsPlots, Statistics


## Usage Note: Top-level functions are setup to be called as
## func(file, genotype). If no genotype is supplied, the functions will calculate the
## generic probabilities of (time to) fixation for any genotype. The provided genotype
## must be one of the initial genotypes

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


## finds the probability for fixation observed in an experiment
## not supplying a genotype to this function will calculate the probability
## of ANY genotype fixing. May be useful in later multi-genotype simulations
function find_π_x(experiment, genotype = nothing)
    experiment_results = Vector{Float64}(undef, 0)
    for replicate in experiment
        t_fix = find_t_fix(replicate, genotype)

        ## the if/elseif loop here works because
        ## find_t_fix either returns ints or NaN
        ## and NaN gets read as Float64
        ## Works on Julia 1.6 6/14/21
        if typeof(t_fix) == Vector{Float64}
            push!(experiment_results, 0)
        elseif typeof(t_fix) == Vector{Int64}
            push!(experiment_results, 1)
        end
    end
    return mean(skipmissing(experiment_results))
end

function get_t_fix(file::JLD2.JLDFile, genotype=nothing)

    ps = Vector{Float64}(undef, 0)
    genotypes = Vector{Int64}(undef, 0)
    t_fix = Vector{Int64}(undef, 0)
    for file in get_experiment(file)
        for replicate in file
            if isnan.(find_t_fix(replicate,genotype)) == [false, false]
                push!(ps, replicate.parameters.init_freqs[genotype])
                push!(genotypes, find_t_fix(replicate,genotype)[2])
                push!(t_fix, find_t_fix(replicate,genotype)[1])
        
            end
        end
    end
    return ps, genotypes, t_fix
end 


## returns a dataframe for plotting the time to fixation accross replicates as a function of initial frequency
function plot_t_fix(file, genotype=nothing)
    ps, genotypes, t_fix = get_t_fix(file, genotype)
    df = DataFrame(init_freqs = ps, genotypes = genotypes, t_fix = t_fix)
    return df
end


## returns a dataframe for plotting the observed mean of fixation accross replicates
function plot_π_fix(file, genotype = nothing)
    ps = Vector{Float64}(undef, length(get_experiment(file)))
    π_x = Vector{Float64}(undef, length(get_experiment(file)))
    for experiment in get_experiment(file)
        push!(ps, experiment[1].parameters.init_freqs[genotype])
        push!(π_x, find_π_x(experiment, genotype))
    end
    df = DataFrame(init_freqs = ps, π_x = π_x)
    return df
end

function create_plots(file, genotype = nothing)
    π_fix_df = plot_π_fix(file, genotype)
    t_fix_df = plot_t_fix(file, genotype)
    l = @layout [a;b]
    p1 = scatter(π_fix_df.init_freqs, π_fix_df.π_x)
    p2 = scatter(t_fix_df.init_freqs, t_fix_df.t_fix)
    plot(p1, p1, layout=l)
end

# df = plot_p_fix(output,2)
# plot_t_fix(output, 1)
# @df df scatter(
#     :init_freqs,
#     :π_x
#     )
