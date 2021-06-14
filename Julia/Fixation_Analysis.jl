using JLD2, Plots, DataFrames, StatsPlots, Statistics


## Usage Note: Top-level functions are setup to be called as
## func(file, genotype). If no genotype is supplied, the functions will calculate the
## generic probabilities of (time to) fixation for any genotype. The provided genotype
## must be one of the initial genotypes

## need to write generalized import at some point
output = jldopen("NetworkGamePopGenTests.jld2")


###############
## Functions ##
###############

    ## Theoretical Prediction Funcs ##

# Conditional fixation of an advantageous allele, i.e. drift and selection, no mutation.
# Based on equation 12, Kimura and Ohta, 1969, written by Daniel Priego Espinosa
function t_bar(p::Float64,N::Float64,s::Float64)
    if s==0.0
        return t_bar(p,N)
    else
        u(p) = (1 - exp(-2*s*N*p)) / (1 - exp(-2*s*N))     # \psi(x) = \frac{2 \int_{0}^{1} G(x) dx}{V(x)G(x)}$
    # with
    # $G(x) = e^{-\int \frac{2 M(x)}{V(x)} dx}$
    # $V(x) = \frac{x(1-x)}{N_e}$        psi(x) = (exp(2*N*s*x)*(1 - exp(-2*N*s))) / (s*(1 - x)*x)
        t_int1(ξ) = psi(ξ)*u(ξ)*(1-u(ξ))
        t_int2(ξ) = psi(ξ)*u(ξ)^2
        tbar1_1, err_tbar1_1 = quadgk(t_int1,p,1)
        tbar1_2, err_tbar1_2 = quadgk(t_int2,0,p)
        return tbar1_1 + (1 - u(p))/u(p) * tbar1_2
    end
end


## From Ewens (2004) 3.9
## approximate time to fixation for a neutral allele
## adjusted from -4N -> -2N because model is haploid
function t_bar_star(p::Float64, N::Float64)
    if p <= 0 
        return 0
    elseif p == 1
        return 0
    else
        return (-4 * N / p)*(1-p)*(log(1-p))
    end
end

function π_x!(N, selective_advantage, ps)
    α = 2 * N * (selective_advantage)
    predicted_values = Vector{Float64}(undef,0)
    for p in ps
        if p == 0
            p_pred = p
        elseif p == 1
            p_pred = p
        else
            p_pred = (1 - exp(-α * p))/(1 - exp(-α))
        end
        push!(predicted_values, ((1 - exp(-α * p))/(1 - exp(-α))))
    end
    return predicted_values
end

    ## File Manipulation Functions ##

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
function plot_t_fix(file, genotype = nothing)
    ps, genotypes, t_fix = get_t_fix(file, genotype)
    df = DataFrame(init_freqs = ps, genotypes = genotypes, t_fix = t_fix)
    return df
end

function plot_mean_t_fix(file, genotype = nothing)
    # ps, genotypes, t_fix = get_t_fix(file, genotype)
    df = plot_t_fix(file, genotype)
    gdf = groupby(df, [:init_freqs])
    gdf = combine(gdf, :t_fix => mean)
    
    return gdf
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

function create_plots(file, pred_resolution::Float64, genotype = nothing)
    π_fix_df = plot_π_fix(file, genotype)
    t_fix_df = plot_t_fix(file, genotype)
    t_fix_mean = plot_mean_t_fix(file, genotype)
    t_bar_pred = Vector{Float64}(undef, 0)
    l = @layout [a;b]

    π_plot = scatter(π_fix_df.init_freqs, π_fix_df.π_x, labels = "π(p) (obs)")
    π_plot = plot!(collect(0.0:pred_resolution:1.0), collect(0.0:pred_resolution:1.0), labels = "π(p) (pred)")
    t_plot = scatter(t_fix_df.init_freqs, t_fix_df.t_fix, labels = "t_fix (obs)")
    t_plot = plot!(t_fix_mean.init_freqs, t_fix_mean.t_fix_mean, labels = "t_fix (mean)")
    for p in collect(0.0:pred_resolution:1.0)
        push!(t_bar_pred, t_bar(p, 100.0, 0.0))
    end
    t_plot = plot!(collect(0.0:pred_resolution:1.0), t_bar_pred, labels = "t_bar (pred)")
    plt = plot(π_plot, t_plot, layout=(2,1))

end
# create_plots(output, 0.001, 1)
savefig(create_plots(output, 0.01, 1), "p_and_t_fix.png")


