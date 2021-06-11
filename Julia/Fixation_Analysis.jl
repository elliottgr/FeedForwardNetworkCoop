using JLD2, Plots

output = jldopen("NetworkGamePopGenTests.jld2")


## Functions
## retrieves the invidividual experiments from the larger data structure
## each element in the output["sim_outputs"] vector is a full experiment + replicates
## within each experiment vector, elements correspond to simulation_output objects from each replicate
get_file(x) = output["sim_outputs"][x]
get_replicate(experiment::Int64, replicate::Int64)
## find time until fixation and the genotype that fixed
function find_t_fix(rep)
    for t in 1:rep.parameters.tmax
        if rep.fixations[t] != 0
            return [t, rep.fixations[t]]
        end
    end
    return [-1, -1]
end


for output_file in 1:length(output["sim_outputs"])
    p = get_file(output_file)[1].parameters.init_freqs
    t_fixations = Vector{Int64}(undef, get_file(output_file)[1].parameters.nreps)
    g_fixations = Vector{Int64}(undef, get_file(output_file)[1].parameters.nreps)
    for replicate in 1:get_file(output_file)[1].parameters.nreps
        fixation = find_t_fix(get_file(output_file)[replicate])
        t_fixations[replicate] = find_t_fix(get_file(output_file)[replicate])[1]
        g_fixations[replicate] = find_t_fix(get_file(output_file)[replicate])[2]
        
    end

end
