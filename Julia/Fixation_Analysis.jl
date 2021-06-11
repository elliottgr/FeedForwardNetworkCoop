using JLD2, Plots


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
# get_replicate(file::JLDFile, experiment::Int64, replicate::Int64)

## find time until fixation and the genotype that fixed
function find_t_fix(rep)
    for t in 1:rep.parameters.tmax
        if rep.fixations[t] != 0
            return [t, rep.fixations[t]]
        end
    end

    ## -1 used as for debug. Could switch to nan later depending on plotting
    return [-1, -1]
end



for 

# for output_file in 1:length(output["sim_outputs"])
#     p = get_file(output_file)[1].parameters.init_freqs
#     t_fixations = Vector{Int64}(undef, get_file(output_file)[1].parameters.nreps)
#     g_fixations = Vector{Int64}(undef, get_file(output_file)[1].parameters.nreps)
#     for replicate in 1:get_file(output_file)[1].parameters.nreps
#         fixation = find_t_fix(get_file(output_file)[replicate])
#         t_fixations[replicate] = find_t_fix(get_file(output_file)[replicate])[1]
#         g_fixations[replicate] = find_t_fix(get_file(output_file)[replicate])[2]
        
#     end

# end
