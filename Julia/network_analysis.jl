using Chain, DataFramesMeta, AlgebraOfGraphics, CairoMakie
include("NetworkPlottingFunctions.jl")
include("activationFuncs.jl")

## Look into ForwardDiff.jl instead?
function finite_dif(f::Function, x, h)
    return (f(x + h*x) - f(x))/h
end

function main()
    
    ## Reading all files in the script's directory, searching for simulation outputs

    files = []
    for file in readdir()
        if last(splitext(file)) == ".jld2"
            push!(files, file)
        end
    end

    ## Creating directories and plots for each simulation output file
    for input_file in files

        ## need to make a generalized file import, but this code should create necessary subdirectories 
        global filepath = "output_figures"
        global sub_folder = replace(input_file, ".jld2" => "")
        create_directory(filepath, sub_folder)
        current_file = jldopen(input_file)
        # parameters = current_file["parameters"]
        df = current_file["output_df"]
        create_log_file(df, filepath,sub_folder)

        ## These plots are generated based on individual game parameter sets
        for group in groupby(df, [:b, :c])  
            # # ## Creating the generic file string for saving plots
            filestr = "CoopGameTest_b_"*replace(string(group[!, :b][1]), "."=>"0")*"_c_"*replace(string(group[!, :c][1]), "."=>"0")
            # savefig(ess_plot(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/ESS/"*filestr))
            # ## Generating the timeseries plots
            # payoff_timeseries = fitness_timeseries_plots(group)
            # init_timeseries = initoffer_timeseries_plots(group)
            # coop_timeseries = cooperation_timeseries_plots(group)
            # combined_timeseries = Plots.plot([payoff_timeseries, init_timeseries, coop_timeseries]..., layout = (3,1), size = (800,900))
            
            # ## Saving each of the generated plots
            # savefig(edge_weight_timeseries(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/edge_weights/"*filestr))
            # savefig(fitness_violin_plots(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/violin_plots/"*filestr))
            # savefig(init_offer_payoff_scatter(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/scatter_plots/"*filestr))
            # savefig(combined_timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/"*filestr))
            # savefig(payoff_timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/payoff/"*filestr))
            # savefig(init_timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/initial_offer/"*filestr))
            

            # Generating plots from JVCs test file
            for nnet_slice in groupby(group, :nnet)

                group_b = nnet_slice.b[1]
                group_c = nnet_slice.c[1]
                group_nnet = nnet_slice.nnet[1]
                mean_output_slice = @chain nnet_slice groupby(:generation) combine([:b, :c, :mean_payoff, :mean_cooperation, :n1, :n2, :e1_2, :mean_initial_offer] .=> mean, renamecols=false)
                coop_payoff = draw(
                    data(@chain mean_output_slice stack([:mean_payoff, :mean_cooperation])) * 
                    mapping(:generation, :value, color = :variable) *
                    visual(Lines); 
                    axis = (width = 400, height = 200, title = "b = $group_b, c = $group_c, net size = $group_nnet")
                )
                net_weights = draw(
                    data(@chain mean_output_slice stack([:n1, :n2, :e1_2, :mean_initial_offer])) * 
                    mapping(:generation, :value, color = :variable) *
                    visual(Lines); 
                    axis = (width = 400, height = 200, title = "b = $group_b, c = $group_c, net size = $group_nnet")
                )
                if group_nnet == 2  ## Plot is only setup for this network size currently
                    bmc = draw(
                        data(@chain mean_output_slice @transform(:bmc = @. :b * :e1_2 - :c )) * 
                        mapping(:generation, :bmc) *
                        visual(Lines);
                        axis = (width = 400, height = 200, xlabel = "Generation", ylabel = "(λ ⋅ b) - c", title = "b = $group_b, c = $group_c, net size = $group_nnet")
                    )
                    save(string(pwd()*"/"*filepath*"/"*sub_folder*"/jvc_plots/bmc/"*filestr*"_nnet_$group_nnet.png"), bmc, px_per_unit = 3)
                end
                save(string(pwd()*"/"*filepath*"/"*sub_folder*"/jvc_plots/coop_payoff/"*filestr*"_nnet_$group_nnet.png"), coop_payoff, px_per_unit = 3)
                save(string(pwd()*"/"*filepath*"/"*sub_folder*"/jvc_plots/net_weights/"*filestr*"_nnet_$group_nnet.png"), net_weights, px_per_unit = 3)
                end

        end
    end  ## End of the loop for each file

end

main()