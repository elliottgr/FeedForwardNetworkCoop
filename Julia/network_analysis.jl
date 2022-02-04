
include("NetworkPlottingFunctions.jl")

    
function main()
    
    ## need to add arg_parse filepath
    input_file = "networksize2testb_c_min_000b_c_max200_nreps_20_tmax_5000.jld2"

    ## need to make a generalized file import, but this code should create necessary subdirectories 
    global filepath = "output_figures"
    # global sub_folder = replace(input_file, "b_c_min_000b_c_max200_nreps_10_tmax_250.jld2" => "")
    global sub_folder = string(split(input_file, "b_c_min")[1])
    create_directory(filepath, sub_folder)
    df = jldopen(input_file)["output_df"]
    print(names(df))
    create_log_file(df, filepath,sub_folder)
    ## splitting DataFrame into groups of B and C in order to generate figures
    
    timeseries_layout = @layout (length(unique(df.b))+1, length(unique(df.c)))
    timeseries_plots = []
    edge_weight_fitness_plot(df)
    for group in groupby(df, [:b, :c])  
        filestr = "CoopGameTest_b_"*replace(string(group[!, :b][1]), "."=>"0")*"_c_"*replace(string(group[!, :c][1]), "."=>"0")
        savefig(fitness_violin_plots(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/violin_plots/"*filestr))
        savefig(init_offer_payoff_scatter(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/scatter_plots/"*filestr))

        payoff_timeseries = fitness_timeseries_plots(group)
        init_timeseries = initoffer_timeseries_plots(group)
        combined_timeseries = plot([payoff_timeseries, init_timeseries]..., layout = (2,1), size = (800,600))
        savefig(combined_timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/"*filestr))
        savefig(payoff_timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/payoff/"*filestr))
        savefig(init_timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/initial_offer/"*filestr))
        # savefig()
        push!(timeseries_plots, fitness_timeseries_plots(group))


    end
    savefig(ess_plot(df), string(pwd()*"/"*filepath*"/"*sub_folder*"/ess_test.png"))
    ## Adding legend to timeseries plot

    # for x in length(unique(df.c)) -1 
    #     push!(timeseries_plots, plot([0], xlabel = "", ylabel = "", xticks = false, yticks = false, xaxis = false, yaxis = false, legend = :none, showaxis = false, grid = false))
    # end
    # push!(timeseries_plots, plot([0,1,2,3,4], [0,1,2,3,4], showaxis = false, xticks = false, yticks = false, xaxis = false, yaxis = false, grid = false, color = palette(:tab10), labels = ("Network Size: 1", "Network Size: 2", "Network Size: 3", "Network Size: 4", "Network Size: 5")))
    # savefig(plot(timeseries_plots..., title = "Mean Payoff over time", layout = timeseries_layout, size = (1000,1000)), string(pwd()*"/"*filepath*"/time_series/merged.png"))
end

main()