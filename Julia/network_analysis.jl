include("NetworkPlottingFunctions.jl")

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
        df = jldopen(input_file)["output_df"]
        print(names(df))
        create_log_file(df, filepath,sub_folder)

        
        ## These plots generate from the whole data frame

        edge_weight_fitness_plot(df)
        savefig(ess_plot(df), string(pwd()*"/"*filepath*"/"*sub_folder*"/ess_test.png"))


        ## These plots are generated based on individual game parameter sets
        for group in groupby(df, [:b, :c])  

            ## Creating the generic file string for saving plots
            filestr = "CoopGameTest_b_"*replace(string(group[!, :b][1]), "."=>"0")*"_c_"*replace(string(group[!, :c][1]), "."=>"0")
        
            ## Generating the timeseries plots
            payoff_timeseries = fitness_timeseries_plots(group)
            init_timeseries = initoffer_timeseries_plots(group)
            coop_timeseries = cooperation_timeseries_plots(group)
            combined_timeseries = plot([payoff_timeseries, init_timeseries, coop_timeseries]..., layout = (3,1), size = (800,900))
            
            ## Saving each of the generated plots
            savefig(edge_weight_timeseries(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/edge_weights/"*filestr))
            savefig(fitness_violin_plots(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/violin_plots/"*filestr))
            savefig(init_offer_payoff_scatter(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/scatter_plots/"*filestr))
            savefig(combined_timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/"*filestr))
            savefig(payoff_timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/payoff/"*filestr))
            savefig(init_timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/initial_offer/"*filestr))
        end
    end  ## End of the loop for each file

end

main()