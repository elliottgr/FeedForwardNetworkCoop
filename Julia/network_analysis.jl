using DataFrames, JLD2, StatsPlots, Statistics, Plots, ColorSchemes, LinearAlgebra, Random, ArgParse, Dates
using DataFrames, JLD2, Plots, StatsPlots, Statistics, ColorSchemes
include("NetworkGameStructs.jl")


function create_directory(file_path::String, sub_folder::String)
    if isdir(string(pwd(), "/", file_path, "/")) == false
        print("Output directory not found, creating a new one at ", string(pwd(), "/", analysis_params.output_folder))
        mkdir(string(pwd(), "/", file_path, "/"))
    end
    if isdir(string(pwd(), "/", file_path, "/", sub_folder, "/")) == false
        mkdir(string(pwd(), "/", file_path, "/", sub_folder, "/"))
    end

    subfolders = ["/b_c_coop_heatmaps/",
                  "/edge_weight_w_heatmaps/",
                  "/violin_plots/",
                  "/time_series/",
                  "/scatter_plots/"]

    for subfolder in subfolders
        if isdir(string(pwd()*"/"*file_path*"/"*sub_folder*subfolder)) == false
            mkdir(string(pwd()*"/"*file_path*"/"*sub_folder*subfolder))
        end
    end
end

function init_offer_payoff_scatter(df)
    b = df.b[1]
    c = df.c[1]
    plt = plot(title = "b=$b & c=$c")
    for group in groupby(df, :nnet)
        label_str = "nnet: "*string(group.nnet[1])
        plt = plot!(group.mean_initial_offer, group.mean_payoff, seriestype = :scatter, xlabel = "Mean Initial Offer", ylabel = "Mean Payoff", label = label_str)
    end
    return plt
end

function fitness_violin_plots(df)
    end_df = df[df[!, :generation] .== maximum(df[!, :generation]), :]
    b = df.b[1]
    c = df.c[1]
    return @df end_df violin(:nnet, :mean_payoff, title = "Mean Fitness at Final Timestep for b = $b and c = $c", ylabel = "Mean Fitness", xlabel = "Network Length", legend = :none)
end


## Creating a heatmap to show correlation of fitness and edge-weights
## gonna be messy but need to map each correlation of edge column and fitness to a matrix
function edge_weight_fitness_plot(df)
    for group in groupby(df, :nnet)
        for b_c_group in groupby(group, [:b, :c])
            correlation_matrix = zeros(Float64, (5, 5))

            correlation_matrix[1,1] = cor(b_c_group.mean_payoff, b_c_group.n1)
            correlation_matrix[1,2] = cor(b_c_group.mean_payoff, b_c_group.e1_2)
            correlation_matrix[1,3] = cor(b_c_group.mean_payoff, b_c_group.e1_3)
            correlation_matrix[1,4] = cor(b_c_group.mean_payoff, b_c_group.e1_4)
            correlation_matrix[1,5] = cor(b_c_group.mean_payoff, b_c_group.e1_5)

            correlation_matrix[2,2] = cor(b_c_group.mean_payoff, b_c_group.n2)
            correlation_matrix[2,3] = cor(b_c_group.mean_payoff, b_c_group.e2_3)
            correlation_matrix[2,4] = cor(b_c_group.mean_payoff, b_c_group.e2_4)
            correlation_matrix[2,5] = cor(b_c_group.mean_payoff, b_c_group.e2_5)

            correlation_matrix[3,3] = cor(b_c_group.mean_payoff, b_c_group.n3)
            correlation_matrix[3,4] = cor(b_c_group.mean_payoff, b_c_group.e3_4)
            correlation_matrix[3,5] = cor(b_c_group.mean_payoff, b_c_group.e3_5)

            correlation_matrix[4,4] = cor(b_c_group.mean_payoff, b_c_group.n4)
            correlation_matrix[4,5] = cor(b_c_group.mean_payoff, b_c_group.e4_5)

            correlation_matrix[5,5] = cor(b_c_group.mean_payoff, b_c_group.n5)
        
            filestr = "b_"*string(b_c_group.b[1])*"_c_"*string(b_c_group.c[1])*"_nnet_"*string(b_c_group.nnet[1])*".png"
            title = "Edge/Fitness Correlation for b = "*string(b_c_group.b[1])*" c = "*string(b_c_group.c[1])*" nnet = "*string(b_c_group.nnet[1])*".png"

            savefig(heatmap(correlation_matrix,
                            xlabel = "Node 1",
                            ylabel = "Node 2",
                            c = :RdBu_9,
                            yflip = true), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/edge_weight_w_heatmaps/"*filestr))
        end
    end    
end

function fitness_timeseries_plots(df)
    b = df.b[1]
    c = df.c[1]
    plt = plot(title = "b=$b & c=$c")
    colors = palette(:tab10)
    color_i = 1
    # df = sort(df, :generation)
    for group in groupby(df, :nnet)
        ts = []
        ws = []
        for replicate in groupby(group, :replicate_id)
            label_str = string("Network Size: ", group.nnet[1])
            # sort!(replicate, :generation)
            plt = plot!(replicate.generation, replicate.mean_payoff, legend = :none, alpha = .2)
        end
        for t in unique(group.generation)
            push!(ts, t)
            w_hat = mean(group.mean_payoff[group.generation .== t, :])
            push!(ws, w_hat)
        end

        plt = plot!(ts, ws, color = color_i, label = "")
        color_i += 1
    end
    return plt
end
    
function main()
    
    ## need to add arg_parse filepath
    input_file = "LongerTimeTestb_c_min_000b_c_max200_nreps_25_tmax_1500.jld2"

    ## need to make a generalized file import, but this code should create necessary subdirectories 
    global filepath = "output_figures"
    global sub_folder = replace(input_file, "b_c_min_000b_c_max200_nreps_10_tmax_250.jld2" => "")
    create_directory(filepath, sub_folder)
    df = jldopen(input_file)["output_df"]
    
    ## splitting DataFrame into groups of B and C in order to generate figures
    
    timeseries_layout = @layout (length(unique(df.b))+1, length(unique(df.c)))
    timeseries_plots = []
    edge_weight_fitness_plot(df)
    for group in groupby(df, [:b, :c])  
        filestr = "CoopGameTest_b_"*replace(string(group[!, :b][1]), "."=>"0")*"_c_"*replace(string(group[!, :c][1]), "."=>"0")
        savefig(fitness_violin_plots(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/violin_plots/"*filestr))
        savefig(init_offer_payoff_scatter(group), string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/scatter_plots/"*filestr))

        timeseries = fitness_timeseries_plots(group)
        savefig(timeseries, string(pwd()*"/"*filepath*"/"*sub_folder*"/"*"/time_series/"*filestr))
        push!(timeseries_plots, fitness_timeseries_plots(group))


    end

    ## Adding legend to timeseries plot

    # for x in length(unique(df.c)) -1 
    #     push!(timeseries_plots, plot([0], xlabel = "", ylabel = "", xticks = false, yticks = false, xaxis = false, yaxis = false, legend = :none, showaxis = false, grid = false))
    # end
    # push!(timeseries_plots, plot([0,1,2,3,4], [0,1,2,3,4], showaxis = false, xticks = false, yticks = false, xaxis = false, yaxis = false, grid = false, color = palette(:tab10), labels = ("Network Size: 1", "Network Size: 2", "Network Size: 3", "Network Size: 4", "Network Size: 5")))
    # savefig(plot(timeseries_plots..., title = "Mean Payoff over time", layout = timeseries_layout, size = (1000,1000)), string(pwd()*"/"*filepath*"/time_series/merged.png"))
end

main()