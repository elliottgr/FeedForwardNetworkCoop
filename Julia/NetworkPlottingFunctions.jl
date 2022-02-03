## This file contains all the functions and imports necessary to call plotting commands
## for other parts of the GRN cooperation simulation

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
        ## manually adding initial_offer and payoff folders for time_series
        if subfolder == "/time_series/"
            if isdir(string(pwd()*"/"*file_path*"/"*sub_folder*subfolder*"payoff/")) == false
                mkdir(string(pwd()*"/"*file_path*"/"*sub_folder*subfolder*"payoff/"))
            end
            if isdir(string(pwd()*"/"*file_path*"/"*sub_folder*subfolder*"initial_offer/")) == false
                mkdir(string(pwd()*"/"*file_path*"/"*sub_folder*subfolder*"initial_offer/"))
            end
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
    plt = plot(title = "Payoff timeseries for b=$b & c=$c", xlabel = "Generation", ylabel = "Relative Fitness")
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

function initoffer_timeseries_plots(df)
    b = df.b[1]
    c = df.c[1]
    plt = plot(title = "Initial offer timeseries for b=$b & c=$c", xlabel = "Generation", ylabel = "Initial Offer")
    colors = palette(:tab10)
    color_i = 1
    # df = sort(df, :generation)
    for group in groupby(df, :nnet)
        ts = []
        ws = []
        for replicate in groupby(group, :replicate_id)
            label_str = string("Network Size: ", group.nnet[1])
            # sort!(replicate, :generation)
            plt = plot!(replicate.generation, replicate.mean_initial_offer, legend = :none, alpha = .2)
        end
        for t in unique(group.generation)
            push!(ts, t)
            w_hat = mean(group.mean_initial_offer[group.generation .== t, :])
            push!(ws, w_hat)
        end

        plt = plot!(ts, ws, color = color_i, label = "")
        color_i += 1
    end
    return plt
end