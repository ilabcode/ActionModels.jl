using ActionModels
using HierarchicalGaussianFiltering
using Test
using CSV
using DataFrames
using Turing
using Plots
using StatsPlots

@testset "HGF tests" begin
    
    #Set up path for tutorials
    actionmodels_path = dirname(dirname(pathof(ActionModels)))
    tutorials_path = actionmodels_path * "/docs/tutorials/" 

    # include(tutorials_path * "hgf_tutorials/" * "classic_binary.jl")
    # include(tutorials_path * "hgf_tutorials/" *"classic_usdchf.jl")

end