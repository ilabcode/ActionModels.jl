using ActionModels
using Test
# using CSV
# using DataFrames
# using Plots
# using StatsPlots

@testset "quick tests" begin
    
    # Test the quick tests that are used as pre-commit tests
    include("quicktests.jl")
end
