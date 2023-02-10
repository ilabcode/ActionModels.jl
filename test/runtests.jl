using ActionModels
using Test
# using CSV
# using DataFrames
# using Plots
# using StatsPlots

@testset "all tests" begin

    @testset "quick tests" begin

        # Test the quick tests that are used as pre-commit tests
        include("quicktests.jl")
    end

    @testset "utility tests" begin
        include("utility_tests.jl")
    end

    @testset "fitting tests" begin
        
    end

end