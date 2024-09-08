using ActionModels, DataFrames
using Test
using Glob

#Get the root path
ActionModels_path = dirname(dirname(pathof(ActionModels)))

@testset "all tests" begin

    test_path = ActionModels_path * "/test/"

    @testset "quick tests" begin
        # Test the quick tests that are used as pre-commit tests
        include(test_path * "quicktests.jl")
    end

    @testset "testsuite" begin

        # List the julia filenames in the testsuite
        filenames = glob("*.jl", test_path * "testsuite")

        # For each file
        for filename in filenames
            #Run it
            include(filename)
        end
    end

    @testset "Documentation" begin
        documentation_path = ActionModels_path * "/docs/src/"
        @testset "sourcefiles" begin

            # List the julia filenames in the documentation source files folder
            filenames = glob("*.jl", documentation_path * "/Julia_src_files")

            for filename in filenames
                include(filename)
            end
        end
    end
end
