using ActionModels
using Test

@testset "all tests" begin

    @testset "quick tests" begin

        # Test the quick tests that are used as pre-commit tests
        include("quicktests.jl")
    end

    @testset "utility tests" begin
        include("utility_tests.jl")
    end

    @testset "plot and fitting tests" begin
        include("fitting_tests.jl")

    end

end


@testset "documentation tests" begin
    AM_path = dirname(dirname(pathof(ActionModels)))
    documentation_path = AM_path * "/docs/src/julia_src_files/"

    # Test the quick tests that are used as pre-commit tests
    include(documentation_path * "agent_and_actionmodel.jl")
    include(documentation_path * "complicated_custom_agents.jl")
    include(documentation_path * "creating_own_action_model.jl")
    include(documentation_path * "fitting_an_agent_model_to_data.jl")
    include(documentation_path * "fitting_vs_simulating.jl")
    include(documentation_path * "index.jl")
    include(documentation_path * "premade_agents_and_models.jl")
    include(documentation_path * "prior_predictive_sim.jl")
    include(documentation_path * "simulation_with_an_agent.jl")
    include(documentation_path * "variations_of_util.jl")
end
