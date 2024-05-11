
using AdvancedMH
using ActionModels
using Test
using Distributions
using DataFrames
using Plots
using StatsPlots
using Turing: externalsampler


@testset "simulate actions and fit" begin

    agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

    param_priors = Dict("learning_rate" => Uniform(0, 1))

    inputs = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

    actions = give_inputs!(agent, inputs)

    chains = fit_model(
        agent,
        param_priors,
        inputs,
        actions,
        n_chains = 1,
        n_iterations = 10,
        verbose = false,
    )

    plot(chains)

    plot_parameter_distribution(chains, param_priors)

    get_posteriors(chains)


end





@testset "fit full dataframe" begin

    agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

    data = vcat(
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 1,
            group = "A",
            experiment = "1",
        ),
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 2,
            group = "A",
            experiment = "1",
        ),
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 3,
            group = "B",
            experiment = "1",
        ),
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 4,
            group = "B",
            experiment = "1",
        ),
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 3,
            group = "C",
            experiment = "1",
        ),
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 4,
            group = "C",
            experiment = "1",
        ),
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 1,
            group = "A",
            experiment = "2",
        ),
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 2,
            group = "A",
            experiment = "2",
        ),
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 3,
            group = "B",
            experiment = "2",
        ),
        DataFrame(
            input = [1, 0, 1],
            action = [1, 0, 1],
            ID = 4,
            group = "B",
            experiment = "2",
        ),
    )

    independent_group_cols = [:experiment]
    multilevel_group_cols = [:ID, :group]
    input_cols = [:input]
    action_cols = [:action]

    priors = Dict(
        "learning_rate" => Multilevel(
            :ID,
            LogitNormal,
            ["learning_rate_ID_mean", "learning_rate_ID_sd"],
        ),
        "learning_rate_ID_mean" => Multilevel(
            :group,
            Normal,
            ["learning_rate_group_mean", "learning_rate_group_sd"],
        ),
        "learning_rate_ID_sd" => LogNormal(0, 1),
        "learning_rate_group_sd" => LogNormal(0, 1),
        "learning_rate_group_mean" => Normal(0, 1),
        "action_precision" => LogNormal(0, 1),
    )

    results = fit_model(
        agent,
        priors,
        data;
        independent_group_cols = independent_group_cols,
        multilevel_group_cols = multilevel_group_cols,
        input_cols = input_cols,
        action_cols = action_cols,
        n_cores = 2,
        n_iterations = 10,
        n_chains = 2,
        progress = false,
        verbose = false,
    )

end


@testset "ensure parameters are reset after fitting" begin

    agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

    initial_parameters = get_parameters(agent)

    param_priors = Dict("learning_rate" => Uniform(0, 1))

    inputs = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

    actions = give_inputs!(agent, inputs)

    chains = fit_model(
        agent,
        param_priors,
        inputs,
        actions,
        n_chains = 1,
        n_iterations = 10,
        verbose = false,
    )

    @test get_parameters(agent) == initial_parameters

end


@testset "Make sure fitting allows using a custom sampler" begin
    
    agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

    param_priors = Dict("learning_rate" => Uniform(0, 1))

    inputs = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

    actions = give_inputs!(agent, inputs)

    rwmh = externalsampler(AdvancedMH.RWMH(10))

    chains = fit_model(
        agent,
        param_priors,
        inputs,
        actions,
        n_chains = 1,
        n_iterations = 10,
        verbose = false,
        sampler = ext_sampler,
    )
    
end