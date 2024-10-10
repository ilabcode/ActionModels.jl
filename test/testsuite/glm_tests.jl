
using Pkg
Pkg.activate("../../docs")

using Test
using LogExpFunctions

using ActionModels
using Distributions
using DataFrames
using MixedModels
using Turing


@testset "linear regression tests" begin

    #Generate dataset
    data = DataFrame(
        input = repeat([1, 1, 1, 2, 2, 2], 6),
        actions = vcat(
            [0, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0, 0.5, 0.8, 1, 1.5, 1.8],
            [0, 2, 0.5, 4, 5, 3],
            [0, 0.1, 0.15, 0.2, 0.25, 0.3],
            [0, 0.2, 0.4, 0.7, 1.0, 1.1],
            [0, 2, 0.5, 4, 5, 3],
        ),
        age = vcat(
            repeat([20], 6),
            repeat([24], 6),
            repeat([28], 6),
            repeat([20], 6),
            repeat([24], 6),
            repeat([28], 6),
        ),
        id = vcat(
            repeat(["Hans"], 6),
            repeat(["Georg"], 6),
            repeat(["Jørgen"], 6),
            repeat(["Hans"], 6),
            repeat(["Georg"], 6),
            repeat(["Jørgen"], 6),
        ),
        treatment = vcat(repeat(["control"], 18), repeat(["treatment"], 18)),
    )

    #Create agent
    agent = premade_agent("continuous_rescorla_wagner_gaussian")

    #Set samplings settings
    sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
    n_iterations = 1000
    sampling_kwargs = (; progress = false)

    @testset "intercept only" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ 1),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)

    end
    @testset "intercept + random effect only" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ 1 + (1 | id)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "THIS IS WRONG: MISSIGN IMPLICIT INTERCEPT fixed effect only" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "fixed effect and random intercept by id" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 | id)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "fixed effect and random intercept by id and treatment" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 | id) + (1 | treatment)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "fixed effect, random intercept + slope by treatment" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 + age | treatment)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "fixed effect, random intercept + slope by treatment" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 + age | treatment) + (1 | id)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "THIS ERRORS: order of random effects reversed" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 | id) + (1 + age | treatment) ),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "multiple formulas" begin
        model = create_model(
            agent,
            [
                @formula(learning_rate ~ age + (1 | id)),
                @formula(action_noise ~ age + (1 | id)),
            ],
            data;
            link_functions = [identity, LogExpFunctions.exp],
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "manual prior specification" begin
        model = create_model(
            agent,
            [
                @formula(learning_rate ~ age + (1 + age | treatment) + (1 | id)),
                @formula(action_noise ~ age + (1 | id)),
            ],
            data;
            link_functions = [logistic, LogExpFunctions.exp],
            priors = [
                RegressionPrior(
                    β = [Normal(0, 1), Normal(0, 1)],
                    σ = [[LogNormal(0, 1), LogNormal(0, 1)], [LogNormal(0, 1)]],
                ),
                RegressionPrior(
                    β = Normal(0, 1),
                ),
            ],
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )
        
        samples = sample(model, sampler, n_iterations; sampling_kwargs...)

        agent_parameters = extract_quantities(model, samples)
        agent_parameter_df = get_estimates(agent_parameters)

        trajectories = get_trajectories(model, samples, ["value", "action"])
        state_trajectories_df = get_estimates(trajectories)

        using StatsPlots
        plot_trajectories(trajectories)

        prior_samples = sample(model, Prior(), n_iterations; sampling_kwargs...)

        prior_agent_parameters = extract_quantities(model, prior_samples)

        prior_trajectories = get_trajectories(model, prior_samples, ["value", "action"])
        plot_trajectories(prior_trajectories)

        plot_parameters(prior_samples, samples)
    end

    # @testset "single linear model interface" begin

    #     X = [1 7 9; 10 2 3]
    #     Z = [[1 2; 3 4], [1 2 4 10 12; 3 4 10 12 20]]
    #     model = ActionModels.linear_model(X, Z)

    #     fitted_model = sample(model, NUTS(), 1000)
    # end

    # @testset "prepare data" begin

    #     formula0 = @formula(learning_rate ~ 0)
    #     formula1 = @formula(learning_rate ~ 1)
    #     formula2 = @formula(learning_rate ~ age)
    #     formula3 = @formula(learning_rate ~ 0 + age)
    #     formula4 = @formula(learning_rate ~ (1 | id))
    #     formula5 = @formula(learning_rate ~ 0 + (1 | id))
    #     formula6 = @formula(learning_rate ~ age + (1 + age | id) + (1 | category))
    #     formula7 = @formula(learning_rate ~ age + (1 + age | treatment))


    #     ActionModels.prepare_regression_data(formula0, unique(example_data, :id))

    #     ActionModels.prepare_regression_data(formula1, unique(example_data, :id))

    #     ActionModels.prepare_regression_data(formula2, unique(example_data, :id))

    #     ActionModels.prepare_regression_data(formula3, unique(example_data, :id))

    #     ActionModels.prepare_regression_data(formula4, unique(example_data, :id))

    #     ActionModels.prepare_regression_data(formula5, unique(example_data, :id))

    #     ActionModels.prepare_regression_data(formula6, unique(example_data, :id))

    #     X, Z = ActionModels.prepare_regression_data(
    #         formula7,
    #         unique(example_data_combined, [:treatment, :id]),
    #     )
    # end
end
