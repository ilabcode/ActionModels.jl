
using Pkg
Pkg.activate("../../docs")

using Test
using Distributions
using TuringGLM
using DataFrames
using LogExpFunctions
using MixedModels

using Revise
using ActionModels


@testset "GLM tests" begin

    example_data = DataFrame(
        input = repeat([1, 1, 1, 2, 2, 2], 3),
        actions = [
            0,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,    
            0,
            0.5,
            0.8,
            1,
            1.5,
            1.8,
            0,
            2,
            0.5,
            4,
            5,
            3,
        ],
        age = vcat([repeat([20], 6), repeat([24], 6), repeat([28], 6)]...),
        category = vcat([repeat(["1"], 6), repeat(["2"], 6), repeat(["2"], 6)]...),
        id = vcat([repeat(["Hans"], 6), repeat(["georg"], 6), repeat(["Jørgen"], 6)]...),
    )
    example_data_combined = begin
        example_data1 = example_data
        example_data1[!, :treatment] = repeat(["control"], 18)

        example_data2 = DataFrame(
        input = repeat([1, 1, 1, 2, 2, 2], 3),
        actions = [
            0,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,    
            0,
            0.2,
            0.4,
            0.7,
            1.0,
            1.1,
            0,
            1,
            0.3,
            3,
            1,
            3,
        ],
        age = vcat([repeat([20], 6), repeat([24], 6), repeat([28], 6)]...),
        category = vcat([repeat(["1"], 6), repeat(["2"], 6), repeat(["2"], 6)]...),
        id = vcat([repeat(["Hans"], 6), repeat(["georg"], 6), repeat(["Jørgen"], 6)]...),
        treatment = repeat(["drug"], 18)
    )

        vcat(example_data1, example_data2)
        
        end

    @testset "dummy turingglm non-hierarchical model" begin
        prior = CustomPrior(Normal(0, 10), Normal(0, 10), nothing)
        formula = @formula(actions ~ 1)
        model = turing_model(formula, example_data; priors = prior)
        samples = sample(model, NUTS(), 10)
    end

    @testset "dummy turingglm hierarchical model" begin
        prior = CustomPrior(Normal(0, 10), Normal(0, 10), nothing)
        formula = @formula(actions ~ age + (1 | id))
        model = turing_model(formula, example_data; priors = prior)
        samples = sample(model, NUTS(), 10)
    end

    @testset "forward sim RW" begin
        agent = premade_agent("continuous_rescorla_wagner_gaussian")
        give_inputs!(agent, example_data[:, :input])
        lr_dist = Uniform(0.1, 0.5)
        # @show learning_rate = rand(lr_dist, length(unique(example_data[:, :id])))

        for df in groupby(example_data, :id)
            @show lr = rand(lr_dist)
            set_parameters!(agent, Dict("learning_rate" => lr))
            reset!(agent)
            @show actions = give_inputs!(agent, df[:, :input])
            @show get_history(agent, "value")
        end
    end

    # @testset "old: estimate non-hierarchical learning rate for RW" begin
    #     prior = Dict("learning_rate" => Uniform(0, 1))
    #     samples = fit_model(agent, prior, example_data;
    #                         independent_group_cols = [:id], action_cols = [:actions], input_cols = [:input])
    # end

    @testset "new interface for statistical model - intercept only" begin
        #prior = Dict("learning_rate" => Uniform(0, 1))
        agent = premade_agent("continuous_rescorla_wagner_gaussian")
        # non hierarchical
        samples = fit_model(
            agent,
            @formula(learning_rate ~ 1),
            example_data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id],
        )


        @show summary(samples)

    end
    @testset "new interface for statistical model - intercept + random effect only" begin
        # # hierarchical
        samples = fit_model(
            agent,
            @formula(learning_rate ~ 1 + (1 | id)),
            example_data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id],
        )
        @show summary(samples)
    end

    @testset "new interface for statistical model - fixed effects" begin
        #prior = Dict("learning_rate" => Uniform(0, 1))

        # non hierarchical
        # samples = fit_model(agent,
        #                     @formula(learning_rate ~ 1)
        #                     example_data;
        #                     action_cols = [:actions],
        #                     input_cols = [:input])

        agent = premade_agent("continuous_rescorla_wagner_gaussian")
        samples = fit_model(
            agent,
            @formula(learning_rate ~ id),
            example_data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id],
        )

        @show summary(samples)

        # # hierarchical
        # samples = fit_model(agent, prior, example_data;
        #                     action_cols = [:actions], input_cols = [:input],
        #                     statistical_model = @formula(learning_rate ~ 1 + (1|id)))

        # [@formula(), @formula(), @formula()] # dont assume defaults -- dont estimate non-specified params
    end

    @testset "new interface for statistical model - random intercept" begin
        agent = premade_agent("continuous_rescorla_wagner_gaussian")
        samples = fit_model(
            agent,
            (@formula(learning_rate ~ age + (1 | id)), LogitNormal),
            example_data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id],
            n_iterations = 100,
        )

        @show summary(samples)
    end

    @testset "new interface for statistical model - random slope" begin
        agent = premade_agent("continuous_rescorla_wagner_gaussian")
        samples = ActionModels.create_model(
            agent,
            @formula(learning_rate ~ age + (1 + age | treatment)),
            example_data_combined;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        @show summary(samples)
    end


    @testset "new interface for statistical model - multiple formulas" begin
        agent = premade_agent("continuous_rescorla_wagner_gaussian")

        samples = fit_model(
            agent,
            [
                (@formula(learning_rate ~ age + (1 | id)), LogitNormal),
                (@formula(action_noise ~ age + (1 | id)), LogNormal),
            ],
            example_data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id],
            n_iterations = 10,
        )

        @show samples
    end


    @testset "single linear model interface" begin

        X = [1 7 9; 10 2 3]
        Z = [[1 2; 3 4], [1 2 4 10 12; 3 4 10 12 20]]
        model = ActionModels.linear_model(X, Z)

        fitted_model = sample(model, NUTS(), 1000)
    end

    @testset "prepare data" begin

        formula0 = @formula(learning_rate ~ 0)
        formula1 = @formula(learning_rate ~ 1)
        formula2 = @formula(learning_rate ~ age)
        formula3 = @formula(learning_rate ~ 0 + age)
        formula4 = @formula(learning_rate ~ (1 | id))
        formula5 = @formula(learning_rate ~ 0 + (1 | id))
        formula6 = @formula(learning_rate ~ age + (1 + age | id) + (1 | category))
        formula7 = @formula(learning_rate ~ age + (1 + age | treatment))


        ActionModels.prepare_regression_data(formula0, unique(example_data, :id))

        ActionModels.prepare_regression_data(formula1, unique(example_data, :id))

        ActionModels.prepare_regression_data(formula2, unique(example_data, :id))

        ActionModels.prepare_regression_data(formula3, unique(example_data, :id))

        ActionModels.prepare_regression_data(formula4, unique(example_data, :id))

        ActionModels.prepare_regression_data(formula5, unique(example_data, :id))

        ActionModels.prepare_regression_data(formula6, unique(example_data, :id))

        X, Z = ActionModels.prepare_regression_data(formula7, unique(example_data_combined, [:treatment, :id]))

    end
end