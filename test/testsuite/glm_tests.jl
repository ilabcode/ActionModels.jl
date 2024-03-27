
using Pkg
Pkg.activate("../../docs")

using Test
using Distributions
using TuringGLM
using DataFrames

using Revise
using ActionModels


@testset "GLM tests" begin

    example_data = DataFrame(input   = repeat([1,1,1,2,2,2], 3),
                             actions = [0, 0.5, 0.8, 1, 1.5, 1.8,
                                        0, 0.2, 0.3, 0.4, 0.5, 0.6,
                                        0, 2, 0.5, 4, 5, 3],
                             age = vcat([repeat([20],6), repeat([22], 6), repeat([28],6)]...),
                             category = vcat([repeat(["1"],6), repeat(["2"], 6), repeat(["2"],6)]...),
                             id = vcat([repeat(["Hans"],6), repeat(["georg"], 6), repeat(["Jørgen"],6)]...))

    @testset "dummy turingglm non-hierarchical model" begin
        prior = CustomPrior(Normal(0,10), Normal(0,10), nothing)
        formula = @formula(actions ~ id + age)
        model = turing_model(formula, example_data; priors = prior)
        samples = sample(model, NUTS(), 10)
    end

    @testset "dummy turingglm hierarchical model" begin
        prior = CustomPrior(Normal(0,10), Normal(0,10), nothing)
        formula = @formula(actions ~ age + (1|id))
        model = turing_model(formula, example_data; priors = prior)
        samples = sample(model, NUTS(), 10)
    end

    @testset "forward sim RW" begin
        agent = premade_agent("continuous_rescorla_wagner")
        give_inputs!(agent, example_data[:,:input])
        lr_dist = Uniform(0.1, 0.5)
        # @show learning_rate = rand(lr_dist, length(unique(example_data[:, :id])))

        for df in groupby(example_data, :id)
            @show lr = rand(lr_dist)
            set_parameters!(agent, Dict("learning_rate" => lr))
            reset!(agent)
            @show actions = give_inputs!(agent, df[:,:input])
            @show get_history(agent, "value")
        end
    end

    # @testset "old: estimate non-hierarchical learning rate for RW" begin
    #     prior = Dict("learning_rate" => Uniform(0, 1))
    #     samples = fit_model(agent, prior, example_data;
    #                         independent_group_cols = [:id], action_cols = [:actions], input_cols = [:input])
    # end

    @testset "new interface for statistical model - fixed effects" begin
        #prior = Dict("learning_rate" => Uniform(0, 1))

        # non hierarchical
        # samples = fit_model(agent,
        #                     @formula(learning_rate ~ 1)
        #                     example_data;
        #                     action_cols = [:actions],
        #                     input_cols = [:input])

        agent = premade_agent("continuous_rescorla_wagner")
        samples = fit_model(agent,
                            @formula(learning_rate ~ id),
                            example_data;
                            action_cols   = [:actions],
                            input_cols    = [:input],
                            grouping_cols = [:id])

        @show summary(samples)

        # # hierarchical
        # samples = fit_model(agent, prior, example_data;
        #                     action_cols = [:actions], input_cols = [:input],
        #                     statistical_model = @formula(learning_rate ~ 1 + (1|id)))

        # [@formula(), @formula(), @formula()] # dont assume defaults -- dont estimate non-specified params
    end

    @testset "new interface for statistical model - random effects" begin

        agent = premade_agent("continuous_rescorla_wagner")
        samples = fit_model(agent,
                            @formula(learning_rate ~ 1 + (1|id)),
                            example_data;
                            action_cols   = [:actions],
                            input_cols    = [:input],
                            grouping_cols = [:id])

        @show summary(samples)

    end



end
