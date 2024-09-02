using Test
using ActionModels, Distributions, Distributed

@testset "parameter recovery" begin

    @testset "non-parallelizsed" begin

        #Agent model to do recovery on
        agent = premade_agent("continuous_rescorla_wagner_gaussian", verbose = false)

        #Parameters to be recovered
        parameter_ranges = Dict(
            "learning_rate" => collect(0:0.1:1),
            ("initial", "value") => collect(-2:1:2),
            "action_noise" => collect(0:0.5:3),
        )

        #Input sequences to use
        input_sequence = [[1, 2, 1, 0, 0, 1, 1, 2, 1, 2], [2, 3, 1, 5, 4, 8, 6, 4, 5]]

        #Sets of priors to use
        priors = [
            Dict(
                "learning_rate" => LogitNormal(0, 1),
                ("initial", "value") => Normal(0, 1),
                "action_noise" => truncated(Normal(0, 1), lower = 0),
            ),
            Dict(
                "learning_rate" => LogitNormal(0, 0.1),
                ("initial", "value") => Normal(0, 0.1),
                "action_noise" => truncated(Normal(0, 0.1), lower = 0),
            ),
        ]

        #Times to repeat each simulation
        n_simulations = 2

        #Sampler settings
        sampler_settings = (n_iterations = 10, n_chains = 1)

        #Run parameter recovery
        results_df = parameter_recovery(
            agent,
            parameter_ranges,
            input_sequence,
            priors,
            n_simulations,
            sampler_settings = sampler_settings,
            show_progress = false,
        )

        @test results_df isa DataFrame

    end

    @testset "parallelised" begin

        addprocs(2, exeflags = "--project")

        @everywhere begin
            using ActionModels, Distributions

            #Agent model to do recovery on
            agent = premade_agent("continuous_rescorla_wagner_gaussian", verbose = false)

            #Parameters to be recovered
            parameter_ranges = Dict(
                "learning_rate" => collect(0:0.1:1),
                ("initial", "value") => collect(-2:1:2),
                "action_noise" => collect(0:0.5:3),
            )

            #Input sequences to use
            input_sequence = [[1, 2, 1, 0, 0, 1, 1, 2, 1, 2], [2, 3, 1, 5, 4, 8, 6, 4, 5]]

            #Sets of priors to use
            priors = [
                Dict(
                    "learning_rate" => LogitNormal(0, 1),
                    ("initial", "value") => Normal(0, 1),
                    "action_noise" => truncated(Normal(0, 1), lower = 0),
                ),
                Dict(
                    "learning_rate" => LogitNormal(0, 0.1),
                    ("initial", "value") => Normal(0, 0.1),
                    "action_noise" => truncated(Normal(0, 0.1), lower = 0),
                ),
            ]

            #Times to repeat each simulation
            n_simulations = 2

            #Sampler settings
            sampler_settings = (n_iterations = 10, n_chains = 1)
        end

        #Run parameter recovery
        results_df = parameter_recovery(
            agent,
            parameter_ranges,
            input_sequence,
            priors,
            n_simulations,
            sampler_settings = sampler_settings,
            parallel = true,
            show_progress = false,
        )

        rmprocs(workers())

        @test results_df isa DataFrame

    end
end
