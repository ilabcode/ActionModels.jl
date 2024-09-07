
### TODO:
# Make check in full_model that returned agents_parameters is same length as inputs/actions
# Benchmark the try-catch in full_model
# Figure out helper function for extracting generated quantities (track_states true/false)
# Fix typing in create_model and full_model to give concrete types
# Make rename_chains also deal with missing actions
# Investigate why missing actions can be crazy samples

using Test
using ActionModels, DataFrames

@testset "fitting tests" begin

    ### SETUP ###
    #Create dataframe
    data = DataFrame(
        inputs = rand([0, 1], 20),
        inputs_2 = rand([0, 1], 20),
        actions = rand([0, 1], 20),
        actions_2 = rand([0, 1], 20),
        ID = [repeat(["A"], 5); repeat(["B"], 5); repeat(["A"], 5); repeat(["B"], 5)],
        category = [repeat(["X"], 10); repeat(["Y"], 10)],
    )

    #Create agent
    agent = premade_agent("binary_rescorla_wagner_softmax")

    #Set prior
    prior = Dict(
        "learning_rate" => LogitNormal(0.0, 1.0),
        "action_precision" => truncated(Normal(0.0, 1.0), lower = 0),
    )

    #Set samplings settings
    sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
    n_iterations = 10
    n_chains = 2


    @testset "single agent" begin
        #Extract inputs and actions from data
        inputs = data[!, :inputs]
        actions = data[!, :actions]

        #Create model
        model = create_model(agent, prior, inputs, actions)

        #Fit model
        fitted_model = sample(model, sampler, n_iterations, n_chains = n_chains)
    end

    @testset "simple statistical model" begin

        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :ID,
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations, n_chains = n_chains)

        #Rename chains
        renamed_model = rename_chains(fitted_model, prior, data, :ID)

        #Create model with tracking states
        model_tracked = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :ID,
            track_states = true,
        )

        #Extract quantities
        agent_parameters, agent_states, statistical_values =
            extract_quantities(fitted_model, model_tracked)
    end

    @testset "multiple grouping cols" begin

        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = [:ID, :category],
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations, n_chains = n_chains)

        #Rename chains
        renamed_model = rename_chains(fitted_model, prior, data, [:ID, :category])

        #Create model with tracking states
        model_tracked = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :ID,
            track_states = true,
        )

        #Extract quantities ###WHY DOES THIS FAIL?###
        agent_parameters, agent_states, statistical_values =
            extract_quantities(fitted_model, model_tracked)

    end

    @testset "missing actions" begin

        #Create new dataframe where three actions = missing
        new_data = allowmissing(data, :actions)
        new_data[[2, 7, 12], :actions] .= missing

        #Create model
        model = create_model(
            agent,
            prior,
            new_data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :ID,
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations, n_chains = n_chains)

        #Rename chains
        renamed_model = rename_chains(fitted_model, prior, data, :ID)

        #Create model with tracking states
        model_tracked = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :ID,
            track_states = true,
        )

        #Extract quantities
        agent_parameters, agent_states, statistical_values =
            extract_quantities(fitted_model, model_tracked)
    end

    @testset "custom statistical model" begin

    end

    @testset "multiple actions" begin

    end

    @testset "multiple inputs" begin

    end
end








# @testset "simulate actions and fit" begin

#     agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

#     param_priors = Dict("learning_rate" => Uniform(0, 1))

#     inputs = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

#     actions = give_inputs!(agent, inputs)

#     chains = fit_model(
#         agent,
#         param_priors,
#         inputs,
#         actions,
#         n_chains = 1,
#         n_iterations = 10,
#         verbose = false,
#         show_progress = false,
#     )

#     plot(chains)

#     plot_parameter_distribution(chains, param_priors)

#     get_posteriors(chains)


# end

# @testset "fit full dataframe" begin

#     agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

#     data = vcat(
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 1,
#             group = "A",
#             experiment = "1",
#         ),
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 2,
#             group = "A",
#             experiment = "1",
#         ),
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 3,
#             group = "B",
#             experiment = "1",
#         ),
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 4,
#             group = "B",
#             experiment = "1",
#         ),
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 3,
#             group = "C",
#             experiment = "1",
#         ),
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 4,
#             group = "C",
#             experiment = "1",
#         ),
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 1,
#             group = "A",
#             experiment = "2",
#         ),
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 2,
#             group = "A",
#             experiment = "2",
#         ),
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 3,
#             group = "B",
#             experiment = "2",
#         ),
#         DataFrame(
#             input = [1, 0, 1],
#             action = [1, 0, 1],
#             ID = 4,
#             group = "B",
#             experiment = "2",
#         ),
#     )

#     independent_group_cols = [:experiment]
#     multilevel_group_cols = [:ID, :group]
#     input_cols = [:input]
#     action_cols = [:action]

#     priors = Dict(
#         "learning_rate" => Multilevel(
#             :ID,
#             LogitNormal,
#             ["learning_rate_ID_mean", "learning_rate_ID_sd"],
#         ),
#         "learning_rate_ID_mean" => Multilevel(
#             :group,
#             Normal,
#             ["learning_rate_group_mean", "learning_rate_group_sd"],
#         ),
#         "learning_rate_ID_sd" => LogNormal(0, 1),
#         "learning_rate_group_sd" => LogNormal(0, 1),
#         "learning_rate_group_mean" => Normal(0, 1),
#         "action_precision" => LogNormal(0, 1),
#     )

#     results = fit_model(
#         agent,
#         priors,
#         data;
#         independent_group_cols = independent_group_cols,
#         multilevel_group_cols = multilevel_group_cols,
#         input_cols = input_cols,
#         action_cols = action_cols,
#         n_cores = 2,
#         n_iterations = 10,
#         n_chains = 2,
#         verbose = false,
#         show_progress = false,
#     )

# end

# @testset "ensure parameters are reset after fitting" begin

#     agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

#     initial_parameters = get_parameters(agent)

#     param_priors = Dict("learning_rate" => Uniform(0, 1))

#     inputs = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

#     actions = give_inputs!(agent, inputs)

#     chains = fit_model(
#         agent,
#         param_priors,
#         inputs,
#         actions,
#         n_chains = 1,
#         n_iterations = 10,
#         verbose = false,
#         show_progress = false,
#     )

#     @test get_parameters(agent) == initial_parameters

# end


# @testset "Make sure fitting allows using a custom sampler" begin

#     agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

#     param_priors = Dict("learning_rate" => Uniform(0, 1))

#     inputs = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

#     actions = give_inputs!(agent, inputs)

#     rwmh = externalsampler(AdvancedMH.RWMH(10))

#     chains = fit_model(
#         agent,
#         param_priors,
#         inputs,
#         actions,
#         n_chains = 1,
#         n_iterations = 10,
#         sampler = rwmh,
#         verbose = false,
#         show_progress = false,
#     )

# end


# @testset "Test with multiple actions" begin

#     #Action model with multiple actions    
#     function testactionfunc(agent, input::Vector)

#         noise = agent.parameters["noise"]

#         input1, input2, input3 = input

#         actiondist1 = Normal(input1, noise)
#         actiondist2 = Normal(input2, noise)

#         return [actiondist1, actiondist2]
#     end
#     #Create agent
#     agent = init_agent(testactionfunc, parameters = Dict("noise" => 1.0))

#     #Create inputs
#     inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

#     #Simulate actions
#     actions = give_inputs!(agent, inputs)

#     #Set priors
#     priors = Dict("noise" => LogNormal(0.0, 1.0))

#     #Fit
#     results = fit_model(
#         agent,
#         priors,
#         inputs,
#         actions,
#         n_iterations = 10,
#         verbose = false,
#         show_progress = false,
#     )
# end
