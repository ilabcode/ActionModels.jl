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
    agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

    #Set prior
    prior = Dict(
        "learning_rate" => LogitNormal(0.0, 1.0),
        "action_precision" => truncated(Normal(0.0, 1.0), lower = 0),
    )

    #Set samplings settings
    sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
    n_iterations = 10
    n_chains = 2
    sampling_kwargs = (; progress = false)


    @testset "single agent" begin
        #Extract inputs and actions from data
        inputs = data[!, :inputs]
        actions = data[!, :actions]

        #Create model
        model = create_model(agent, prior, inputs, actions)

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        renamed_model = rename_chains(fitted_model, model, data, :ID)
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
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model, data, :ID)

        #Create model with tracking states
        model_tracked = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :ID,
        )

        #Extract quantities
        agent_parameters, agent_states, statistical_values =
            extract_quantities(fitted_model, model_tracked)
    end

    @testset "no grouping cols" begin

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
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model, data, [:ID, :category])

        #Create model with tracking states
        model_tracked = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = [:ID, :category],
        )

        #Extract quantities
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
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model, data, :ID)

        #Create model with tracking states
        model_tracked = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :ID,
        )

        #Extract quantities
        agent_parameters, agent_states, statistical_values =
            extract_quantities(fitted_model, model_tracked)
    end

    @testset "custom statistical model" begin

    end

    @testset "multiple actions" begin

        #Action model with multiple actions    
        function multi_action(agent, input::Real)

            noise = agent.parameters["noise"]

            actiondist1 = Normal(input, noise)
            actiondist2 = Normal(input, noise)

            return [actiondist1, actiondist2]
        end
        #Create agent
        new_agent = init_agent(multi_action, parameters = Dict("noise" => 1.0))

        new_prior = Dict("noise" => LogNormal(0.0, 1.0))

        #Create model
        model = create_model(
            new_agent,
            new_prior,
            data,
            input_cols = :inputs,
            action_cols = [:actions, :actions_2],
            grouping_cols = :ID,
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

    end

    @testset "multiple inputs" begin

        #Action model with multiple actions    
        function multi_input(agent, input::Tuple{R,R}) where {R<:Real}

            noise = agent.parameters["noise"]

            input1, input2 = input

            actiondist = Normal(input1, noise)

            return actiondist
        end
        #Create agent
        new_agent = init_agent(multi_input, parameters = Dict("noise" => 1.0))

        new_prior = Dict("noise" => LogNormal(0.0, 1.0))

        #Create model
        model = create_model(
            new_agent,
            new_prior,
            data,
            input_cols = [:inputs, :inputs_2],
            action_cols = :actions,
            grouping_cols = :ID,
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)
    end

    @testset "multiple inputs and multiple actions" begin

        #Action model with multiple actions    
        function multi_input_action(agent, input::Tuple{R,R}) where {R<:Real}

            noise = agent.parameters["noise"]

            input1, input2 = input

            actiondist1 = Normal(input1, noise)
            actiondist2 = Normal(input2, noise)

            return [actiondist1, actiondist2]
        end
        #Create agent
        new_agent = init_agent(multi_input_action, parameters = Dict("noise" => 1.0))

        new_prior = Dict("noise" => LogNormal(0.0, 1.0))

        #Create model
        model = create_model(
            new_agent,
            new_prior,
            data,
            input_cols = [:inputs, :inputs_2],
            action_cols = [:actions, :actions_2],
            grouping_cols = :ID,
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)
    end
end




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