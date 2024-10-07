using Test
using ActionModels, DataFrames
@testset "fitting tests" begin

    ### SETUP ###
    #Create dataframe
    data = DataFrame(
        inputs = repeat([1, 1, 1, 2, 2, 2], 3),
        inputs_2 = repeat([1, 1, 1, 2, 2, 2], 3),
        actions = [0,0.5,0.8,1,1.5,1.8,0,0.2,0.3,0.4,0.5,0.6,0,2,0.5,4,5,3,],
        actions_2 = [0,0.5,0.8,1,1.5,1.8,0,0.2,0.3,0.4,0.5,0.6,0,2,0.5,4,5,3,],
        age = vcat([repeat([20], 6), repeat([22], 6), repeat([28], 6)]...),
        category = vcat([repeat(["1"], 6), repeat(["2"], 6), repeat(["2"], 6)]...),
        id = vcat([repeat(["Hans"], 6), repeat(["Georg"], 6), repeat(["Jørgen"], 6)]...),
    )

    #Create agent
    agent = premade_agent("continuous_rescorla_wagner_gaussian", verbose = false)

    #Set prior
    prior = Dict(
        "learning_rate" => LogitNormal(0.0, 1.0),
        "action_noise" => truncated(Normal(0.0, 1.0), lower = 0),
        ("initial", "value") => Normal(0.0, 1.0),
    )

    #Set samplings settings
    sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
    n_iterations = 1000
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

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)
       
        renamed_model = rename_chains(fitted_model, model)
    end

    @testset "simple statistical model" begin

        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :id,
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)
    
        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
    end

    @testset "custom statistical model" begin

    end

    @testset "no grouping cols" begin
        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = Symbol[]
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        # #Extract quantities
        # agent_parameters = extract_quantities(model, fitted_model)

        # #Rename chains
        # renamed_model = rename_chains(fitted_model, model)
    end

    @testset "multiple grouping cols" begin

        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = [:id, :category],
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
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
            grouping_cols = :id,
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
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

        #Set prior
        new_prior = Dict("noise" => LogNormal(0.0, 1.0))

        #Create model
        model = create_model(
            new_agent,
            new_prior,
            data,
            input_cols = :inputs,
            action_cols = [:actions, :actions_2],
            grouping_cols = :id,
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)

    end

    @testset "multiple actions, missing actions" begin
        

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
            grouping_cols = :id,
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
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
            grouping_cols = :id,
        )

        #Fit model
        fitted_model =
            sample(model, sampler, n_iterations; n_chains = n_chains, sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
    end
end
