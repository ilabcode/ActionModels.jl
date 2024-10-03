###############################################
### WITH SINGLE ACTION / NO MISSING ACTIONS ###
###############################################
@model function agent_models(agent::Agent, agent_ids::Vector{Symbol}, parameters_per_agent::Vector{D}, inputs_per_agent::Vector{I}, actions_per_agent::Vector{Vector{R}}, actions_flattened::Vector{R}, missing_actions::Nothing) where {D<:Dict, I<:Vector, R<:Real}

    #TODO: Could use a list comprehension here to make it more efficient
    #Initialize a vector for storing the action probability distributions
    action_distributions = Vector(undef, length(actions_flattened))

    #Initialize action index
    action_idx = 0
    
    #Go through each agent
    for (agent_parameters, agent_inputs, agent_actions) in zip(parameters_per_agent, inputs_per_agent, actions_per_agent)

        #Set the agent parameters
        set_parameters!(agent, agent_parameters)
        reset!(agent)

        #Go through each timestep 
        for (input, action) in zip(agent_inputs, agent_actions)

            #Increment one action index
            action_idx += 1

            #Get the action probability distributions from the action model
            @inbounds action_distributions[action_idx] = agent.action_model(agent, input)

            #Store the agent's action in the agent
            update_states!(agent, "action", action)
        end
    end
            
    #Make sure the action distributions are stored as a concrete type (by constructing a new vector)
    action_distributions = [dist for dist in action_distributions]

    #Sample the actions from the probability distributions
    actions_flattened ~ arraydist(action_distributions)
end


##################################################
### WITH MULTIPLE ACTIONS / NO MISSING ACTIONS ###
##################################################
@model function agent_models(agent::Agent, agent_ids::Vector{Symbol}, parameters_per_agent::Vector{D}, inputs_per_agent::Vector{I}, actions_per_agent::Vector{Matrix{R}}, actions_flattened::Matrix{R}, missing_actions::Nothing) where {D<:Dict, I<:Vector, R<:Real}

    #Initialize a vector for storing the action probability distributions
    action_distributions = Matrix(undef, size(actions_flattened)...)

    #Initialize action index
    action_idx = 0
    
    #Go through each agent
    for (agent_parameters, agent_inputs, agent_actions) in zip(parameters_per_agent, inputs_per_agent, actions_per_agent)

        #Set the agent parameters
        set_parameters!(agent, agent_parameters)
        reset!(agent)

        #Go through each timestep 
        for (input, action) in zip(agent_inputs, Tuple.(eachrow(agent_actions)))

            #Increment one action index
            action_idx += 1

            #Get the action probability distributions from the action model
            @inbounds action_distributions[action_idx, :] = agent.action_model(agent, input)

            #Store the agent's action in the agent
            update_states!(agent, "action", action)
        end
    end
            
    #Make sure the action distributions are stored as a concrete type (by constructing a new vector)
    action_distributions = [dist for dist in action_distributions]

    #Sample the actions from the probability distributions
    actions_flattened ~ arraydist(action_distributions)
end









############################################
### WITH MISSING ACTIONS - SUPERFUNCTION ###
############################################
@model function agent_models(agent::Agent, agent_ids::Vector{Symbol}, parameters_per_agent::Vector{D}, inputs_per_agent::Vector{I}, actions_per_agent::Vector{A}, actions_flattened::A2, missing_actions::MissingActions) where {D<:Dict, I<:Vector, A<:Array, A2<:Array}

    #For each agent 
    for (agent_id, agent_parameters, agent_inputs, agent_actions) in zip(agent_ids, parameters_per_agent, inputs_per_agent, actions_per_agent)

        #Fit it to the data
        @submodel prefix = "$agent_id" agent_model(agent, agent_parameters, agent_inputs, agent_actions)
    end
end

#################################################
### WITH SINGLE ACTION / WITH MISSING ACTIONS ###
#################################################
@model function agent_model(agent::Agent, parameters::D, inputs::I, actions::Vector{Union{Missing, R}}) where {D<:Dict, I<:Vector, R<:Real}

    #Set the agent parameters
    set_parameters!(agent, parameters)
    reset!(agent)

    #Go through each timestep 
    for (timestep, input) in enumerate(inputs)

        #Get the action probability distributions from the action model
        action_distribution = agent.action_model(agent, input)

        #Sample the action from the probability distribution
        @inbounds actions[timestep] ~ action_distribution

        #Save the action to the agent in case it needs it in the future
        @inbounds update_states!(
            agent,
            "action",
            ad_val(actions[timestep]),
        )
    end
end

####################################################
### WITH MULTIPLE ACTIONS / WITH MISSING ACTIONS ###
####################################################
@model function agent_model(agent::Agent, parameters::D, inputs::I, actions::Matrix{Union{Missing, R}}) where {D<:Dict, I<:Vector, R<:Real}

    #Set the agent parameters
    set_parameters!(agent, parameters)
    reset!(agent)

    #Go through each timestep 
    for (timestep, input) in enumerate(inputs)

        #Get the action probability distributions from the action model
        action_distributions = agent.action_model(agent, input)

        #Go through each action
        for (action_idx, single_distribution) in enumerate(action_distributions)

            #Sample the action from the probability distribution
            actions[timestep, action_idx] ~
                single_distribution
                #TODO: can use @inbounds here when there's a check for whether the right amount of actions are produced
        end

        #Add the actions to the agent in case it needs it in the future
        update_states!(
            agent,
            "action",
            ad_val.(actions[timestep, :]),
        )
        #TODO: can use @inbounds here when there's a check for whether the right amount of actions are produced
    end
end

###############################################
### WITH SINGLE ACTION / NO MISSING ACTIONS ###
###############################################
@model function agent_model(agent::Agent, parameters::D, inputs::I, actions::Vector{R}) where {D<:Dict, I<:Vector, R<:Real}

    #Set the agent parameters
    set_parameters!(agent, parameters)
    reset!(agent)

    #Initialize a vector for storing the action probability distributions
    action_distributions = Vector(undef, length(inputs))

    #Go through each timestep 
    for (timestep, (input, action)) in enumerate(zip(inputs, actions))

        #Get the action probability distributions from the action model
        @inbounds action_distributions[timestep] = agent.action_model(agent, input)

        #Store the agent's action in the agent
        update_states!(agent, "action", action)
    end

    #Make sure the action distributions are stored as a concrete type (by constructing a new vector)
    action_distributions = [dist for dist in action_distributions]

    #Sample the actions from the probability distributions
    actions ~ arraydist(action_distributions)
end

##################################################
### WITH MULTIPLE ACTIONS / NO MISSING ACTIONS ###
##################################################

@model function agent_model(agent::Agent, parameters::D, inputs::I, actions::Matrix{R}) where {D<:Dict, I<:Vector, R<:Real}

    #Set the agent parameters
    set_parameters!(agent, parameters)
    reset!(agent)

    #Initialize a matrix for storing the action probability distributions
    action_distributions = Matrix(undef, size(actions)...)

    #Go through each timestep 
    for (timestep, (input, action)) in enumerate(zip(inputs, Tuple.(eachrow(actions))))

        #Get the action probability distributions from the action model
        action_distributions[timestep, :] = agent.action_model(agent, input) #TODO: can use @inbounds here when there's a check for whether the right amount of actions are used

        #Store the agent's action in the agent
        update_states!(agent, "action", action)
    end

    #Make sure the action distributions are stored as a concrete type (by constructing a new matrix)
    action_distributions = [dist for dist in action_distributions]

    #Sample the actions from the probability distributions
    actions ~ arraydist(action_distributions)
end