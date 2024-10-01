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
    action_distributions = Matrix(undef, size(actions))

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

#################################################
### WITH SINGLE ACTION / WITH MISSING ACTIONS ###
#################################################

@model function agent_model(agent::Agent, parameters::D, inputs::I, actions::Vector{Union{Missing, R}}) where {D<:Dict, I<:Vector, R<:Real}

    @show "yep"
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
                #TODO: Could use arraydist here if this was formatted as a vector of vectors (probably not!)
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