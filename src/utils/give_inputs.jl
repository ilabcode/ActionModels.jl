"""
    single_input!(agent::AgentStruct, input)

Function for giving an input to an AgentStruct.
"""
function single_input!(agent::AgentStruct, input::Any)

    #Run the action model to get the action distribution
    action_distribution = agent.action_model(agent, input)

    #If a single action distribution is returned
    if length(action_distribution) == 1

        #Sample an action from the distribution
        agent.states["action"] = rand(action_distribution)

        #If multiple action distributions are returned
    else
        #Initialize vector for storing actions
        actions = []

        #For each action distribution
        for distribution in action_distribution
            #Sample an action and add it to the vector
            push!(actions, rand(distribution))
        end

        #And store it 
        agent.states["action"] = actions
    end

    #Record the action
    push!(agent.history["action"], agent.states["action"])

    #Return the action
    return agent.states["action"]
end


### Functions for inputting multiple values ###
"""
    give_inputs!(agent::AgentStruct, inputs::Real)

Convenience method for inputting multiple observations to an agent. Input is here just a single value.
"""
function give_inputs!(agent::AgentStruct, inputs::Real)

    #Input the single input
    single_input!(agent, inputs)

    #Return the action trajectory
    return agent.states["action"]
end

"""
    give_inputs!(agent::AgentStruct, inputs::Vector)

Function for inputting multiple observations to an agent. Input is structured as an Array, with each row being a single input consisting of multiple numbers.
"""
function give_inputs!(agent::AgentStruct, inputs::Vector)

    #Each value in the vector is a single input
    for input in inputs

        #Input that row
        single_input!(agent, input)

    end

    #Return the action trajectory, without the initial state
    return agent.history["action"][2:end]
end

"""
    give_inputs!(agent::AgentStruct, inputs::Array)

Function for inputting multiple observations to an agent. Input is structured as an Array, with each row being a single input consisting of multiple numbers.
"""
function give_inputs!(agent::AgentStruct, inputs::Array)

    #Each row in the array is a single input
    for input in eachrow(inputs)

        #Input that row
        single_input!(agent, input)

    end

    #Return the action trajectory
    return agent.history["action"][2:end]
end