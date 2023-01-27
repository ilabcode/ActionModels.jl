"""
    reset!(agent::Agent)

Reset an agent to its initial state. Use initial state parameters when specified, and otherwise just use the first value in the state history.
"""
function reset!(agent::Agent)

    #For each of the agent's states
    for state_name in keys(agent.states)

        #If the state has an initial state parameter
        if state_name in keys(agent.initial_state_parameters)
            #Set it to that value
            agent.states[state_name] = agent.initial_state_parameters[state_name]
        else
            #Set it to the first value in the history
            agent.states[state_name] = agent.history[state_name][1]
        end
    end

    #For each state in the history
    for state in keys(agent.history)
        #Reset the history to the new state
        agent.history[state] = [agent.states[state]]
    end

    #Reset the agents substruct
    reset!(agent.substruct)
end

function reset!(substruct::Nothing)
    return nothing
end
