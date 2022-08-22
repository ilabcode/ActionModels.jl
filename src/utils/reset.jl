"""
"""
function reset!(agent::AgentStruct)

    #For each of the agent's states
    for state_name in keys(agent.states)
        #Set it to the first value in the history
        agent.states[state_name] = agent.history[state_name][1]
    end

    #For each state in the history
    for state in keys(agent.history)
        #Reset the history to the new state
        agent.history[state] = [agent.states[state]]
    end

    #Reset the agents substruct
    reset!(agent.substruct)
end

"""
"""
function reset!(substruct::Nothing)
    return nothing
end