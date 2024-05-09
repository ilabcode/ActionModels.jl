### Function which updates a single state, and saves it history
function update_states!(agent::Agent, state::String, value)
    #Update state
    agent.states[state] = value

    #Save to history
    if agent.save_history
        push!(agent.history[state], value)
    end
end


### Function which updates a dictionary of states to their values
function update_states!(agent::Agent, states::Dict)
    #For each state and value
    for (state, value) in states
        #Update it
        update_states!(agent, state, value)
    end
end
