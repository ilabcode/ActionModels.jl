"""
    get_history(agent::Agent, target_state::Union{String,Tuple})

Get the history of a single state from an agent. Returns a vector.

    get_history(agent::Agent, target_states::Vector)

Get set of a vector of states from an agent. Returns a dictionary of states and their histories.

    get_history(agent::Agent)

Get histories for states from an agent. Returns a dictionary of states and their histories.
"""
function get_history end


### Functions for getting a single state ###
function get_history(agent::Agent, target_state::Union{String,Tuple})
    #If the state is in the agent's history
    if target_state in keys(agent.history)
        #Extract it
        state_history = agent.history[target_state]
    else
        #Otherwise look in the substruct
        state_history = get_history(agent.substruct, target_state)
    end

    return state_history
end

function get_history(substruct::Nothing, target_state::Union{String,Tuple})
    throw(
        ArgumentError(
            "The specified state $target_state does not exist in the agent's history",
        ),
    )
    return nothing
end


### Functions for getting multiple states ###
function get_history(agent::Agent, target_states::Vector)
    #Initialize dict
    state_histories = Dict()

    #Go through each state
    for state_name in target_states
        #Get them with get_history, and add to the tuple
        state_histories[state_name] = get_history(agent, state_name)
    end

    return state_histories
end


### Function for getting all states ###
function get_history(agent::Agent)

    #Get the agent's states' histories
    state_histories = agent.history

    #Get state histories from the substruct
    substruct_state_histories = get_history(agent.substruct)

    #Add them to the agent's states
    state_histories = merge(substruct_state_histories, state_histories)

    return state_histories
end

function get_history(substruct::Nothing)
    #For empty substructs, return an empty list
    return Dict()
end
