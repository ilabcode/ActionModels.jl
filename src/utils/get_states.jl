"""
    get_states(agent::Agent, target_state::Union{String,Tuple})

Get a single state from an agent. Returns a single value.

    get_states(agent::Agent, target_state::Vector)

Get a set of state values from an agent. Returns a dictionary of state names and their values.

    get_states(agent::Agent)

Get all states from an agent. Returns a dictionary of state names and their values.
"""
function get_states end


### Functions for getting a single state
function get_states(agent::Agent, target_state::Union{String,Tuple})
    #If the state is in the agent's states
    if target_state in keys(agent.states)
        #Extract it from the agent
        state = agent.states[target_state]
        #Otherwise
    else
        #Look in the substruct
        state = get_states(agent.substruct, target_state)
    end

    return state
end

function get_states(substruct::Nothing, target_state::Union{String,Tuple})
    throw(
        ArgumentError(
            "The specified state $state_name does not exist in the agent or in the substructure",
        ),
    )

end


### Function for getting multiple states ###
function get_states(agent::Agent, target_states::Vector)

    #Initialize tuple for populating with states
    states = Dict()

    #Go through each state name
    for state_name in target_states
        #Add its value to the list
        states[state_name] = get_states(agent, state_name)
    end

    return states
end


### Function for getting all of an agent's states
function get_states(agent::Agent)

    #Get all state names for the agent
    target_states = collect(keys(agent.states))

    #Get the agent's states 
    agent_states = get_states(agent, target_states)

    #Get states from the substruct
    substruct_states = get_states(agent.substruct)

    #Merge into one list
    states = merge(substruct_states, agent_states)

    return states
end

function get_states(substruct::Nothing)
    return Dict()
end
