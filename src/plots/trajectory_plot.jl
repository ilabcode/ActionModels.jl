"""
"""
function trajectory_plot(agent::AgentStruct, target_state::Union{String,Tuple}; kwargs...)

    #If the target state is in the agent's history
    if target_state in keys(agent.history)
        #Plot that
        agent_trajectory_plot(agent, target_state; kwargs...)
        #Otherwise
    else
        #Look in the substruct
        trajectory_plot(agent.substruct, target_state; kwargs...)
    end
end

"""
"""
function trajectory_plot!(agent::AgentStruct, target_state::Union{String,Tuple}; kwargs...)

    #If the target state is in the agent's history
    if target_state in keys(agent.history)
        #Plot that
        agent_trajectory_plot!(agent, target_state; kwargs...)
        #Otherwise
    else
        #Look in the substruct
        trajectory_plot!(agent.substruct, target_state; kwargs...)
    end
end


"""
"""
function trajectory_plot(substruct::Nothing, target_state::Union{String,Tuple}; kwargs...)
    throw(ArgumentError("The specified state does not exist in the agent's history"))
end

"""
"""
function trajectory_plot!(substruct::Nothing, target_state::Union{String,Tuple}; kwargs...)
    throw(ArgumentError("The specified state does not exist in the history"))
end


@userplot Agent_Trajectory_Plot

@recipe function f(pl::Agent_Trajectory_Plot)

    #Get out the agent and the target state
    agent = pl.args[1]
    target_state = pl.args[2]

    #Get the history of the state
    state_history = agent.history[target_state]
    #Replace missings with NaNs for plotting
    state_history = replace(state_history, missing => NaN)

    #Plot the history
    @series begin
        seriestype --> :scatter
        label --> target_state
        markersize --> 5
        state_history
    end
end