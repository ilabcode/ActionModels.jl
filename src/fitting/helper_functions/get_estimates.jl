############################################################################################
####### FUNCTION FOR GENERATING SUMMARIZED VARIABLES FROM AN AGENT_PARMAETERS AXISARRAY ####
############################################################################################
function get_estimates(
    agent_parameters::AxisArray{
        Float64,
        4,
        Array{Float64,4},
        Tuple{
            Axis{:agent,Vector{Symbol}},
            Axis{:parameter,Vector{Symbol}},
            Axis{:sample,UnitRange{Int64}},
            Axis{:chain,UnitRange{Int64}},
        },
    },
    output_type::T = DataFrame,
    summary_function::Function = median,
) where {T<:Union{Type{Dict},Type{DataFrame}}}

    get_estimates(agent_parameters, summary_function, output_type)

end


##########################################################
####### DSISPATCH FUNCTION FOR GENERATING A DATAFRAME ####
##########################################################
function get_estimates(
    agent_parameters::AxisArray{
        Float64,
        4,
        Array{Float64,4},
        Tuple{
            Axis{:agent,Vector{Symbol}},
            Axis{:parameter,Vector{Symbol}},
            Axis{:sample,UnitRange{Int64}},
            Axis{:chain,UnitRange{Int64}},
        },
    },
    output_type::Type{DataFrame},
    summary_function::Function = median,
)

    #Extract agents and parameters
    agents = agent_parameters.axes[1]
    parameters = agent_parameters.axes[2]

    # Initialize an empty DataFrame
    df = DataFrame(Dict(Symbol(parameter) => Float64[] for parameter in parameters))
    df[!, :agent] = Symbol[]

    # Populate the DataFrame with median values
    for agent in agents
        row = Dict()
        for parameter in parameters
            # Extract the values for the current agent and parameter across samples and chains
            values = agent_parameters[agent, parameter, :, :]
            # Calculate the median value
            median_value = summary_function(values)
            # Add the median value to the row
            row[Symbol(parameter)] = median_value
        end
        #Add an agent id to the row
        row[:agent] = agent
        # Add the row to the DataFrame
        push!(df, row)
    end

    # Reorder the columns to have agent_id as the first column
    select!(df, :agent, names(df)[1:end-1]...)

    return df
end


#########################################################
####### VERSION WHICH GENERATES A DICTIONARY INSTEAD ####
#########################################################
function get_estimates(
    agent_parameters::AxisArray{
        Float64,
        4,
        Array{Float64,4},
        Tuple{
            Axis{:agent,Vector{Symbol}},
            Axis{:parameter,Vector{Symbol}},
            Axis{:sample,UnitRange{Int64}},
            Axis{:chain,UnitRange{Int64}},
        },
    },
    output_type::Type{Dict},
    summary_function::Function = median,
)

    #Extract agents and parameters
    agents = agent_parameters.axes[1]
    parameters = agent_parameters.axes[2]

    # Initialize an empty dictionary
    estimates_dict = Dict{Symbol,Dict{Symbol,Float64}}()

    # Populate the dictionary with median values
    for (i, agent) in enumerate(agents)
        agent_dict = Dict{Symbol,Float64}()
        for (j, parameter) in enumerate(parameters)
            # Extract the values for the current agent and parameter across samples and chains
            values = agent_parameters[agent, parameter, :, :]
            # Calculate the median value
            median_value = summary_function(values)
            # Add the median value to the agent's dictionary
            agent_dict[Symbol(parameter)] = median_value
        end
        # Add the agent's dictionary to the main dictionary
        estimates_dict[Symbol(agent)] = agent_dict
    end

    return estimates_dict
end








###########################################################################################
###### FUNCTION FOR GENERATING SUMMARIZED VARIABLES FROM AN AGENT_PARMAETERS AXISARRAY ####
###########################################################################################
function get_estimates(
    state_trajectories::AxisArrays.AxisArray{
        Union{Missing,Float64},
        5,
        Array{Union{Missing,Float64},5},
        Tuple{
            AxisArrays.Axis{:agent,Vector{Symbol}},
            AxisArrays.Axis{:state,Vector{Symbol}},
            AxisArrays.Axis{:timestep,UnitRange{Int64}},
            AxisArrays.Axis{:sample,UnitRange{Int64}},
            AxisArrays.Axis{:chain,UnitRange{Int64}},
        },
    },
    summary_function::Function = median,
)

    #Extract agents and parameters
    agents = state_trajectories.axes[1]
    states = state_trajectories.axes[2]
    timesteps = state_trajectories.axes[3]

    # Initialize an empty DataFrame
    df = DataFrame(
        Dict(
            begin
                #Join tuples
                if state isa Tuple
                    state = join(state, tuple_separator)
                end

                #Join the agent and the state
                Symbol(join((string(agent), string(state)), id_separator)) => Float64[]
            end for (state, agent) in Iterators.product(states, agents)
        ),
    )
    df[!, :timestep] = Int[]

    # Populate the DataFrame with median values
    for timestep in timesteps
        row = Dict()

        for agent in agents

            for state in states
                # Extract the state for the current agent and state, at the current timestep
                values = state_trajectories[agent, state, timestep+1, :, :]
                # Calculate the point estimate
                median_value = summary_function(values)

                #Join tuples
                if state isa Tuple
                    state = join(state, tuple_separator)
                end

                # Add the value to the row
                row[Symbol(join((string(agent), string(state)), id_separator))] =
                    median_value
            end

            #Add the timestep to the row
            row[:timestep] = timestep
        end

        # Add the row to the DataFrame
        push!(df, row, promote = true)
    end

    # Reorder the columns to have agent_id as the first column
    select!(df, :timestep, names(df)[1:end-1]...)

    return df
end
