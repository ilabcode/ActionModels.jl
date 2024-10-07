

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
) where T<:Union{Type{Dict}, Type{DataFrame}}

    get_estimates(agent_parameters, summary_function, output_type) 

end


#############################################################################################################
####### DSISPATCH FUNCTION FOR GENERATING A DATAFRAME ####
#############################################################################################################
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
    for (i, agent) in enumerate(agents)
        row = Dict()
        for (j, parameter) in enumerate(parameters)
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
function get_estimates(agent_parameters::AxisArray{
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
    estimates_dict = Dict{Symbol, Dict{Symbol, Float64}}()

    # Populate the dictionary with median values
    for (i, agent) in enumerate(agents)
        agent_dict = Dict{Symbol, Float64}()
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