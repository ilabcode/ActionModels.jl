###########################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A STATISTICAL MODEL ###
###########################################################################################################
function create_model(
    agent::Agent,
    population_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3},
    check_parameter_rejections::Union{Nothing, CheckRejections} = nothing,
    id_separator::String = ".",
    verbose::Bool = true,
) where {T1<:Union{String,Symbol},T2<:Union{String,Symbol},T3<:Union{String,Symbol}}

    ## SETUP ##
    #Create a copy of the agent to avoid changing the original 
    agent_model = deepcopy(agent)

    #Turn off saving the history of states
    set_save_history!(agent_model, false)

    ## Make sure columns are vectors of symbols ##
    if !(input_cols isa Vector)
        input_cols = [input_cols]
    end
    input_cols = Symbol.(input_cols)

    if !(action_cols isa Vector)
        action_cols = [action_cols]
    end
    action_cols = Symbol.(action_cols)

    if !(grouping_cols isa Vector)
        grouping_cols = [grouping_cols]
    end
    grouping_cols = Symbol.(grouping_cols)

    #Run checks for the model specifications
    check_model(
        agent,
        population_model,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        verbose = verbose,
    )
    
    
    ## Extract data ##
    #If there is only one input column
    if length(input_cols) == 1
        #Inputs are a vector of vectors of <:reals
        inputs = [Vector(agent_data[!,first(input_cols)]) for agent_data in groupby(data, grouping_cols)]
    else
        #Otherwise, they are a vector of vectors of tuples
        inputs = [Tuple.(eachrow(agent_data[!,input_cols])) for agent_data in groupby(data, grouping_cols)]
    end
    
    #If there is only one action column
    if length(action_cols) == 1
        #Actions are a vector of arrays (vectors if there is only one action, matrices if there are multiple)
        actions =
            [Vector(agent_data[!, first(action_cols)]) for agent_data in groupby(data, grouping_cols)]
    else
        #Actions are a vector of arrays (vectors if there is only one action, matrices if there are multiple)
        actions =
            [Array(agent_data[!, action_cols]) for agent_data in groupby(data, grouping_cols)]
    end

    #Extract agent id's as combined symbols in a vector
    # agent_ids = [
    #     Symbol(join(string.(Tuple(row)), id_separator))
    #      for row in eachrow(unique(data[!, grouping_cols]))]
    agent_ids = [Symbol(join([string(col_name) * "" * string(row[col_name]) for col_name in grouping_cols], id_separator)) for row in eachrow(unique(data[!, grouping_cols]))]

    ## Determine whether any actions are missing ##
    if actions isa Vector{A} where {R<:Real, A<:Array{Union{Missing, R}}}
        #If there are missing actions
        missing_actions = MissingActions()
    elseif actions isa Vector{A} where {R<:Real, A<:Array{R}}
        #If there are no missing actions
        missing_actions = nothing
    end

    #Create a full model combining the agent model and the statistical model
    return full_model(agent_model, population_model, inputs, actions, agent_ids, missing_actions = missing_actions, check_parameter_rejections = check_parameter_rejections)
end

####################################################################
### FUNCTION FOR DOING FULL AGENT AND STATISTICAL MODEL COMBINED ###
####################################################################
@model function full_model(
    agent::Agent,
    population_model::DynamicPPL.Model,
    inputs_per_agent::Vector{I},
    actions_per_agent::Vector{A},
    agent_ids::Vector{Symbol};
    missing_actions::Union{Nothing, MissingActions} = MissingActions(),
    check_parameter_rejections::Nothing = nothing,
    actions_flattened::A2 = vcat(actions_per_agent...)
) where {I<:Vector, R<:Real, A1 <:Union{R,Union{Missing,R}}, A<:Array{A1}, A2<:Array}

    #Generate the agent parameters from the statistical model
    @submodel population_values = population_model

    #Generate the agent's behavior
    @submodel agent_models(agent, agent_ids, population_values.agent_parameters, inputs_per_agent, actions_per_agent, actions_flattened, missing_actions)

    #Return values fron the population model (agent parameters and oher values)
    return population_values
end