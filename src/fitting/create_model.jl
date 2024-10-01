###########################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A STATISTICAL MODEL ###
###########################################################################################################
function create_model(
    agent::Agent,
    statistical_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3} = Vector{Symbol}(),
    check_parameter_rejections::Union{Nothing, CheckRejections} = nothing,
    id_separator::String = "__",
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
        statistical_model,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        verbose = verbose,
    )

    ## Extract data ##
    #One matrix per agent, for inputs and actions separately
    #FIXME: put inputs into correct format
    inputs =
        [Array(agent_data[:, input_cols]) for agent_data in groupby(data, grouping_cols)]
    actions =
        [Array(agent_data[:, action_cols]) for agent_data in groupby(data, grouping_cols)]

    #Extract agent id's as combined symbols in a vector
    agent_ids = [Symbol(join(string.(Tuple(row)), id_separator)) for row in eachrow(unique(data[!, grouping_cols]))]

    #Create a full model combining the agent model and the statistical model
    return full_model(agent_model, statistical_model, inputs, actions, agent_ids, check_parameter_rejections = check_parameter_rejections)
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
    check_parameter_rejections::Nothing = nothing,
) where {I<:Vector, R<:Real, A1 <:Union{R,Union{Missing,R}}, A<:Array{A1}}

    #Generate the agent parameters from the statistical model
    @submodel population_values = population_model

    ## For each agent ##
    for (agent_id, agent_parameters, agent_inputs, agent_actions) in zip(agent_ids, population_values.agent_parameters, inputs_per_agent, actions_per_agent)

        @submodel prefix = agent_id agent_model(agent, agent_parameters, agent_inputs, agent_actions)
    end

    #Return agents' parameters and tracked states
    return statistical_model_return
end


#############################################################################
### WRAPPER FUNCTION FOR FULL_MODEL FOR CHECKING FOR PARAMETER REJECTIONS ###
#############################################################################

# @model function full_model(
#     agent::Agent,
#     statistical_model::DynamicPPL.Model,
#     inputs_per_agent::Array{IA},
#     actions_per_agent::Array{AA};
#     agent_ids::Vector{Union{Symbol,Vector{Symbol}}},
#     check_parameter_rejections::CheckRejections,
# ) where {IAR<:Union{Real,Missing},AAR<:Union{Real,Missing},IA<:Array{IAR},AA<:Array{AAR}}

#     #Check whether errors occur
#     try

#         #Run the full model
#         @submodel generated_quantities = full_model(
#             agent,
#             statistical_model,
#             inputs_per_agent,
#             actions_per_agent;
#             agent_ids = agent_ids,
#             check_parameter_rejections = nothing,
#         )

#         return generated_quantities

#         #If an error occurs
#     catch error
#         #If it is of the custom errortype RejectParameters
#         if error isa RejectParameters
#             #Make Turing reject the sample
#             Turing.@addlogprob!(-Inf)
#         else
#             #Otherwise, just throw the error
#             rethrow(error)
#         end
#     end
# end