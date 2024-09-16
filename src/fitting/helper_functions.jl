#####################################################################################################
####### FUNCTIONS FOR EXTRACTING A VALUE WHICH WORKS WITH DIFFERENT AUTODIFFERENTIATION BACKENDS ####
#####################################################################################################

function ad_val(x::ReverseDiff.TrackedReal)
    return ReverseDiff.value(x)
end
function ad_val(x::ReverseDiff.TrackedArray)
    return ReverseDiff.value(x)
end

function ad_val(x::ForwardDiff.Dual)
    return ForwardDiff.value(x)
end

function ad_val(x::Real)
    return x
end


###############################################
#### FUNCTION FOR CHECKING A CREATED MODEL ####
###############################################
function rename_chains(
    chains::Chains,
    model::DynamicPPL.Model,
    data::DataFrame,
    grouping_cols::Union{Vector{C},C},
) where {C<:Union{String,Symbol}}
    #This will multiple dispatch on the type of statistical model
    rename_chains(chains, data, grouping_cols, model.args.statistical_model.args...)
end


###############################################
#### FUNCTION FOR CHECKING A CREATED MODEL ####
###############################################
function check_model(
    agent::Agent,
    statistical_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3},
    track_states::Bool,
    verbose::Bool = true,
) where {T1<:Union{String,Symbol},T2<:Union{String,Symbol},T3<:Union{String,Symbol}}

    #Run the check of the statistical model    check_statistical_model(statistical_model.args...; verbose = verbose, agent = agent)

    #Check that user-specified columns exist in the dataset
    if any(grouping_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified group columns that do not exist in the dataframe",
            ),
        )
    elseif any(input_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified input columns that do not exist in the dataframe",
            ),
        )
    elseif any(action_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified action columns that do not exist in the dataframe",
            ),
        )
    end
end