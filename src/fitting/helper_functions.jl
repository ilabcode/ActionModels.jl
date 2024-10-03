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


############################################################
#### FUNCTION FOR RENAMING THE CHAINS OF A FITTED MODEL ####
############################################################
function rename_chains(
    chains::Chains,
    model::DynamicPPL.Model,
) where {C<:Union{String,Symbol}}

    #This will multiple dispatch on the type of statistical model
    rename_chains(chains, model, model.args.population_model.args...)
end


###############################################
#### FUNCTION FOR CHECKING A CREATED MODEL ####
###############################################
function check_model(
    agent::Agent,
    population_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3},
    verbose::Bool = true,
) where {T1<:Union{String,Symbol},T2<:Union{String,Symbol},T3<:Union{String,Symbol}}

    #TODO： Make check for whether the agent model outputs the right amount of actions / accepts the right amoiunts of inputs

    #Run the check of the statistical model    
    check_population_model(population_model.args...; verbose = verbose, agent = agent)

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