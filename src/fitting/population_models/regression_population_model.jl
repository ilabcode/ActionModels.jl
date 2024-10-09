
# DONE: random intercepts (hint: construct model matrix somehow instead of modelmatrix(MixedEffects(@formula)), which expects y
# DONE: expand to multiple formulas / flexible names
# - DONE: parameters inside different statistical models gets overridden by each other!
# TODO: finish the prepare_data function
# DONE: think about tuple parameter names (ie initial values or HGF params)
# DONE: random slopes
# DONE: more than one random intercept
# DONE: intercept-only model
# DONE better / custom priors
# DONE: (1.0) check integration of the new functionality
# DONE: Compare with old implementation of specifying statistical model
# TODO: (1.0) Example / usecase / tutorials)
# DONE: check if we can get rid of TuringGLM
# DONE: support dropping intercepts (fixed and random)
# TODO: implement rename_chains for linear regressions
# DONE: prepare to merge
# TODO: merge
    #(search for FIXME)
# TODO: allow for varying priors: set up a regressionprior
# TODO: Decide whether to have a type including formula, prior and link function that the users use
# TODO: create model API decision
# TODO: add to documentation that there shoulnd't be random slopes for the most specific level of grouping column (particularly when you only have one grouping column)



using ActionModels, Turing, Distributions
##########################################################################################################
## User-level function for creating a æinear regression statiscial mdoel and using it with ActionModels ##
##########################################################################################################
function create_model(
    agent::Agent,
    regression_formulas::Union{F,Vector{F}},
    data::DataFrame;
    priors::Union{RegressionPrior,Vector{RegressionPrior}} = RegressionPrior(),
    link_functions::Union{Function,Vector{Function}} = identity,
    input_cols::Vector{C},
    action_cols::Vector{C},
    grouping_cols::Vector{C},
) where {F<:MixedModels.FormulaTerm,C<:Union{String,Symbol}}

    ## Setup ##

    #If there is only one formula
    if regression_formulas isa F
        #Put it in a vector
        regression_formulas = F[regression_formulas]
    end

    #If there is only one prior specified
    if priors isa RegressionPrior
        #Make a copy of it for each formula
        priors = RegressionPrior[priors for _ = 1:length(regression_formulas)]
    end

    #If there is only one link function specified
    if link_functions isa Function
        #Put it in a vector
        link_functions = Function[link_functions for _ = 1:length(regression_formulas)]
    end

    #Check that lengths are all the same
    if !(length(regression_formulas) == length(priors) == length(link_functions))
        throw(
            ArgumentError(
                "The number of regression formulas, priors, and link functions must be the same",
            ),
        )
    end

    #Extract just the data needed for the linear regression
    statistical_data = unique(data, grouping_cols)
    #Extract number of agents
    n_agents = nrow(statistical_data)

    ## Condition single regression models ##

    #Initialize vector of sinlge regression models
    regression_models = Vector{DynamicPPL.Model}(undef, length(regression_formulas))
    parameter_names = Vector{String}(undef, length(regression_formulas))

    #For each formula in the regression formulas, and its corresponding prior and link function
    for (model_idx, (formula, prior, link_function)) in
        enumerate(zip(regression_formulas, priors, link_functions))

        #Prepare the data for the regression model
        X, Z = prepare_regression_data(formula, statistical_data)

        #Convert to normal matrix
        Z = Matrix.(Z)
        
        #Condition the linear model
        regression_models[model_idx] =
            linear_model(X, Z, link_function = link_function, prior = prior)

        #Store the parameter name from the formula
        parameter_names[model_idx] = string(formula.lhs)
    end

    #Create the combined regression statistical model
    statistical_model =
        regression_statistical_model(regression_models, parameter_names, n_agents)

    return statistical_model

    #TODO: Instead of returning the model, pass it to the standard create_model function
    ##HERE##
end


#############################################################
## Turing model to do linear regression for each parameter ##
#############################################################
@model function regression_statistical_model(
    linear_submodels::Vector{T},
    parameter_names::Vector,
    n_agents::Int,
) where {T<:DynamicPPL.Model}

    #Initialize vector of dicts with agent parameters
    agents_params = Dict{String,Real}[
        Dict{String,Real}() for _ = 1:n_agents
    ]

    #For each parameter and its corresponding linear regression model
    for (linear_submodel, parameter_name) in zip(linear_submodels, parameter_names)
        #Run the linear regression model to extract parameters for each agent
        @submodel prefix = string(parameter_name) parameter_values = linear_submodel

        ## Map the output to the agent parameters ##
        #For each agent and the parameter value for the given parameter
        for (agent_idx, parameter_value) in enumerate(parameter_values)
            #Set it in the corresponding dictionary
            agents_params[agent_idx][parameter_name] = parameter_value
        end
    end

    return agents_params
end


#########################################
## Linear model for a single parameter ##
#########################################
"""
Generalized linear regression models following the equation:
θ = X⋅β
with optionally:
random effets: θ += Z⋅r
link function: link(θ)
"""
@model function linear_model(
    X::Matrix{R1}, # model matrix for fixed effects
    Z::Vector{MR}; # vector of model matrices for each random effect
    link_function::Function = identity,
    prior::RegressionPrior = RegressionPrior(),
    n_β::Int = size(X, 2), # number of fixed effect parameters
    size_r::Vector{Int} = size.(Z, 2), # number of random effect parameters, per group
    has_ranef::Bool = length(Z) > 0 && any(length.(Z) .> 0),
) where {R1<:Real,R2<:Real,MR<:Matrix{R2}}

    # FIXME: support different priors here

    #Sample beta / effect size parameters (including intercept)
    β ~ filldist(prior.β, n_β)

    #Do fixed effect linear regression
    η = X * β

    #If there are random effects
    if has_ranef
        
        #Initialize vector of random effect parameters
        r = Vector{Matrix{Real}}(undef, length(Z))

        #For each random effect j, and its corresponding model matrix Zⱼ
        for (ranefⱼ, (Zⱼ, size_rⱼ)) in enumerate(zip(Z, size_r))

            #Sample its parameters (both intercepts and slopes)
            r[ranefⱼ] ~ filldist(prior.r, size_rⱼ)

            #Add the random effect to the linear model
            η += Zⱼ * r[ranefⱼ] 
        end
    end

    #Apply the link function, and return the resulting parameter for each participant
    return link_function.(η)
end


#########################################################
## Function to prepare the data for a regression model ##
#########################################################
function prepare_regression_data(
    formula::MixedModels.FormulaTerm,
    statistical_data::DataFrame,
)

    #### COMNTINUE HERE #####
    ### Z is weird, because it looks like it treats age as categorical ###
    ### find out why this happens in MixedModel, when it does fixed effects just fine (know age is continuous) !!!!!!!!!! ###
    ### Using apply_schema beforehand breaks MixedModel (because MixedModel applies apply_schema itself)

    #Inset column with the name fo the agetn parameter, to avoid error from MixedModel
    insertcols!(statistical_data, Symbol(formula.lhs) => 1) #TODO: FIND SOMETHING LESS HACKY

    # formula = StatsModels.apply_schema(
    #     formula,
    #     StatsModels.schema(statistical_data),
    #     StatsModels.StatisticalModel,
    # )

    if ActionModels.has_ranef(formula)
        X = MixedModel(formula, statistical_data).feterm.x
        Z = MixedModel(formula, statistical_data).reterms
    else
        X = StatsModels.modelmatrix(formula, statistical_data)

        Z = []
    end

    return (X, Z)
end


####################################################
## Check if there are random effects in a formula ##
####################################################
function has_ranef(formula::FormulaTerm)

    #WORKS IF APPLY SCHEMA USED
    # return any(t -> t isa FunctionTerm{typeof(|)}, formula.rhs.terms)

    #WORKS OTHERWISE
    #If there is only one term
    if formula.rhs isa AbstractTerm
        #Check if it is a random effect
        if formula.rhs isa FunctionTerm{typeof(|)}
            return true
        else
            return false
        end
        #If there are multiple terms
    elseif formula.rhs isa Tuple
        #Check if any are random effects
        return any(t -> t isa FunctionTerm{typeof(|)}, formula.rhs)
    end
end




function rename_chains()
    #TODO

    # replacement_names = Dict()
    # for (param_name, _, __) in statistical_submodels
    #     for (idx, id) in enumerate(eachrow(statistical_data[!,grouping_cols]))
    #         if length(grouping_cols) > 1
    #             name_string = string(param_name) * "[$(Tuple(id))]"
    #         else
    #             name_string = string(param_name) * "[$(String(id[first(grouping_cols)]))]"
    #         end
    #         replacement_names[string(param_name) * ".agent_param[$idx]"] = name_string
    #     end
    # end
    # return replacenames(chains, replacement_names)
end














# function statistical_model_turingglm(
#     formula::TuringGLM.FormulaTerm,
#     data,
#     link_function::Function = identity;
#     priors::RegressionPrior = RegressionPrior(),
# ) where {T<:UnivariateDistribution}
#     # extract X and Z ( y is the output that goes to the agent model )
#     X = actionmodels_data_fixed_effects(formula, data)
#     if has_ranef(formula)
#         Z = actionmodels_data_random_effects(formula, data)
#     end

#     ranef = actionmodels_ranef(formula)

#     model = if ranef === nothing
#         _model(priors, [], [], 0)
#     else
#         intercept_ranef = TuringGLM.intercept_per_ranef(ranef)
#         group_var = first(ranef).rhs
#         idx = TuringGLM.get_idx(TuringGLM.term(group_var), data)
#         # idx is a tuple with 1. indices and 2. a dict mapping id names to indices
#         idxs = first(idx)
#         n_gr = length(unique(idxs))
#         # print for the user the idx
#         println("The idx are $(last(idx))\n")
#         _model(priors, intercept_ranef, idxs, n_gr)
#     end

#     return (model, X)
# end



# function data_fixed_effects(formula::FormulaTerm, data::D) where {D}
#     if has_ranef(formula)
#         X = MixedModels.modelmatrix(MixedModel(formula, data))
#     else
#         X = StatsModels.modelmatrix(
#             StatsModels.apply_schema(formula, StatsModels.schema(data)),
#             data,
#         )
#     end
#     return X
# end


# function actionmodels_data_random_effects(formula::FormulaTerm, data::D) where {D}
#     if !has_ranef(formula)
#         return nothing
#     end
#     slopes = TuringGLM.slope_per_ranef(actionmodels_ranef(formula))

#     Z = Dict{String,AbstractArray}() # empty Dict
#     if length(slopes) > 0
#         # add the slopes to Z
#         # this would need to create a vector from the column of the X matrix from the
#         # slope term
#         for slope in values(slopes.grouping_vars)
#             if slope isa String
#                 Z["slope_"*slope] = get_var(term(slope), data)
#             else
#                 for s in slope
#                     Z["slope_"*s] = get_var(term(s), data)
#                 end
#             end
#         end
#     else
#         Z = nothing
#     end
#     return Z
# end


# function actionmodels_ranef(formula::FormulaTerm)
#     if has_ranef(formula)
#         terms = filter(t -> t isa FunctionTerm{typeof(|)}, formula.rhs)
#         terms = map(terms) do t
#             lhs, rhs = first(t.args), last(t.args)
#             RandomEffectsTerm(lhs, rhs)
#         end
#         return terms
#     else
#         return nothing
#     end
# end
