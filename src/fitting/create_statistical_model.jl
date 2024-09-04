
# DONE: random intercepts (hint: construct model matrix somehow instead of modelmatrix(MixedEffects(@formula)), which expects y
# DONE: expand to multiple formulas / flexible names
# - DONE: parameters inside different statistical models gets overridden by each other!
# TODO: think about tuple parameter names (ie initial values or HGF params)
# TODO: random slopes
# TODO: more than one random intercept
# DONE: intercept-only model
# DONE better / custom priors
# DONE: (1.0) check integration of the new functionality
# TODO: Compare with old implementation of specifying statistical model
# TODO: (1.0) Example / usecase / tutorials)
# TODO: check if we can get rid of TuringGLM
# TODO: support dropping intercepts (fixed and random)
# TODO: implement rename_chains for linear regressions
# TODO: prepare to merge
# TODO: allow for varying priors
#

using ActionModels, Turing, Distributions

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
    Z::Vector{MR}, # model matrix for random effects
    link_function::Function = identity,
    prior::RegressionPrior = RegressionPrior();
    n_β::Int = size(X, 2), # number of fixed effect parameters
    size_r::Vector{Tuple{Int,Int}} = size.(Z), # number of random effect parameters, per group
    has_ranef::Bool = length(Z) > 0 && any(length.(Z) .> 0),
) where {R1<:Real,R2<:Real,MR<:Matrix{R2}}

    # FIXME: support different priors here
    β ~ filldist(prior.β, n_β)
    outcome = X * β

    if has_ranef

        r = Vector{Matrix{Real}}(undef, length(Z))

        for (IDⱼ, (Zⱼ, size_rⱼ)) in enumerate(zip(Z, size_r))

            r[IDⱼ] ~ filldist(prior.r, size_rⱼ...)

            outcome += sum(Zⱼ * transpose(r[IDⱼ]), dims = 2) #FIXME: MAKE SOMEONE CHECK THIS
        end
    end
    return link_function(outcome)
end


function prepare_regression_data(
    formula::MixedModels.FormulaTerm,
    statistical_data::DataFrame,
)

    insertcols!(statistical_data, Symbol(formula.lhs) => 1) #TODO: FIND SOMETHING LESS HACKY

    if ActionModels.has_ranef(formula)
        X = MixedModels.modelmatrix(MixedModel(formula, statistical_data))
    else
        X = StatsModels.modelmatrix(
            StatsModels.apply_schema(
                formula,
                StatsModels.schema(statistical_data),
                StatsModels.StatisticalModel,
            ),
            statistical_data,
        )
    end
end





# @model function regression_statistical_model(
#     linear_models::Vector{T},
#     parameter_names::Vector,
# ) where {T<:DynamicPPL.Model}

#     for (parameter_name, linear_submodel) in zip(linear_submodels, parameter_names)
#         @submodel prefix=string(parameter_name) parameter_values = linear_submodel
#     end


#     for (param_idx, (param_name, statistical_submodel, X)) in enumerate(statistical_submodels)
#         # run statistical_submodels
#         @submodel prefix=string(param_name) param_values[param_idx] = statistical_submodel(X)
#         # map output to agent parameters
#         for (agent_idx, param_value) in enumerate(param_values[param_idx])
#             agent_params[agent_idx][param_name] = param_value
#         end
#     end

# end











function statistical_model_turingglm(
    formula::TuringGLM.FormulaTerm,
    data,
    link_function::Function = identity;
    priors::RegressionPrior = RegressionPrior(),
) where {T<:UnivariateDistribution}
    # extract X and Z ( y is the output that goes to the agent model )
    X = actionmodels_data_fixed_effects(formula, data)
    if has_ranef(formula)
        Z = actionmodels_data_random_effects(formula, data)
    end

    ranef = actionmodels_ranef(formula)

    model = if ranef === nothing
        _model(priors, [], [], 0)
    else
        intercept_ranef = TuringGLM.intercept_per_ranef(ranef)
        group_var = first(ranef).rhs
        idx = TuringGLM.get_idx(TuringGLM.term(group_var), data)
        # idx is a tuple with 1. indices and 2. a dict mapping id names to indices
        idxs = first(idx)
        n_gr = length(unique(idxs))
        # print for the user the idx
        println("The idx are $(last(idx))\n")
        _model(priors, intercept_ranef, idxs, n_gr)
    end

    return (model, X)
end



## custom shims for working around TuringGLM
##
##
function has_ranef(formula::FormulaTerm)
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


function data_fixed_effects(formula::FormulaTerm, data::D) where {D}
    if has_ranef(formula)
        X = MixedModels.modelmatrix(MixedModel(formula, data))
    else
        X = StatsModels.modelmatrix(
            StatsModels.apply_schema(formula, StatsModels.schema(data)),
            data,
        )
    end
    return X
end


function actionmodels_data_random_effects(formula::FormulaTerm, data::D) where {D}
    if !has_ranef(formula)
        return nothing
    end
    slopes = TuringGLM.slope_per_ranef(actionmodels_ranef(formula))

    Z = Dict{String,AbstractArray}() # empty Dict
    if length(slopes) > 0
        # add the slopes to Z
        # this would need to create a vector from the column of the X matrix from the
        # slope term
        for slope in values(slopes.grouping_vars)
            if slope isa String
                Z["slope_"*slope] = get_var(term(slope), data)
            else
                for s in slope
                    Z["slope_"*s] = get_var(term(s), data)
                end
            end
        end
    else
        Z = nothing
    end
    return Z
end


function actionmodels_ranef(formula::FormulaTerm)
    if has_ranef(formula)
        terms = filter(t -> t isa FunctionTerm{typeof(|)}, formula.rhs)
        terms = map(terms) do t
            lhs, rhs = first(t.args), last(t.args)
            RandomEffectsTerm(lhs, rhs)
        end
        return terms
    else
        return nothing
    end
end
