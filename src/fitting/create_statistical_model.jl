
# DONE: random intercepts (hint: construct model matrix somehow instead of modelmatrix(MixedEffects(@formula)), which expects y
# DONE: expand to multiple formulas / flexible names
# - DONE: parameters inside different statistical models gets overridden by each other!
# TODO: think about tuple parameter names (ie initial values or HGF params)
# TODO: random slopes
# DONE: intercept-only model
# DONE better / custom priors
# TODO: (1.0) check integration of the new functionality
# - Compare with old implementation of specifying statistical model
# TODO: (1.0) Example / usecase / tutorials)
# TODO: check if we can go back to turingglm._statistical_model_turingglm() with a few changes
# TODO: support dropping intercepts (fixed and random)
# TODO: re-prettify out (:id names) (including making it extract? hans/georg/jørgen)
# - One approach: build a vector for MCMCChains.Chains.
#   - bug: returns all the same numbers.
#   - todo: how to combine with existing chains object
#

function statistical_model_turingglm(
    formula::TuringGLM.FormulaTerm,
    data,
    link_function::Function = identity;
    priors::RegressionPrior = RegressionPrior()
) where {T<:UnivariateDistribution}
    # extract X and Z ( y is the output that goes to the agent model )
    X = actionmodels_data_fixed_effects(formula, data)
    if actionmodels_has_ranef(formula)
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

# Linear model
function _model(prior, intercept_ranef, idxs, n_gr)
    @model function linear_model(
        X;
        n_β=size(X, 2),
        idxs=idxs,
        n_gr=n_gr,
        intercept_ranef=intercept_ranef,
        prior=prior,
    )
        α ~ prior.α
        if n_β != 0
            β ~ TuringGLM.filldist(prior.β, n_β)
            agent_param = α .+ X * β
        else
            agent_param = α .+ repeat([0], size(X, 1))
        end

        if !isempty(intercept_ranef)
            τ ~ prior.τ
            zⱼ ~ filldist(Normal(), n_gr)
            agent_param = agent_param .+ τ .* getindex.((zⱼ,), idxs)
        end
        return agent_param
    end
end


## custom shims for working around TuringGLM
##
##
function actionmodels_has_ranef(formula::FormulaTerm)
    if formula.rhs isa StatsModels.Term
        return false
    elseif formula.rhs isa StatsModels.ConstantTerm
        return false
    else
        return any(t -> t isa FunctionTerm{typeof(|)}, formula.rhs)
    end
end


function actionmodels_data_fixed_effects(formula::FormulaTerm, data::D) where {D}
    if actionmodels_has_ranef(formula)
        X = MixedModels.modelmatrix(MixedModel(formula, data))
        X = X[:, 2:end]
    else
        # @show typeof(formula)
        # @show typeof(StatsModels.apply_schema(formula, StatsModels.schema(data)))
        # @show StatsModels.apply_schema(formula, StatsModels.schema(data))
        X = StatsModels.modelmatrix(StatsModels.apply_schema(formula, StatsModels.schema(data)), data)
        # @show X
        if hasintercept(formula)
            X = X[:, 2:end]
        end
    end
    return X
end


function actionmodels_data_random_effects(formula::FormulaTerm, data::D) where {D}
    if !actionmodels_has_ranef(formula)
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
                Z["slope_" * slope] = get_var(term(slope), data)
            else
                for s in slope
                    Z["slope_" * s] = get_var(term(s), data)
                end
            end
        end
    else
        Z = nothing
    end
    return Z
end


function actionmodels_ranef(formula::FormulaTerm)
    if actionmodels_has_ranef(formula)
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
