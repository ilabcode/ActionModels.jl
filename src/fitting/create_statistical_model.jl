
function statistical_model_turingglm(
    formula::TuringGLM.FormulaTerm,
    data;
    model::Type{T} = Distributions.Normal,
    priors::TuringGLM.Prior = TuringGLM.DefaultPrior()
) where {T<:UnivariateDistribution}
    return _statistical_model_turingglm(formula, data, T; priors)
end

function _statistical_model_turingglm(
    formula::TuringGLM.FormulaTerm,
    data,
    ::Type{T};
    priors::TuringGLM.Prior = TuringGLM.DefaultPrior()
) where {T<:UnivariateDistribution}


    # TODO: expand to multiple formulas
    # TODO: random intercepts (hint: construct model matrix somehow instead of modelmatrix(MixedEffects(@formula)), which expects y
    # TODO: better default priors / able to specify priors for all params

    # extract y, X and Z
    # y = data_response(formula, data)
    X = TuringGLM.data_fixed_effects(formula, data)
    # Z = TuringGLM.data_random_effects(formula, data)

    # μ and σ identities
    μ_X = 0
    σ_X = 1
    μ_y = 0
    σ_y = 1

    prior = _prior(priors, T)
    ranef = TuringGLM.ranef(formula)

    model = if ranef === nothing
        _model(μ_X, σ_X, prior, T)
    else
        intercept_ranef = TuringGLM.intercept_per_ranef(ranef)
        group_var = first(ranef).rhs
        idx = TuringGLM.get_idx(term(group_var), data)
        # print for the user the idx
        println("The idx are $(last(idx))\n")
        _model(μ_X, σ_X, prior, intercept_ranef, idx, T)
    end

    return (model, X)
end

# Models with Normal likelihood
function _model(μ_X, σ_X, prior, intercept_ranef, idx, ::Type{Normal})
    idxs = first(idx)
    n_gr = length(unique(first(idx)))
    @model function normal_model_ranef(
        X;
        predictors=size(X, 2),
        idxs=idxs,
        n_gr=n_gr,
        intercept_ranef=intercept_ranef,
        μ_X=μ_X,
        σ_X=σ_X,
        prior=prior,
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        σ ~ Exponential(10)
        if isempty(intercept_ranef)
            μ = α .+ X * β
        else
            τ ~ truncated(TDist(3); lower=0)
            zⱼ ~ filldist(Normal(), n_gr)
            μ = α .+ τ .* getindex.((zⱼ,), idxs) .+ X * β
        end
        #TODO: implement random-effects slope
        y ~ MvNormal(μ, σ^2 * I)
        return y
    end
end

function _model(μ_X, σ_X, prior, ::Type{Normal})
    @model function normal_model(
        X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=prior
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        σ ~ Exponential(10)
        y ~ MvNormal(α .+ X * β, σ^2 * I)
        return y
    end
end

function _prior(::DefaultPrior, ::Type{Normal})
    return CustomPrior(TDist(3), TDist(3), nothing)
end
