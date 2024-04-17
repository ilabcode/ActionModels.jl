
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


# DONE: random intercepts (hint: construct model matrix somehow instead of modelmatrix(MixedEffects(@formula)), which expects y
    # TODO: expand to multiple formulas / flexible names
    # - TODO: parameters inside different statistical models gets overridden by each other!
    # TODO: make LogitNormal save the param on the right scale
    # TODO: think about tuple parameter names (ie initial values or HGF params)
    # TODO: more functionality than turingGLM:
    # - random slopes
    # - intercept-only model
    # - better / custom priors
    # TODO: check integration of the new functionality
    # TODO: check if we can go back to turingglm._statistical_model_turingglm() with a few changes

    # extract y, X and Z
    # y = data_response(formula, data)
    #
    X = TuringGLM.data_fixed_effects(formula, data)
    if TuringGLM.has_ranef(formula)
        Z = TuringGLM.data_random_effects(formula, data)
    end


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
        idx = TuringGLM.get_idx(TuringGLM.term(group_var), data)
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
        β ~ TuringGLM.filldist(prior.predictors, predictors)
        σ ~ Exponential(10)
        if isempty(intercept_ranef)
            μ = α .+ X * β
        else
            τ ~ truncated(TDist(3); lower=0)
            zⱼ ~ filldist(Normal(), n_gr)
            μ = α .+ τ .* getindex.((zⱼ,), idxs) .+ X * β
        end
        #TODO: implement random-effects slope
        agent_param ~ MvNormal(μ, σ^2 * I)
        return agent_param
    end
end

# fixed-effects model with Normal likelihood
function _model(μ_X, σ_X, prior, ::Type{Normal})
    @model function normal_model(
        X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=prior
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        σ ~ Exponential(10)
        agent_param ~ MvNormal(α .+ X * β, σ^2 * I)
        return agent_param
    end
end

# fixed-effects model with LogNormal likelihood (log link)
function _model(μ_X, σ_X, prior, ::Type{LogNormal})
    @model function normal_model(
        X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=prior
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        σ ~ Exponential(10)
        agent_param ~ MvLogNormal(α .+ X * β, σ^2 * I)
        return agent_param
    end
end

# random-intercept model with LogNormal likelihood
function _model(μ_X, σ_X, prior, intercept_ranef, idx, ::Type{LogNormal})
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
        β ~ TuringGLM.filldist(prior.predictors, predictors)
        σ ~ Exponential(10)
        if isempty(intercept_ranef)
            μ = α .+ X * β
        else
            τ ~ truncated(TDist(3); lower=0)
            zⱼ ~ filldist(Normal(), n_gr)
            μ = α .+ τ .* getindex.((zⱼ,), idxs) .+ X * β
        end
        #TODO: implement random-effects slope
        agent_param ~ MvLogNormal(μ, σ^2 * I)
        return agent_param
    end
end

# fixed-effects model with LogitNormal likelihood (log link)
function _model(μ_X, σ_X, prior, ::Type{LogitNormal})
    @model function normal_model(
        X; predictors=size(X, 2), μ_X=μ_X, σ_X=σ_X, prior=prior
    )
        α ~ prior.intercept
        β ~ filldist(prior.predictors, predictors)
        σ ~ Exponential(10)
        agent_param ~ MvNormal(α .+ X * β, σ^2 * I)
        return logistic.(agent_param)
    end
end

# random-intercept model with LogitNormal likelihood
function _model(μ_X, σ_X, prior, intercept_ranef, idx, ::Type{LogitNormal})
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
        β ~ TuringGLM.filldist(prior.predictors, predictors)
        σ ~ Exponential(10)
        if isempty(intercept_ranef)
            μ = α .+ X * β
        else
            τ ~ truncated(TDist(3); lower=0)
            zⱼ ~ filldist(Normal(), n_gr)
            μ = α .+ τ .* getindex.((zⱼ,), idxs) .+ X * β
        end
        #TODO: implement random-effects slope
        agent_param ~ MvNormal(μ, σ^2 * I)
        return logistic.(agent_param)
    end
end



function _prior(::DefaultPrior, ::Type{Normal})
    return CustomPrior(TDist(3), TDist(3), nothing)
end

function _prior(::DefaultPrior, ::Type{LogNormal})
    return CustomPrior(TDist(3), TDist(3), nothing)
end

function _prior(::DefaultPrior, ::Type{LogitNormal})
    return CustomPrior(TDist(3), TDist(3), nothing)
end
