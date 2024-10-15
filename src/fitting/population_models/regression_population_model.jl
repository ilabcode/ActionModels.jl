
# DONE: random intercepts (hint: construct model matrix somehow instead of modelmatrix(MixedEffects(@formula)), which expects y
# DONE: expand to multiple formulas / flexible names
# - DONE: parameters inside different statistical models gets overridden by each other!
# DONE: finish the prepare_data function
# DONE: think about tuple parameter names (ie initial values or HGF params)
# DONE: random slopes
# DONE: more than one random intercept
# DONE: intercept-only model
# DONE better / custom priors
# DONE: (1.0) check integration of the new functionality
# DONE: Compare with old implementation of specifying statistical model
# DONE: check if we can get rid of TuringGLM
# DONE: prepare to merge
# DONE: merge
# DONE: support dropping intercepts (fixed and random)
# DONE: allow for varying priors: set up a regressionprior constructor
# DONE: Copy input data to avoid the above mutating the input
# DONE: Make sure the centering of the random slopes is good (γ, τ, σ)
# DONE: Clean tests up
# TODO: Adapt so it can split a parameter into a tuple (until we get rid of tuple parameter names for good)
# TODO: Add normalization of predictors
# TODO: (1.0) rename link_function to inv_link_function
# TODO: (1.0) models withut random effects: make sure there is an intercept
# TODO: (1.0) implement rename_chains for linear regressions (also for ordering of the random effects 
#             - MAYBE some code uses the data order, maybe some uses the formula order)
# TODO: (1.0) implement Regression type
# TODO: implement check_population_model
#       - TODO: Make check for whether there is a name collision with creating column with the parameter name
#       - TODO: check for whether the vector of priors is the correct amount
# TODO: (1.0) Example / usecase / tutorials)
#      - TODO: Fit a real dataset
# TODO: add to documentation that there shoulnd't be random slopes for the most specific level of grouping column (particularly when you only have one grouping column)

using ActionModels, Turing, Distributions
##########################################################################################################
## User-level function for creating a æinear regression statiscial mdoel and using it with ActionModels ##
##########################################################################################################
function create_model(
    agent::Agent,
    regression_formulas::Union{F,Vector{F}},
    data::DataFrame;
    priors::Union{R,Vector{R}} = RegressionPrior(),
    link_functions::Union{Function,Vector{Function}} = identity,
    input_cols::Vector{C},
    action_cols::Vector{C},
    grouping_cols::Vector{C},
    kwargs...,
) where {F<:MixedModels.FormulaTerm,C<:Union{String,Symbol}, R<:RegressionPrior}

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
    population_data = unique(data, grouping_cols)
    #Extract number of agents
    n_agents = nrow(population_data)

    ## Condition single regression models ##

    #Initialize vector of sinlge regression models
    regression_models = Vector{DynamicPPL.Model}(undef, length(regression_formulas))
    parameter_names = Vector{String}(undef, length(regression_formulas))

    #For each formula in the regression formulas, and its corresponding prior and link function
    for (model_idx, (formula, prior, link_function)) in
        enumerate(zip(regression_formulas, priors, link_functions))

        #Prepare the data for the regression model
        X, Z = prepare_regression_data(formula, population_data)

        if has_ranef(formula)
            #Extract each function term (random effect part of formula)
            ranef_groups =
                [term for term in formula.rhs if term isa MixedModels.FunctionTerm]
            #For each random effect, extract the number of categories there are in the dataset
            n_ranef_categories = [
                nrow(unique(population_data, Symbol(term.args[2]))) for term in ranef_groups
            ]

            #Set priors
            internal_prior = RegPrior(
                β = if prior.β isa Vector arraydist(prior.β) else filldist(prior.β, size(X, 2)) end,
                σ = if prior.σ isa Vector arraydist.(prior.σ) else [filldist(prior.σ, Int(size(Zⱼ, 2) / n_ranef_categories[ranefⱼ])) for (ranefⱼ, Zⱼ) in enumerate(Z)] end )
        else

            n_ranef_categories = nothing

            #Set priors, and no random effects
            internal_prior = RegPrior(
                β = if prior.β isa Vector arraydist(prior.β) else filldist(prior.β, size(X, 2)) end,
                σ = nothing)
        end

        #Condition the linear model
        regression_models[model_idx] = linear_model(
            X,
            Z,
            n_ranef_categories,
            link_function = link_function,
            prior = internal_prior,
        )

        #Store the parameter name from the formula
        parameter_names[model_idx] = string(formula.lhs)
    end

    #Create the combined regression statistical model
    population_model =
        regression_population_model(regression_models, parameter_names, n_agents)

    #Create a full model combining the agent model and the statistical model
    return create_model(
        agent,
        population_model,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        kwargs...,
    )
end


#############################################################
## Turing model to do linear regression for each parameter ##
#############################################################
@model function regression_population_model(
    linear_submodels::Vector{T},
    parameter_names::Vector,
    n_agents::Int,
) where {T<:DynamicPPL.Model}

    #Initialize vector of dicts with agent parameters
    agent_parameters = Dict{String,Real}[Dict{String,Real}() for _ = 1:n_agents]

    #For each parameter and its corresponding linear regression model
    for (linear_submodel, parameter_name) in zip(linear_submodels, parameter_names)
        #Run the linear regression model to extract parameters for each agent
        @submodel prefix = string(parameter_name) parameter_values = linear_submodel

        ## Map the output to the agent parameters ##
        #For each agent and the parameter value for the given parameter
        for (agent_idx, parameter_value) in enumerate(parameter_values)
            #Set it in the corresponding dictionary
            agent_parameters[agent_idx][parameter_name] = parameter_value
        end
    end

    return PopulationModelReturn(agent_parameters)
end


#########################################
## Linear model for a single parameter ##
#########################################
"""
Generalized linear regression models following the equation:
η = X⋅β
with optionally:
for each random effect: η += Zⱼ⋅rⱼ
link function: link(η)
"""
@model function linear_model(
    X::Matrix{R1}, # model matrix for fixed effects
    Z::Union{Nothing,Vector{MR}}, # vector of model matrices for each random effect
    n_ranef_categories::Union{Nothing,Vector{Int}}; # number of random effect parameters, per group
    link_function::Function,
    prior::RegPrior,
    size_r::Union{Nothing,Vector{Int}} = if isnothing(Z)
        nothing
    else
        size.(Z, 2)
    end, # number of random effect parameters, per group
    has_ranef::Bool = !isnothing(Z),
) where {R1<:Real,R2<:Real,MR<:Matrix{R2}}

    #Sample beta / effect size parameters (including intercept)
    β ~ prior.β

    #Do fixed effect linear regression
    η = X * β

    #If there are random effects
    if has_ranef

        #Initialize vector of random effect parameters
        σ = Vector{Vector{Real}}(undef, length(Z))
        r = Vector{Matrix{Real}}(undef, length(Z))

        #For each random effect group j, and its corresponding model matrix Zⱼ
        for (ranefⱼ, (Zⱼ, size_rⱼ)) in enumerate(zip(Z, size_r))

            #Calculate number of random effect parameters in the group
            n_ranef_params = Int(size_rⱼ / n_ranef_categories[ranefⱼ])

            #Sample the standard deviation of the random effect
            σ[ranefⱼ] ~ prior.σ[ranefⱼ]

            #Expand the standard deviation to the number of parameters
            r[ranefⱼ] ~ arraydist([
                Normal(0, σ[ranefⱼ][idx]) for idx = 1:n_ranef_params for
                _ = 1:n_ranef_categories[ranefⱼ]
            ])

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
    population_data::DataFrame,
)
    #Inset column with the name fo the agetn parameter, to avoid error from MixedModel
    insertcols!(population_data, Symbol(formula.lhs) => 1) #TODO: FIND SOMETHING LESS HACKY

    if ActionModels.has_ranef(formula)
        X = MixedModel(formula, population_data).feterm.x
        Z = Matrix.(MixedModel(formula, population_data).reterms)
    else
        X = StatsModels.modelmatrix(formula, population_data)
        Z = nothing
    end

    return (X, Z)
end


####################################################
## Check if there are random effects in a formula ##
####################################################
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




function rename_chains()
    #TODO

    # replacement_names = Dict()
    # for (param_name, _, __) in statistical_submodels
    #     for (idx, id) in enumerate(eachrow(population_data[!,grouping_cols]))
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


function check_population_model(
    ::Vector{DynamicPPL.Model},
    ::Vector{String},
    ::Int64;
    verbose::Bool,
    agent::Agent,
)
    #TODO
    return nothing
end
