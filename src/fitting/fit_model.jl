#####################################################################
### CONVENIENCE FUNCTION FOR DOING FULL MODEL FITTING IN ONE LINE ###
#####################################################################
function fit_model(
    model::DynamicPPL.Model;
    parallelization::Union{Nothing,AbstractMCMC.AbstractMCMCEnsemble} = nothing,
    sampler::Union{DynamicPPL.AbstractSampler,Turing.Inference.InferenceAlgorithm} = NUTS(
        -1,
        0.65;
        adtype = AutoReverseDiff(),
    ),
    n_iterations::Integer = 1000,
    n_chains = 1,
    show_sample_rejections::Bool = false,
    sampler_kwargs...,
)

    ## Set logger ##
    #If sample rejection warnings are to be shown
    if show_sample_rejections
        #Use a standard logger
        sampling_logger = Logging.SimpleLogger()
    else
        #Use a logger which ignores messages below error level
        sampling_logger = Logging.SimpleLogger(Logging.Error)
    end

    ## Fit model ##
    if !isnothing(parallelization)
        #With parallelization
        chains = Logging.with_logger(sampling_logger) do
            sample(model, sampler, parallelization, n_iterations, n_chains; sampler_kwargs...)
        end
    else
        chains = Logging.with_logger(sampling_logger) do
            sample(model, sampler, n_iterations, n_chains = n_chains; sampler_kwargs...)
        end
    end

    return FitModelResults(chains, model)
end

