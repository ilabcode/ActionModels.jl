using HDF5
using MCMCChains
using MCMCChainsStorage

struct ChainSaveResume
    save_every::Int
    path::String
    plot_progress::Bool
    chain_prefix::String
end


ChainSaveResume() = ChainSaveResume(100, "./.samplingstate", false, "ActionModels_chain_link")



function prepare_sampler(
    sampler::Union{DynamicPPL.AbstractSampler,Turing.Inference.InferenceAlgorithm},
    chains::Chains,
)
    @error "Save and continue for sampler type $(typeof(sampler)) not implemented"
end

# prepare sampler for nuts
function prepare_sampler(
    sampler::NUTS,
    chains::Chains,
)
    # get the last step size
    sampler_ϵ = chains[:step_size][end]
    # create a new sampler with the last state
    # n_adapts can be zero because we load the warmed-up sampler state
    return NUTS(;
        n_adapts = 0,
        δ = sampler.δ,
        Δ_max = sampler.Δ_max,
        adtype = sampler.adtype,
        init_ϵ = sampler_ϵ,
        max_depth = sampler.max_depth,
    )
end

function validate_saved_sampling_state!(
    save_resume::ChainSaveResume,
    n_segments::Int,
    n_chains::Int,
)
    # check if the path exists
    if !isdir(save_resume.path)
        @warn "Path $(save_resume.path) does not exist, creating it"
        mkdir(save_resume.path)
    end
    # check if the path is a directory
    if !isdir(save_resume.path)
        @error "Path $(save_resume.path) is not a directory"
    end
    # check if the path is writable
    if !iswritable(save_resume.path)
        @error "Path $(save_resume.path) is not writable"
    end

    # find the last segment (for each chain)
    last_segment = Int[]
    for chain in 1:n_chains
        last_seg = 0
        n_segs = 0
        for cur_seg in 1:n_segments
            if isfile(joinpath(save_resume.path, "$(save_resume.chain_prefix)_c$(chain)_s$(cur_seg).h5"))
                last_seg = cur_seg
                n_segs += 1
            end
        end
        if n_segs < last_seg
            @error "Chain $chain has missing segments, check the path $(save_resume.path)"
        end
        push!(last_segment, last_seg)
    end

    return last_segment
end

function load_segment(
    save_resume::ChainSaveResume,
    chain::Int,
    segment::Int,
)
    # load the chain
    chain = h5open(joinpath(save_resume.path, "$(save_resume.chain_prefix)_c$(chain)_s$(segment).h5"), "r") do file
        read(file, Chains)
    end
    # extra validation?
    return chain
end

function save_segment(
    chain::Chains,
    save_resume::ChainSaveResume,
    segment::Int,
)
    seg_length = size(chain, 1)
    seg_start = (segment-1) * save_resume.save_every + 1
    seg_end = seg_start - 1 + seg_length

    # update the chain range
    chain = setrange(chain, seg_start:seg_end)

    # save the chain
    h5open(joinpath(save_resume.path, "$(save_resume.chain_prefix)_c$(chain.chain_id)_s$(segment).h5"), "w") do file
        write(file, "chain", chain)
    end
end

function combine_segments(
    save_resume::ChainSaveResume,
    n_segments::Int,
    n_chains::Int,
)
    chains = Chains[]
    for chain in 1:n_chains
        segments = Chains[]
        for segment in a:n_segments
            segments[segment] = load_segment(save_resume, chain, segment)
        end
        chains[chain] = cat(segments..., dims=1)
    end

    return chainscat(segments...)
end


#####################################################################
### CONVENIENCE FUNCTION FOR DOING FULL MODEL FITTING IN ONE LINE ###
#####################################################################
function fit_model(
    model::DynamicPPL.Model;
    parallelization::Union{Nothing,AbstractMCMC.AbstractMCMCEnsemble}=nothing,
    sampler::Union{DynamicPPL.AbstractSampler,Turing.Inference.InferenceAlgorithm}=NUTS(;
        adtype=AutoReverseDiff(compile=true),
    ),
    n_iterations::Integer=1000,
    n_chains=1,
    show_sample_rejections::Bool=false,
    show_progress::Bool=true,
    result_save_path::Union{Nothing,String}=nothing,
    save_resume::Union{ChainSaveResume, Nothing}=nothing,
    sampler_kwargs...,
)

    # ## Set logger ##
    # #If sample rejection warnings are to be shown
    # if show_sample_rejections
    #     #Use a standard logger
    #     sampling_logger = Logging.SimpleLogger()
    # else
    #     #Use a logger which ignores messages below error level
    #     sampling_logger = Logging.SimpleLogger(Logging.Error)
    # end

    if save_resume isa ChainSaveResume && save_resume.save_every < n_iterations
        final_segment = n_iterations % save_resume.save_every
        n_segments = floor(n_iterations / save_resume.save_every) + Int(final_segment > 0)
        resume_from::Array[Union{Nothing,Chains}] = fill(nothing, n_chains)
        samplers = fill(sampler, n_chains)
        last_complete_segment = validate_saved_sampling_state!(save_resume, n_segments, n_chains)
        for chain in 1:n_chains
            if last_complete_segment[chain] > 0
                resume_from[chain] = load_segment(save_resume, chain, last_complete_segment[chain])
                samplers[chain] = prepare_sampler(sampler, resume_from[chain])
            end
            for cur_seg in start_seg:n_segments
                # use the save_every value unless there are some iterations left over
                n_iter = final_segment > 0 && cur_seg == n_segments ? final_segment : save_resume.save_every
                # Run the sampler
                chain = sample(
                    model,
                    samplers[chain],
                    n_iter;
                    nchains=1,
                    progress=false,
                    resume_from = resume_from[chain],
                    sampler_kwargs...,
                )
                samplers[chain] = prepare_sampler(sampler, chain)
                resume_from[chain] = chain
                # Save the chain
                save_segment(chain, save_resume, cur_seg)
            end
        end
        chains = combine_segments(save_resume, n_segments, n_chains)
    else
        chains = sample(
            model,
            sampler,
            n_iterations;
            nchains=n_chains,
            progress=show_progress,
            sampler_kwargs...,
        )
    end

    # ## Fit model ##
    # if !isnothing(parallelization) && n_chains > 1
    #     #With parallelization
    #     chains = Logging.with_logger(sampling_logger) do
    #         # see if it's a threads or distributed ensemble
    #         if parallelization isa AbstractMCMC.AbstractMCMCThreads
    #             sample(model, sampler, n_iterations, n_chains; progress=show_progress, sampler_kwargs...)
    #         else
    #             sample(model, sampler, n_iterations, n_chains=n_chains; progress=show_progress, sampler_kwargs...)
    #         end
    #     end
    # elseif save_resume.save_every < 1
    #     chains = Logging.with_logger(sampling_logger) do
    #         sample(model, sampler, n_iterations, n_chains=n_chains; progress=show_progress, sampler_kwargs...)
    #     end
    # end

    return FitModelResults(model, nothing, chains)
end


####################################################################
### FULL CONVENIENCE FUNCTION THAT CAN ADD ADDITIONAL COMPONENTS ###
####################################################################
# function fit_model(
#     agent::Agent,
#     population_model::Union{M,P},
#     data::DataFrame;
#     parallelization::Union{Nothing,AbstractMCMC.AbstractMCMCEnsemble} = nothing,
#     extract_quantities::Bool = true,
#     sampler_kwargs...,
# ) where {M<:DynamicPPL.Model,T<:Union{String,Tuple,Any},D<:Distribution,P<:Dict{T,D}}

#     #Create a full model combining the agent model and the statistical model
#     model = create_model(agent, population_model, data)

#     #Fit the model
#     results = fit_model(model; parallelization = parallelization, sampler_kwargs...)

#     #Add tracked model
#     results.tracked_model =
#         create_model(agent, population_model, data, track_states = true)

#     #Extract tracked states
#     results.agent_parameters, results.agent_states, results.statistical_values =
#         extract_quantities(results.chains, results.tracked_model)
# end
