using Distributions
"""
Function for performing parameter recovery
"""
function parameter_recovery(
    parameter_ranges::Dict{K2, Vector{R}},
    input_sequences::Vector{V},
    priors::Union{P, Vector{P}},
    n_simulations::Int,
    ) where {
            K1 <: Any,
            K2 <: Any,
            D <: Distribution,
            P <: Dict{K1, D},
            R <: Real,
            V <: Any
            
    }


end
# parameter_ranges = Dict(
#     "GG" => collect(0:0.05:1),  
#     "OP" => collect(-10:0.5:-0.5)
# )
# parameter_ranges = Dict(
#     "GG" => collect(0:0.05:1),  
#     ("xprob", "volatility") => collect(-10:0.5:-0.5)
# )
parameter_ranges = Dict(
    ("xprob", "ZZ") => collect(0:0.05:1),  
    ("xprob", "volatility") => collect(-10:0.5:-0.5)
)

# input_sequence = [1,2,1]
input_sequence = [[1,2,1], [2,3,1]]

# priors = Dict(
#     "GG" => Normal(),  
#     "OP" => Normal()
# )
# priors = Dict(
#     "GG" => Normal(),  
#     ("xprob", "volatility") => Normal()
# )
# priors = Dict(
#     ("xprob", "ZZ") => Normal(),  
#     ("xprob", "volatility") => Normal()
# )
priors = [
    Dict(
        "GG" => Normal(),  
        "OP" => Normal()
    ),
    Dict(
        "GG" => Normal(),  
        "OP" => Normal()
    )
]


parameter_recovery(parameter_ranges, input_sequence, priors, 10)




function testfun3( 
    input_sequences::Union{Vector{T}, Vector{Vector{T}}}) where {
    K <: Union{String, Tuple},
    D <: Distribution,
    P <: Dict{K, D},
    T <: Real,
}
    print("WUWU")
end


typeof(input_sequence)

testfun3(input_sequence)

function testfun1(parameter_ranges::Dict{K, Vector{T}}) where {K <: Union{String, Tuple}, T <: Real}
    print("WUWU")
end

function testfun2( priors::Union{P, Vector{P}},) where 
    {
    K <: Union{String, Tuple},
    D <: Distribution,
    P <: Dict{K, D},
    T <: Real,
    }
    print("WUWU")
end

testfun2(priors)

typeof(priors)

testfun(parameter_ranges)




    using Turing, Distributions, LogExpFunctions, ReverseDiff
    using ActionModels, HierarchicalGaussianFiltering
    using Plots, StatsPlots
    using CSV, DataFrames, Glob, JLD2, Logging, ProgressMeter
    using Distributed
    
    include("load_datasets.jl")
    include("create_agent.jl")
    
    
    ## SETTINGS ##
    #Define a dictionary called parameter_ranges to store the ranges of different parameters
    parameter_ranges = Dict(
        "prior_posterior_weight" => collect(0:0.05:1),  # Range of values for the "prior_posterior_weight" parameter
        ("xprob", "volatility") => collect(-10:0.5:-0.5),  # Range of values for the ("xprob", "volatility") parameter
        ("xvol", "volatility") => collect(-10:0.5:-0.5),  # Range of values for the ("xvol", "volatility") parameter
        "action_precision" => collect(0.1:0.1:2).^(-1)  # Range of values for the "action_precision" parameter
    )
    
    parameter_ranges = Dict(
        "prior_posterior_weight" => collect(0:0.5:1),  # Range of values for the "prior_posterior_weight" parameter
        ("xprob", "volatility") => collect(-10:5:-0.5),  # Range of values for the ("xprob", "volatility") parameter
        ("xvol", "volatility") => collect(-10:5:-0.5),  # Range of values for the ("xvol", "volatility") parameter
        "action_precision" => collect(0.1:1:2).^(-1)  # Range of values for the "action_precision" parameter
    )
    
    #Set priors for inference
    priors = [Dict(
        "prior_posterior_weight" => LogitNormal(),  
        ("xprob", "volatility") => truncated(Normal(-5, 3), upper = -0.5),
        ("xvol", "volatility") => truncated(Normal(-5, 3), upper = -0.5),  
        "action_precision" => truncated(Normal(0.5, 0.5), lower = 0) 
    )]
    
    #The input sequences for these participants will be used for parameter recovery
    input_sequences_to_use = [101, 102]
    ## MAKE CHECK IF THESE ARE NOT IN DATA? ##
    
    #Number of simulations to run for each parameter combination, for each input sequence
    n_simulations = 10  
    n_simulations = 1
    
    #Settings to use for the sampling
    sampler_settings = (n_iterations = 10, n_chains = 1)
    
    ## SETUP ##
    #Create agent
    agent = create_agent("binary_3level")
    
    #Load data
    data = load_datasets(
            glob("*.csv", "data/"),
            "group_identifiers.csv",
            [:condition, :response, :modality, :ID, :group],
        )
    
        
    ## - prepare input sequences - ##
    # Make a vector of vectors to store the condition column separated by ID
    input_vectors = Vector{Vector}()
    
    # Iterate over each unique ID in the data
    for id in unique(data.ID)
       
        #For selected ID's
        if id in input_sequences_to_use
            # Get the condition values for the current ID
            conditions = data[data.ID .== id, :condition]
            
            # Append the condition values to the condition_vectors vector
            push!(input_vectors, conditions)
        end
    end
    
    ## - prepare parameter combinations - ##
    # Create two arrays, param_keys and param_values, to store the keys and values of the parameter_ranges dictionary
    param_keys = collect(keys(parameter_ranges))
    param_values = collect(values(parameter_ranges))
    
    # Generate all possible combinations of parameter values using the Cartesian product of param_values
    combinations = collect(Iterators.product(param_values...))
    
    # Create an empty vector called parameter_combinations to store dictionaries of parameter combinations
    parameter_combinations = Vector{Dict}()
    
    # Iterate over each combination of parameter values
    for combination in combinations
        # Create a dictionary called dict_from_vectors by pairing each key in param_keys with its corresponding value in combination
        dict_from_vectors = Dict(param_keys[i] => combination[i] for i in 1:length(param_keys))
        
        # Add the dictionary dict_from_vectors to the parameter_combinations vector
        push!(parameter_combinations, dict_from_vectors)
    end
    
    
    ## SIMULATION ##
    #Construct all combinations of parameters, priors, input sequences, one for each simulation
    recovery_infos = collect(Iterators.product(parameter_combinations, enumerate(priors), enumerate(input_vectors), 1:n_simulations))
    
    
    
    
    #Run simulations in parallel
    outcome = @showprogress pmap(recovery_info -> single_recovery(agent, sampler_settings, recovery_info...), recovery_infos)
    
    outcome = vcat(outcome...)
    
    
    
    
    function single_recovery(agent::Agent, sampler_settings::NamedTuple, parameters::Dict, prior_and_idx::Tuple, input_sequence_and_idx::Tuple, simulation_idx)
        #Extract the prior and input sequence
        prior_idx, prior = prior_and_idx
        input_sequence_idx, input_sequence = input_sequence_and_idx
        
        #Set the parameters for the agent
        set_parameters!(agent, parameters)
        reset!(agent)
    
        #Give inputs and get simulated actions
        simulated_actions = give_inputs!(agent, input_sequence)
        
        #Fit the model to the simulated data
        fitted_model = fit_model(agent, prior, input_sequence, simulated_actions; sampler_settings...)
    
        #Extract the posterior medians
        posterior_medians = get_posteriors(fitted_model)
    
        ## - Rename dictionaries - ##
        renamed_parameters = Dict()
        for key in keys(parameters)
            #Concatenate tuples
            if key isa Tuple
                new_key = join(key, "__")
            else 
                new_key = key
            end
    
            #Add prefix to show that it is an estimate
            new_key = "true__"*new_key
    
            #Save in the new dict
            renamed_parameters[new_key] = parameters[key]
        end
        
        renamed_posterior_medians = Dict()
        for key in keys(posterior_medians)
            #Concatenate tuples
            if key isa Tuple
                new_key = join(key, "__")
            else 
                new_key = key
            end
    
            #Add prefix to show that it is an estimate
            new_key = "estimated__"*new_key
    
            #Save in the new dict
            renamed_posterior_medians[new_key] = posterior_medians[key]
        end
        
        #Gather a dataframe row
        dataframe_row = hcat(
            DataFrame(renamed_parameters),
            DataFrame(renamed_posterior_medians),
            DataFrame(prior_idx = prior_idx, input_sequence_idx = input_sequence_idx, simulation_idx = simulation_idx),
            makeunique = true
        )
    end
    
    single_recovery(agent, sampler_settings, recovery_infos[1]...)
    
    
    
    
    #### - move into ActionModels
    
    #### - turn off showprogress for single_recovery
    
    #### - make into single function



end