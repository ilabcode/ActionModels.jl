### Function for extracting parameters from a Turing chain ###
"""
"""
function get_posteriors(chain::Chains; type::String = "median")

    #Get parameter symbols from the chain
    param_symbols = describe(chain)[2].nt.parameters

    #Create empty list for populating with parameter keys
    param_keys = []

    #Go through each parameter name
    for param_symbol in param_symbols

        #If there is a comma in the symbol, it is a composite parameter name
        if occursin(",", String(param_symbol))
            #So get the param key by parsing the string containing a tuple
            param_key = eval(Meta.parse(String(param_symbol)))
        else
            #Just convert it to a string
            param_key = String(param_symbol)
        end

        #Add it to the list
        push!(param_keys, param_key)
    end

    #Initialize tuple for storing parameter posteriors
    param_posteriors = Dict()

    #For each parameter key and name pair
    for (param_key, param_name) in zip(param_keys, param_symbols)

        #If the median parameter values has been asked for
        if type == "median"

            #Return the mdeian of the posterior distribution
            param_posteriors[param_key] = median(chain[:, param_name, :])

            #If full parameter posteriors distributions have been asked for
        elseif type == "distribution"

            #Return the full sampled distirbution
            param_posteriors[param_key] = chain[:, param_name, :]

        else
            throw(
                ArgumentError(
                    "argument 'type' has been misspecified. It should be either 'median' or 'distribution'",
                ),
            )
        end
    end

    return param_posteriors
end