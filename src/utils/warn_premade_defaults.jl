"""
    warn_premade_defaults(defaults::Dict, config::Dict, prefix::String = "")

Checking each default value, and inform whether default value is used
"""

function warn_premade_defaults(defaults::Dict, config::Dict, prefix::String = "")

    #Go through each default value
    for (default_key, default_value) in defaults
        #If it not set by user
        if !(default_key in keys(config))
            #Warn them that the default is used
            @warn "$prefix $default_key was not set by the user. Using the default: $default_value"
        end
    end

    #Go trough each specified setting
    for (config_key, config_value) in config
        #If the user has set an invalid spec
        if !(config_key in keys(defaults))
            #Warn them
            @warn "$prefix a key $config_key was set by the user. This is not valid for this agent, and is discarded"
        end
    end
end