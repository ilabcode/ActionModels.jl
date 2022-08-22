function warn_premade_defaults(defaults::Dict, specs::Dict)

    #Go through each default value
    for (default_key, default_value) in defaults
        #If it not set by user
        if !(default_key in keys(specs))
            #Warn them that the default is used
            @warn "$default_key was not set by the user. Using the default: $default_value"
        end
    end

    #Go trough each specified setting
    for (specs_key, specs_value) in specs
        #If the user has set an invalid spec
        if !(specs_key in keys(defaults))
            #Warn them
            @warn "a key $specs_key was set by the user. This is not valid for this agent, and is discarded"
        end
    end
end