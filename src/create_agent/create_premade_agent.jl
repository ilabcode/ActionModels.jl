"""
    function premade_agent(
        model_name::String, params_list::NamedTuple = (;)
    )

Function for making a premade agent.
"""
function premade_agent(model_name::String, params::Dict = Dict(); verbose::Bool = true)

    #A list of all the included premade models
    premade_models = Dict(
        "hgf_gaussian_action" => "premade_hgf_gaussian",                           #A gaussian action based on an hgf
        "hgf_binary_softmax_action" => "premade_hgf_binary_softmax",               #A binary softmax action based on an hgf
        "hgf_unit_square_sigmoid_action" => "premade_hgf_unit_square_sigmoid",     #A binary unit square sigmoid action based on an hgf
    )

    #Check that the specified model is in the list of keys
    if model_name in keys(premade_models)
        #Create the specified model
        return premade_models[model_name](params; verbose = verbose)

        #If the user asked for help
    elseif model_name == "help"
        #Return the list of keys
        print(keys(premade_models))
        return nothing

        #If the model was misspecified
    else
        #Raise an error
        throw(
            ArgumentError(
                "the specified string does not match any model. Type premade_agent('help') to see a list of valid input strings",
            ),
        )
    end
end