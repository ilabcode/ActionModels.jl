#Create global dictionary for storing functions that create premade agents
"""
"""
global const premade_agents = Dict{String, Function}()

"""
    function premade_agent(
        model_name::String, params_list::NamedTuple = (;)
    )

Function for making a premade agent.
"""
function premade_agent(model_name::String, params::Dict = Dict(); verbose::Bool = true)

    #Check that the specified model is in the list of keys
    if model_name in keys(premade_agents)
        #Create the specified model
        return premade_agents[model_name](params; verbose = verbose)

        #If the user asked for help
    elseif model_name == "help"
        #Return the list of keys
        print(keys(premade_agents))
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