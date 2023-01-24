#Create global dictionary for storing functions that create premade agents
"""
    premade_agents::Dict{String,Function}

Global dictionary constant that contains premade agents. This is updated by other packages that add new premade agents.
"""
const global premade_agents = Dict{String,Function}()

"""
    premade_agent(model_name::String, config::Dict = Dict(); verbose::Bool = true)

Create an agent from the list of premade agents.

# Arguments
 - 'model_name::String': Name of the premade model. Returns a list of possible model names if set to 'help'. 
 - 'config::Dict = Dict()': A dictionary with configurations for the agent, like parameters and settings.
 - 'verbose::Bool = true': If set to false, warnings are hidden.
"""
function premade_agent(model_name::String, config::Dict = Dict(); verbose::Bool = true)

    #Check that the specified model is in the list of keys
    if model_name in keys(premade_agents)

        #If warnings are not hidden
        if verbose
            #Create the specified model
            agent = premade_agents[model_name](config)

        else
            #Create a logger which ignores messages below error level
            silent_logger = Logging.SimpleLogger(Logging.Error)
            #Use that logger
            agent = Logging.with_logger(silent_logger) do
                #Create the specified model
                premade_agents[model_name](config)
            end
        end

        #Return the agent
        return agent

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
