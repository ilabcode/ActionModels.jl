function Base.show(io::IO, agent::Agent)

    ##Get information from agent struct
    action_model_name = string(agent.action_model)
    n_parameters = length(get_parameters(agent))
    n_states = length(get_states(agent))
    n_settings = length(agent.settings)
    n_observations = length(agent.history["action"]) - 1

    ##Print information
    #Basic info
    println("-- Agent struct --")
    println("Action model name: $action_model_name")

    #Substruct info
    if !isnothing(agent.substruct)
        substruct_type = string(typeof(agent.substruct))
        println("Substruct type: $substruct_type")
    end

    #parameters
    if n_parameters > 0
        println("Number of parameters: $n_parameters")
    end

    #States
    println("Number of states (including the action): $n_states")

    #Settings
    if n_settings > 0
        println("Number of settings: $n_settings")
    end

    #Number of observations
    if n_observations > 0
        println("This agent has received $n_observations inputs")
    end
end
