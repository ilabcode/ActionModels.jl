function Base.show(io::IO, agent::AgentStruct)

    ##Get information from agent struct
    action_model_name = string(agent.action_model)
    n_params = length(agent.params)
    n_states = length(agent.states)
    n_settings = length(agent.settings)
    n_observations = length(agent.history["action"])

    ##Print information
    #Basic info
    println("-- Agent struct --")
    println("Action model name: $action_model_name")

    #Substruct info
    if agent.substruct != nothing
        substruct_type = string(typeof(agent.substruct))
        println("Substruct type: $substruct_type")
    end

    #Params
    if n_params > 0
        println("Number of parameters: $n_params")
    end

    #States
    println("Number of states (including the action): $n_states")

    #Settings
    if n_settings > 0
        println("Number of settings: $n_settings")
    end

    #Number of observations
    println("This agent has received $n_observations inputs")
end
