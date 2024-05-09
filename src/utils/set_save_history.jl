#Function for changing the save_history setting
function set_save_history!(agent::Agent, save_history::Bool)

    #Change it in the agent
    agent.save_history = save_history

    #And in its substruct
    set_save_history!(agent.substruct, save_history)
end

#If there is an empty substruct, do nothing
function set_save_history!(substruct::Nothing, save_history::Bool)
    return nothing
end
