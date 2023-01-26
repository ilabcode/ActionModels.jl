
using ActionModels
action_model2 = ActionModels.premade_binary_rw_softmax

parameters = Dict(
    "learning_rate" => 2,
    "softmax_action_precision" => 3,
    "Test_parameter_1"=> 2,
    "Test_parameter_2"=> 2,
    ("initial","thingy")=>1,   
)

length(parameters)
keys(parameters)


#haskey(agent.initial_state_parameters, "thingy")

states = Dict(
        "alala" => missing,
        "value_probability" => missing,
        "action_probability" => missing,
        "thingy"=> missing
    )

#shared_parameters = Dict(("test","shared_param")=>(2,["learning_rate","softmax_action_precision"]))

#length(get_parameters(agent))

agent=init_agent(
    action_model2,
    parameters=parameters,
    states=states,
    #shared_parameters=shared_parameters,
)

get_parameters(agent)

set_parameters!(agent,("test","shared_param"),8)

get_parameters(agent,"softmax_action_precision")


get_parameters(agent,"Test_parameter_1")

