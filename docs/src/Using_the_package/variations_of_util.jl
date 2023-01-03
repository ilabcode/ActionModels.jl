# # Variations of utility functions

# This section contains a list of convenient utility functions. Many of these are used throughout the documentation and you can find all here.


#  ## Content
#  ### - get_states()
#  ### - get_parameters()
#  ### - set_parameters()
#  ### - get_history()
#  ###s - get_posteriors() and reset()

# We will define an agent to use during demonstrations of the utility functions:
using ActionModels #hide

agent = premade_agent("premade_binary_rw_softmax")

# ### get_states()
# The get_states() function can give you a single state, multiple states and all states of an agent. 

# Let's start with all states
get_states(agent)

# Get a single state
get_states(agent, "transformed_value")

# Get multiple states
get_states(agent, ["transformed_value", "action"])


# ### get_parameters()

# get\_parameters() work just like get_states, but will give you the parameters of the agent:

# lets start with all parameters
get_parameters(agent)

# Get a single parameter
get_parameters(agent,("initial", "value"))

# Get multiple parameters
get_parameters(agent, [("initial", "value"), "learning_rate"])


# ### set_parameters()

# Setting a single parameter in an agent
set_parameters!(agent,("initial", "value"), 1 )

# Setting multiple parameters in an agent
set_parameters!(agent, Dict("learning_rate" => 3, "softmax_action_precision"=>0.5))

# See the parameters we have set uising get_params function
get_parameters(agent)


# ### get_history()

# To get the history we need to give inputs to the agent. Let's start by giving a single input

give_inputs!(agent, 1)

# We can now get the history of the agent's states. We can have a look at the "value" state.

get_history(agent, "value")

# Get multiple states' histories

get_history(agent, ["value","action"])

# Lastly, get all history of the agent

get_history(agent)


# ### get_posteriors()

# get\_posteirors() is a funcion for extracting parameters from a Turing chain. Let us set up a fitted model:

# Let us reset our agent and make it ready for new input
reset!(agent)

# Define a range of inputs
inputs = [1,0,0,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,0,1,0,1]

# Define actions
actions = give_inputs!(agent,inputs)

# Set a prior for the parameter we wish to fit
priors = Dict("softmax_action_precision" => Normal(1, 0.5), "learning_rate"=> Normal(1, 0.1))

# Fit the model
fitted_model = fit_model(agent, inputs, actions, priors)


# We can now use the get_posteriors() 

get_posteriors(fitted_model)
