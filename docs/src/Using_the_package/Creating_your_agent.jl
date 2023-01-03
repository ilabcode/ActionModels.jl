
# # Creating your agent using the ActionModels package

# In this section we will demonstrate:
#    - Defining an agent with the premade_agent() function
#    - Get an overview of the parameters in a premade agent and get a specific parameter 
#    - Get an overview of the states in a premade agent and get a specific state
#    - Set one and more parameter values in an agent
#    - Defining an agent with custom parameter values

# ## Defining an agent with the premade_agent()


using ActionModels #hide

# you can define a premade agent using the premade_agent() function. The function call is the following:

premade_agent(model_name::String, config::Dict = Dict(); verbose::Bool = true)

# Model_name is the type of premade agent you wish to use. You can get a list of premade agents with the command:

premade_agent("help") 

# Lets create an agent. We will use the "premade\_binary\_rw\_softmax" agent with the "binary\_rw\_softmax" action model. You define a default premade agent with the syntax below:
agent = premade_agent("premade_binary_rw_softmax")

# In the Actionmodels.jl package, an agent struct consists of the action model name (which can be premade or custom), parameters and states.  
# The premade agents are initialized with a set of configurations for parameters, states and initial state parameters.


# ### Get an overview of the parameters and states in a premade agent 

# Let us have a look at the parameters in our predefined agent with the fucntion get_parameters

get_parameters(agent)

# If you wish to get one of the parameters, you can change the method of the function by specifying the parameter you wish to retrieve:

get_parameters(agent, "learning_rate")

# Now, let us have a look at the states of the agent with the get_states() function.

get_states(agent)

# the "value" state is initialized with an initial state parameter (this is the ("initial", "value") parameter seen in the get_parameters(agent) call). The other states are missing due to the fact, that the agent has not recieved any inputs. 
# You can get specific states by inputting the state you wish to retrieve. This will get more interesting once we give the agents inputs.

get_states(agent, "transformed_value")


# ### Set one and more parameter values in an agent

# As mentioned above, the premade agent is equipped with dedault parameter values. If you wish to change these values you can do it in different ways.
# We can change one of the parameters in our agent like below by specifying parameter and the value to set the parameter to.

set_parameters!(agent, "learning_rate", 1)

get_parameters(agent)

# If you wish to change multiple parameters in the agent, you can define a dict of parameters folowed by the value like this

set_parameters!(agent, Dict("learning_rate" => 0.79,    
                        "softmax_action_precision" => 0.60, 
                        ("initial", "value") => 1))
                        
# ### Defining an agent with custom parameter values

# If you know which parameter values you wish to use when defining your agent, you can specify them in the beginning as a dict() with parameter name as a string followed by the value.
agent_custom_parameters = premade_agent("premade_binary_rw_softmax", Dict("learning_rate" => 0.7, 
                                        "softmax_action_precision" => 0.8, 
                                        ("initial", "value") => 1)
)

#and we can retrieve the new parameters with the get_parameters() function

get_parameters(agent_custom_parameters)


# ## Creating your own agent

# If you wish to create your own custom agent, it is fairly simple and straight forward with the init_agent() function. 


# The elements to specify in init_agent() can be seen below. 
init_agent(
    action_model::Function;
    substruct::Any = nothing,
    parameters::Dict = Dict(),
    states::Union{Dict,Vector} = Dict(),
    settings::Dict = Dict(),
)

# The use of substructs and settings are optional, see advanced usage for more information on this.

# Let's start by defining the parameters for our custom agent. 
parameters = Dict(
    "Parameter_1" => 1,
    "Parameter_2" => 2,
    ("initial", "state_1") => 3)  

# The ("initial", "state\_1") and ("initial", "state\_2") parameters are initial state parameters. Depending on your action model, certain states need to be initialized with a starting value which are the initial state parameters.


# We can now define the action model to be used in the agent, which can be both premade or custom made.

action_model = your_chosen_actionmodel





# At this point we can define our states. These states are all set to missing, but the states that thave initial state parameters will be overwritten once we initialize our agent.

states = Dict(
    "state_1" => missing,
    "state_2" => missing,
    "state_3" => missing
)


# We can now input parameters, action model and states in the init_agent() function

custom_agent = init_agent(
    action_model,
    params = parameters,
    states = states)

# Let's have a look at the parameters and states in our agent. 
get_parameters(custom_agent)


#We can conclude that the agent is set corretly up, since we can see our configurations
get_states(custom_agent)




