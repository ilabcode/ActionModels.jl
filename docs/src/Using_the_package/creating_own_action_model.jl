
# ## Building your own action model

# ### In this section we will demonstrate the following

#   - A structure comparison between an agent and an action model
#   - Components and structure of an action model
#   - Tutorial on how to make the Rescorla-Wagner softmax action model
#   - Creating a custom agent to the action model with init_agent()

# An action model should always return an action distribution from which an aciton can be sampled.

# ### visual structure of agent and action model 

# ![Image1](./images/actionmodel_custom.png)


# let us start by the left side of the model. We have an agent with a set of defined parameters, states and a setting. 

# When initializing an action model you can start by reading in the agent's parameters. It can depend on the action model how many states you use during the update step. In this example we only use one state (which is the one that has an initial state parameter).

# During the update step we compute the remaining two states in the agent. State\_2 and state\_3 are outputs of the update step which are saved in the agent's history. In our example structure you can see that the updated\_state\_3 is the parameter in our action distribution.

# When building a custom action model you will have to (roughly) think of 3 things:

#   - What type of action do I want to simulate? 
#   - What does the update step constist of?
#   - What variables (parameters and states) are used in the update step and which do I wish to save in the agent history?

# ## Building the Rescorla-Wagner action model

# In this example we will construct an action model according to Rescorla-Wagner. First, let us recall the parameters and states needed for this action model. 

# We need two parameters: learning rate and softmax action precision. These parameters are specified when defining the agent. The main state for the update step is the "input value" state, and this is the only value we need to have in order to calculate an action.

# It can be nice to save other elements from the action model as states for analysis and plotting. We would in this case like to save the action probability and the transformed value in probability space. These two states don't influence the model in new trials, but are just extra output from the action model.

# Let us build our action model. The structure of the action model is equal to the strucutre in the figure.

# First, let us define the function name and which input it takes:


function custom_rescorla_wagner_softmax(agent::Agent, input) 

    learning_rate = agent.parameters["learning_rate"] 
    action_precision = agent.parameters["softmax_action_precision"] 
    
    old_value = agent.states["value"] 

    transformed_old_value = 1 / (1 + exp(-old_value))  

    new_value = old_value + learning_rate * (input - transformed_old_value) 

    action_probability = 1 / (1 + exp(-action_precision * new_value))
    
    action_distribution = Distributions.Bernoulli(action_probability)

    agent.states["value"] = new_value
    agent.states["transformed_value"] = 1 / (1 + exp(-new_value))
    agent.states["action_probability"] = action_probability
    
    push!(agent.history["value"], new_value)
    push!(agent.history["transformed_value"], 1 / (1 + exp(-new_value)))
    push!(agent.history["action_probability"], action_probability)

    return action_distribution
end


# We can define the agent now. Let us do it with the init_agent() function. 

#Set the parameters:

parameters = Dict(
    "learning_rate" => 1,
    "softmax_action_precision" => 1,
    ("initial", "value") => 0,)

# We set the initial state parameter for "value" state because we need a starting value in the update step. 

# Let us define the states in the agent:
states = Dict(
        "value" => missing,
        "transformed_value" => missing,
        "action_probability" => missing,
    )

# And lastly the action model:
action_model = custom_rescorla_wagner_softmax

# We can now initialize our agent with the action model, parameters and states.
agent = init_agent(
        action_model,
        parameters = parameters,
        states = states,
    )

# To see how to simulate actions by giving input to the agent see section "Simulation with an agent".



