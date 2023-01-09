# # Simulating actions with an agent

# We will in this section introduce action simulation with the Actionmodels.jl package. 

# ### contents of this section 
#   - [The give_inputs() function](#Giving-Inputs-To-Agent)
#   - [Giving a single input to the agent and retrieving history](#Give-a-single-input)
#   - [Resetting the agent to initial state values](#Reset-Agent)
#   - [Give_inputs() with a vector of inputs](#Multiple-Inputs)
#   - [Plot State Trajectories](#Plotting-Trajectories-of-states)

# ### Giving Inputs To Agent

# With the ActionModels package you can, once you have defined your agent, use the function give_inputs to simulate actions.

#give_inputs(agent::Agent, inputs::Real) 

# When you give inputs to the agent, it produces actions according to the action model it is defined with.

# As can be seen in the figure below, when we know all parameter values, states and the inputs we can simulate actions

# ![Image1](Using_the_package/images/fitting_vs_simulation.png)

# The type of inputs you can give to your agent depends on the agent and the action it generates depends on the corresponding action model.

# Let us define our agent and use the dedault parameter configurations
using ActionModels 

agent = premade_agent("premade_binary_rw_softmax")

# ### Give a single input
# we can now give the agent a single input with the give_inputs!() function. The inputs for the Rescorla-Wagner agent are binary, so we input the value 1. 
give_inputs!(agent, 1)

# The agent returns either "false" or "true" which translates to an action of either 0 or 1. Given that we have evolved the agent with one input the states of the agent are updated. Let's see how we recover the history and the states of the agent after one run on input. We can do this with the get_history() function. With get_history we can get one or more target states or get the history of all states.

# Let us have a look at the history from all states:

get_history(agent)

# You can see in the "value" state contains two numbers. The first number is the initial state parameter which is set in the agent's configurations (see "Creating your agent" for more on the parameters and states). The second value in the "value" state is updated by the input.
# The three other states are initialized with "missing" and evolve as we give it inputs. The states in the agent are updated according to which computations the action model does with the input. For more information on the Rescorla-Wagner action model, check out the [LINK TO CHAPTER]

# ### Reset Agent
# We would like to reset the agent to its default values with the reset!() function:

reset!(agent)

# As you can see below, we have cleared the history of the agent.

get_history(agent)

# ### Multiple Inputs
# We will now define a sequence of inputs to the agent. 

inputs = [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]

# Let's give the inputs to the agent, and see which type of actions it will generate based on the inputs

actions = give_inputs!(agent, inputs)

# We can in the same manner get the history of the agent's states. We will have a look at the action state:
get_history(agent,"action_probability")


# ### Plotting Trajectories of states

# we can visualize the different types of states using the function:

#plot_trajectory(agent::Agent, target_state::Union{String,Tuple}; kwargs...)

# The default title when using plot_trajectory() is "state trajectory". This can be changed by adding a title-call as below. We can plot the actions and the action probability of the agent in two seperate plots:
using Plots 
using StatsPlots

plot_trajectory(agent,"action", title = "actions")

# We can change the state to plotting the action_probability

plot_trajectory(agent,"action_probability", title="acton probability")

# We can add a new linetype for the plot:

plot_trajectory(agent,"action_probability", title="acton probability",linetype = :path)

# We can layer the plots by adding "!" at the function call. We can add the actions plot to prior action probability plot:

plot_trajectory!(agent,"action", title = "action probability and action")


# ## If your agent computes more actions 

# If you wish to set up an agent who produces multiple actions, e.g. reaction time and a choice,
# you can use the "multiple_actions()" function. When setting up this type of agent, you define the different action models you want to use for each one of the wanted actions. 
# Currently in the ActionModels.jl package we have not yet predefined actionmodels for different actions. For multiple actions you should define your own action models (see the advanced usage for how to do this)

# If we were to define an agent with multiple action models, let's for this example say action_model_1 and action_model_2. The agent would be defined as:

# ---- EXAMPLE WITH INIT_AGENT WITH MULTIPLE ACTIONS ------

