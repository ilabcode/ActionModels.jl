```@meta
EditURL = "<unknown>/src/Using_the_package/variations_of_util.jl"
```

# Variations of utility functions

This section contains a list of convenient utility functions. Many of these are used throughout the documentation and you can find all here.

 ## Content
  - [getting states of an agent with: get_states()](#Getting-States)
  - [getting parameters of an agent with: get_parameters()](#Getting-Parameters)
  - [setting parameters of an agent with: set_parameters()](#Setting-Parameters)
  - [getting history of an agent with: get_history()](#Getting-History)
  - [resetting an agent with reset!() and getting posteriors with get_posteriors()](#Getting-Posteriors)

We will define an agent to use during demonstrations of the utility functions:

````@example variations_of_util
using ActionModels #hide

agent = premade_agent("premade_binary_rw_softmax")
````

### Getting States
The get_states() function can give you a single state, multiple states and all states of an agent.

Let's start with all states

````@example variations_of_util
get_states(agent)
````

Get a single state

````@example variations_of_util
get_states(agent, "transformed_value")
````

Get multiple states

````@example variations_of_util
get_states(agent, ["transformed_value", "action"])
````

### Getting Parameters

get\_parameters() work just like get_states, but will give you the parameters of the agent:

lets start with all parameters

````@example variations_of_util
get_parameters(agent)
````

Get a single parameter

````@example variations_of_util
get_parameters(agent,("initial", "value"))
````

Get multiple parameters

````@example variations_of_util
get_parameters(agent, [("initial", "value"), "learning_rate"])
````

### Setting Parameters

Setting a single parameter in an agent

````@example variations_of_util
set_parameters!(agent,("initial", "value"), 1 )
````

Setting multiple parameters in an agent

````@example variations_of_util
set_parameters!(agent, Dict("learning_rate" => 3, "softmax_action_precision"=>0.5))
````

See the parameters we have set uising get_parameters function

````@example variations_of_util
get_parameters(agent)
````

### Getting History

To get the history we need to give inputs to the agent. Let's start by giving a single input

````@example variations_of_util
give_inputs!(agent, 1)
````

We can now get the history of the agent's states. We can have a look at the "value" state.

````@example variations_of_util
get_history(agent, "value")
````

Get multiple states' histories

````@example variations_of_util
get_history(agent, ["value","action"])
````

Lastly, get all history of the agent

````@example variations_of_util
get_history(agent)
````

### Getting Posteriors

get\_posteirors() is a funcion for extracting parameters from a Turing chain. Let us set up a fitted model:

Let us reset our agent and make it ready for new input

````@example variations_of_util
reset!(agent)
````

Define a range of inputs

````@example variations_of_util
inputs = [1,0,0,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,0,1,0,1]
````

Define actions

````@example variations_of_util
actions = give_inputs!(agent,inputs)
````

Set a prior for the parameter we wish to fit

````@example variations_of_util
using Distributions
priors = Dict("softmax_action_precision" => Normal(1, 0.5), "learning_rate"=> Normal(1, 0.1))
````

Fit the model

````@example variations_of_util
fitted_model = fit_model(agent, priors, inputs, actions)
````

We can now use the get_posteriors()

````@example variations_of_util
get_posteriors(fitted_model)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

