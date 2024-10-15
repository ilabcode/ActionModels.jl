# ActionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ilabcode.github.io/ActionModels.jl)
[![Build Status](https://github.com/ilabcode/ActionModels.jl/actions/workflows/CI_full.yml/badge.svg?branch=main)](https://github.com/ilabcode/ActionModels.jl/actions/workflows/CI_full.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ilabcode/ActionModels.jl/branch/main/graph/badge.svg?token=NVFiiPydFA)](https://codecov.io/gh/ilabcode/ActionModels.jl)
[![License: GNU](https://img.shields.io/badge/License-GNU-yellow)](<https://www.gnu.org/licenses/>)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Welcome to the ActionModels.jl package!

ActionModels.jl is a powerfull and novel package for computational modelling of behavior and cognition. The package is developed with a intention to make computaitonal modelling intuitive, fast and easily adaptive to your experimental and simulation needs.

With ActionModels.jl you can define a fully customizable behavioral model and easily fit them to experimental data and used for computational modelling.

we provide a consice introduction to this framework for computational modelling of behvior and cognition and its accompanying terminology.

After this introduction, you will be presented with a detailed step-by-step guide on how to use ActionModels.jl to make your computational model runway-ready.

## Getting started

Defning a premade agent

````julia @example Introduction
using ActionModels
````

Find premade agent, and define agent with default parameters

````julia @example Introduction
premade_agent("help")

agent = premade_agent("premade_binary_rescorla_wagner_softmax")
````

Set inputs and give inputs to agent

````julia @example Introduction
inputs = [1,0,0,0,1,1,1,1,0,1,0,1,0,1,1]
actions = give_inputs!(agent,inputs)

using StatsPlots
plot_trajectory(agent, "action_probability")
````

Fit learning rate. Start by setting prior

````julia @example Introduction
using Distributions
priors = Dict("learning_rate" => Normal(0.5, 0.5))
````

Run model

````julia @example Introduction
chains = fit_model(agent, priors, inputs, actions, n_chains = 1, n_iterations = 10)
````

Plot prior and posterior

````julia @example Introduction
plot_parameter_distribution(chains,priors)
````

Get posteriors from chains

````julia @example Introduction
get_posteriors(chains)
````
