using ActionModels
using Plots
using StatsPlots
using Distributions

@testset "binary RW" begin
    
    agent = premade_agent("premade_rw_softmax")

    inputs = [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    inputs_2 = [missing, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]

    sim_actions = give_inputs!(agent, inputs)

    plot_trajectory(agent, "value", linetype = :path)
    plot_trajectory!(agent, "action")
    plot!(inputs_2, linetype = :scatter)

    param_priors = Dict(
        "learning_rate" => LogNormal(0,1),
        ("initial", "value") => Normal(0,1),
        "softmax_action_precision" => LogNormal(0,1),
    )

    plot_predictive_simulation(param_priors, agent, inputs, "value")

    fitted_model = fit_model(agent, inputs, sim_actions, param_priors)

    plot_parameter_distribution(fitted_model, param_priors)

end

