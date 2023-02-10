using ActionModels
using Documenter
using Literate

#Remove old tutorial markdown files
for filename in readdir("docs/src/markdowns")
    rm("docs/src/markdowns/" * filename)
end
#Generate new tutorial markdown files
for filename in readdir("docs/src/Using_the_package")
    if endswith(filename, ".jl")
        Literate.markdown(
            "docs/src/Using_the_package/" * filename,
            "docs/src/Markdowns",
            documenter = true,
        )
    end
end


#Set documenter metadata
DocMeta.setdocmeta!(ActionModels, :DocTestSetup, :(using ActionModels); recursive = true)

#Create documentation
makedocs(;
    modules = [ActionModels],
    authors = "Peter Thestrup Waade ptw@cas.au.dk, Jacopo Comoglio jacopo.comoglio@gmail.com, Christoph Mathys chmathys@cas.au.dk
                  and contributors",
    repo = "https://github.com/ilabcode/ActionModels.jl/blob/{commit}{path}#{line}",
    sitename = "ActionModels.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ilabcode.github.io/ActionModels.jl",
        assets = String[],
    ),
    pages = [
        "Introduction to Action Models" => [
            "markdowns/Introduction.md",
            "Conceptual_introduction/agent_and_actionmodel.md",
            "Conceptual_introduction/fitting_vs_simulating.md",
        ]
        "Creating Your Model" => [
            "markdowns/creating_own_action_model.md",
            "markdowns/premade_agents_and_models.md",
        ]
        "Agent Based Simulation" =>
            ["markdowns/simulation_with_an_agent.md", "markdowns/variations_of_util.md"]
        "Fitting an Agent Model" => [
            "markdowns/fitting_an_agent_model_to_data.md",
            "markdowns/prior_predictive_sim.md",
        ]
        "Advanced Usage" => ["markdowns/custom_fit_model.md"]
    ],
)

deploydocs(; repo = "github.com/ilabcode/ActionModels.jl", devbranch = "origin/quick_tests")
