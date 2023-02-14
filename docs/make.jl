using ActionModels
using Documenter
using Literate

#Remove old tutorial markdown files
for filename in readdir("docs/src/generated_markdown_files")
    rm("docs/src/generated_markdown_files/" * filename)
end

for filename in readdir("docs/src/julia_src_files")
    if endswith(filename, ".jl")
        Literate.markdown(
            "docs/src/julia_src_files/" * filename,
            "docs/src/generated_markdown_files",
            documenter = true,
        )
    end
end


#Set documenter metadata
DocMeta.setdocmeta!(ActionModels, :DocTestSetup, :(using ActionModels); recursive = true)

#Create documentation
makedocs(;
    modules = [ActionModels],
    authors = "Peter Thestrup Waade ptw@cas.au.dk, Anna Hedvig Møller Daugaard hedvig.2808@gmail.com, Jacopo Comoglio jacopo.comoglio@gmail.com, Christoph Mathys chmathys@cas.au.dk
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
            "generated_markdown_files/introduction.md",
            "generated_markdown_files/agent_and_actionmodel.md",
            "generated_markdown_files/fitting_vs_simulating.md",
        ]
        "Creating Your Model" => [
            "generated_markdown_files/creating_own_action_model.md",
            "generated_markdown_files/premade_agents_and_models.md",
        ]
        "Agent Based Simulation" => [
            "generated_markdown_files/simulation_with_an_agent.md",
            "generated_markdown_files/variations_of_util.md",
        ]
        "Fitting an Agent Model" => [
            "generated_markdown_files/fitting_an_agent_model_to_data.md",
            "generated_markdown_files/prior_predictive_sim.md",
        ]
        "Advanced Usage" => ["generated_markdown_files/custom_fit_model.md"]
    ],
)

deploydocs(;
    repo = "github.com/ilabcode/ActionModels.jl",
    devbranch = "main",
    push_preview = false,
)
