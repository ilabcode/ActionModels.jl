using ActionModels, Documenter, Literate

## Set paths ##
actionmodels_path = dirname(pathof(ActionModels))

juliafiles_path = actionmodels_path * "/docs/julia_files"
user_guides_path = juliafiles_path * "/user_guide"
tutorials_path = juliafiles_path * "/tutorials"

markdown_src_path = actionmodels_path * "/docs/src"
theory_path = markdown_src_path * "/theory"
generated_user_guide_path = markdown_src_path * "/generated/user_guide"
generated_tutorials_path = markdown_src_path * "/generated/tutorials"


#Remove old tutorial markdown files 
for filename in readdir(generated_user_guide_path)
    if endswith(filename, ".md")
        rm(generated_user_guide_path * "/" * filename)
    end
end
for filename in readdir(generated_tutorials_path)
    if endswith(filename, ".md")
        rm(generated_tutorials_path * "/" * filename)
    end
end
rm(markdown_src_path * "/" * "index.md")

#Generate index markdown file
Literate.markdown(juliafiles_path * "/" * "index.jl", markdown_src_path, documenter = true)

#Generate markdown files for user guide
for filename in readdir(user_guides_path)
    if endswith(filename, ".jl")
        Literate.markdown(
            user_guides_path * "/" * filename,
            generated_user_guide_path,
            documenter = true,
        )
    end
end

#Generate markdown files for tutorials
for filename in readdir(tutorials_path)
    if endswith(filename, ".jl")
        Literate.markdown(
            tutorials_path * "/" * filename,
            generated_tutorials_path,
            documenter = true,
        )
    end
end


#Set documenter metadata
DocMeta.setdocmeta!(ActionModels, :DocTestSetup, :(using ActionModels); recursive = true)

#Create documentation
makedocs(;
    modules = [ActionModels],
    authors = "Peter Thestrup Waade ptw@cas.au.dk, Malte Lau, Anna Hedvig Møller Daugaard hedvig.2808@gmail.com, Jacopo Comoglio jacopo.comoglio@gmail.com, Luke Ring, Christoph Mathys chmathys@cas.au.dk
                  and contributors",
    repo = "https://github.com/ilabcode/ActionModels.jl/blob/{commit}{path}#{line}",
    sitename = "ActionModels.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ilabcode.github.io/ActionModels.jl",
        assets = String[],
    ),
    pages = [
        # "Introduction to Action Models" => [
    #         "generated_markdown_files/introduction.md",
    #         "generated_markdown_files/agent_and_actionmodel.md",
    #         "generated_markdown_files/fitting_vs_simulating.md",
    #     ]
    #     "Creating Your Model" => [
    #         "generated_markdown_files/creating_own_action_model.md",
    #         "generated_markdown_files/premade_agents_and_models.md",
    #     ]
    #     "Agent Based Simulation" => [
    #         "generated_markdown_files/simulation_with_an_agent.md",
    #         "generated_markdown_files/variations_of_util.md",
    #     ]
    #     "Fitting an Agent Model" => [
    #         "generated_markdown_files/fitting_an_agent_model_to_data.md",
    #         "generated_markdown_files/prior_predictive_sim.md",
    #     ]
    #     "Advanced Usage" => ["generated_markdown_files/custom_fit_model.md"]
    # ],
)

deploydocs(;
    repo = "github.com/ilabcode/ActionModels.jl",
    devbranch = "main",
    push_preview = false,
)
