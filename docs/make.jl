using ActionModels
using Documenter
using Literate

# #Remove old tutorial markdown files
# for filename in readdir("src/tutorials")
#     rm("src/tutorials/" * filename)
# end
#Generate new tutorial markdown files
# for filename in readdir("tutorials")
#     if endswith(filename, ".jl")
#         Literate.markdown("tutorials/" * filename, "src/tutorials", documenter = true)
#     end
# end

#Set documenter metadata
DocMeta.setdocmeta!(ActionModels, :DocTestSetup, :(using ActionModels); recursive = true)

#Create documentation
makedocs(;
    modules = [ActionModels],
    authors = "Peter Thestrup Waade ptw@cas.au.dk, Jacopo Comoglio jacopo.comoglio@gmail.com, Christoph Mathys chmathys@cas.au.dk
                  and contributors",
    repo = "https://github.com/ilabcode/ActionModels.jl/blob/{commit}{path}#{line}",
    sitename = "HGF.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ilabcode.github.io/ActionModels.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/ilabcode/ActionModels.jl", devbranch = "dev")
