using ActionModels
using Documenter
using Literate

 #Remove old tutorial markdown files
 for filename in readdir("src/markdowns")
     rm("src/markdowns/" * filename)
 end
#Generate new tutorial markdown files
 for filename in readdir("src/Using_the_package")
     if endswith(filename, ".jl")
         Literate.markdown("./src/Using_the_package/" * filename, "src/Markdowns", documenter = true)
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
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/ilabcode/ActionModels.jl", devbranch = "dev")
