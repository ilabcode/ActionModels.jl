### Functions for removing level dependencies higher
function remove_higher_dependencies!(
    multilevel_struct::Multilevel,
    levels::Dict,
    priors::Dict,
    groups_list::Vector,
)

    #Add the group for the specific parameter to the list fo groups to remove
    push!(groups_list, multilevel_struct.group)
    new_groups_list = copy(groups_list)

    #Go through each of the higher-level parameters
    for parameter in multilevel_struct.parameters
        #Remove the group from their dependencies
        levels[parameter] = filter(x -> x ∉ groups_list, levels[parameter])

        #Repeat recursively for the higer-level parameter
        remove_higher_dependencies!(priors[parameter], levels, priors, new_groups_list)
    end
end
function remove_higher_dependencies!(
    multilevel_struct::Distribution,
    levels::Dict,
    priors::Dict,
    groups_list,
)
    return nothing
end
