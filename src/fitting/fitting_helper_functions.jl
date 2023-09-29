################################################################
####### Functions for removing level dependencies higher #######
################################################################
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


######################################################################
####### Function for extracting information necessary for a fit ######
######################################################################
function extract_structured_parameter_info(; priors::Dict, multilevel_group_cols::Vector)

    ## Create dictionary with the groups that each parameter depends on ##
    #Create dictionary where each parameter has all levels
    group_dependencies =
        Dict(zip(keys(priors), repeat([copy(multilevel_group_cols)], length(priors))))

    #Go through each specified prior
    for (parameter_key, info) in priors

        #For hierarchically dependent parameters 
        if info isa Multilevel

            #If the specified group is not in the group columns
            if info.group ∉ multilevel_group_cols
                throw(
                    ArgumentError(
                        "the parameter $parameter_key depends on the group $(info.group), but this group is not in the specified group columns",
                    ),
                )
            end

            #Recursively remove its group from all higher parameters 
            remove_higher_dependencies!(info, group_dependencies, priors, [])
        end
    end

    ## Create dictionary with all necessary information about parameters ##
    #Initialize empty dict
    parameters_info = Dict()

    #Create subdicts for each amount of group dependencies
    for n_dependencies = 0:length(multilevel_group_cols)
        parameters_info[n_dependencies] = Dict()
    end

    #Go through each prior
    for (parameter_key, info) in priors

        #Get out the group dependencies for the parameter
        parameter_group_dependencies = group_dependencies[parameter_key]

        #For multilevel dependent parameters
        if info isa Multilevel
            #Set the multilevel dependency to true
            multilevel_dependent = true

            #Extract the distribution and parameters
            distribution = info.distribution
            parameters = info.parameters

            #For normal parameters
        else
            #Set the multilevel dependency to false
            multilevel_dependent = false

            #Extract the distribution and parameters
            distribution = info
            parameters = []
        end

        #Save the information for using when fitting
        parameters_info[length(parameter_group_dependencies)][parameter_key] =
            ParameterInfo(
                name = parameter_key,
                group_dependencies = parameter_group_dependencies,
                multilevel_dependent = multilevel_dependent,
                distribution = distribution,
                parameters = parameters,
            )
    end

    return parameters_info
end



########################################################
####### Function for structuring data for fitting ######
########################################################
function extract_structured_data(;
    data::SubDataFrame,
    multilevel_group_cols::Vector,
    input_cols::Vector,
    action_cols::Vector,
    general_parameters_info::Dict,
)

    ## Extract data for the independent group ##
    #Create empty containers for actions and inputs grouped in multilevel structures
    inputs = Dict()
    actions = Dict()
    #And for storing the multilevel groups in this dataset
    multilevel_groups = []

    #Go through each multilevel group
    for multilevel_group_data in groupby(data, multilevel_group_cols)

        #Get out the group combination
        multilevel_group_key = Tuple(multilevel_group_data[1, multilevel_group_cols])

        #Store inputs and actions as Arrays
        inputs[Tuple(multilevel_group_key)] = Array(multilevel_group_data[:, input_cols])
        actions[Tuple(multilevel_group_key)] = Array(multilevel_group_data[:, action_cols])

        #Save the multilevel groups in the dataset
        push!(multilevel_groups, multilevel_group_key)
    end


    ## Get levels for individual multilevel groups for the independent group ##
    #Initialize empty dict
    multilevel_group_levels = Dict()

    #For each column in the group columns
    for multilevel_col in multilevel_group_cols
        #Save the unique levels from the column
        multilevel_group_levels[multilevel_col] = unique(data[:, multilevel_col])
    end

    ## Set group levels for each parameter ##
    #Initialize vectors for storing ordered parameter info
    agent_parameters_info = []
    multilevel_parameters_info = []

    #For each amount of group dependencies, starting from 0
    for n_dependencies = 0:length(multilevel_group_cols)

        #For each parameter with that many group dependencies
        for (parameter_key, parameter_info_template) in
            general_parameters_info[n_dependencies]

            #Make a copy of the parameter info for this independent group
            parameter_info = deepcopy(parameter_info_template)

            #If there are no group dependencies
            if n_dependencies == 0
                #There are no group levels
                parameter_info.group_levels = [()]

                #If there are group dependencies
            else
                #Get out all group combinations for the parameter with the given independent group
                parameter_info.group_levels = collect(
                    Iterators.product(
                        map(
                            key -> multilevel_group_levels[key],
                            parameter_info.group_dependencies,
                        )...,
                    ),
                )
            end

            #If the parameter has the maximum number of group dependencies
            if n_dependencies == length(multilevel_group_cols)
                #Save it as an agent parameter
                push!(agent_parameters_info, parameter_info)

                #Otherwise
            else
                #It is a multilevel group parameter
                push!(multilevel_parameters_info, parameter_info)
            end
        end
    end

    #Return the structures inputs and actions
    return (
        inputs = inputs,
        actions = actions,
        multilevel_groups = multilevel_groups,
        multilevel_parameters_info = multilevel_parameters_info,
        agent_parameters_info = agent_parameters_info,
    )
end



################################################################
####### Function for renaming parameters in Chains object ######
################################################################

function rename_chains(chains::Chains, independent_group_info::NamedTuple)

    #Initialize dict for replacement names
    replacement_names = Dict()

    ## Set replacement names for multilevel parameters ##
    #Go through each parameter
    for parameter_info in independent_group_info.multilevel_parameters_info

        #Go through each group with that parameter
        for group in parameter_info.group_levels

            #As default, group indicators are separated from parameter names with a space
            separator = " "

            #If the parameter name is a string
            if parameter_info.name isa String
                #Include the quotation marks
                parameter_string = "\"$(parameter_info.name)\""
            #Otherwise
            else 
                #Keep it as it is
                parameter_string = parameter_info.name
            end

            #If there are no group dependencies
            if isempty(group)
                #Don't print anything
                group_string = ""
                #And don't have a separator
                separator = ""

                #If there is only one group
            elseif length(group) == 1
                #Extract the group from the tuple it is in
                group_string = group[1]

                #If there are multiple group dependencies
            else
                #Print the whole group dependency tuple
                group_string = group
            end

            #Set a replacement name
            replacement_names["multilevel_parameters[$parameter_string][$group]"] = "$group_string$separator$(parameter_info.name)"
        end
    end

    ## Set replacement names for agent parameters ##
    #Go through each agent parameter
    for parameter_info in independent_group_info.agent_parameters_info

        #Go through each group with that parameter
        for group in parameter_info.group_levels

            #As default, group indicators are separated form parameter names with a space
            separator = " "

            #If the parameter name is a string
            if parameter_info.name isa String
                #Include the quotation marks
                parameter_string = "\"$(parameter_info.name)\""
            #Otherwise
            else 
                #Keep it as it is
                parameter_string = parameter_info.name
            end
            

            #If there are no group dependencies
            if isempty(group)
                #Don't print anything
                group_string = ""
                #And don't have a separator
                separator = ""

                #If there is only one group
            elseif length(group) == 1
                #Extract the group from the tuple it is in
                group_string = group[1]

                #If there are multiple group dependencies
            else
                #Print the whole group dependency tuple
                group_string = group

            end

            #Set a replacement name
            replacement_names["agent_parameters[$group][$parameter_string]"] = "$group_string$separator$(parameter_info.name)"

        end
    end

    #Input the dictionary to replace the names
    chains = replacenames(chains, replacement_names)

    return chains
end
