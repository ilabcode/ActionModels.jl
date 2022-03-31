"""
    function init_HGF(
        default_params,
        input_nodes,
        state_nodes,
        child_parent_relations,
        update_order = false,
    )

Function for initializing the structure of an HGF model.
"""
function init_HGF(
    default_params,
    input_nodes,
    state_nodes,
    child_parent_relations,
    update_order = false,
)
    ### Checks ###
    # Throw warning if not all parameters and starting states
    # have been specified in the default_params

    # Check that all input nodes have at least one value parent
    # Check that no input nodes have more than one value parent (TEMPORARY)

    ### Initialize nodes ###
    #Initialize empty dictionary for storing nodes
    nodes_dict = Dict()

    ## Input nodes
    #For each specified input node
    for node_info in input_nodes

        #Initialize it, passing global params and specific params
        node = InputNode(
            name = node_info.name,
            params = InputNodeParams(; default_params.params..., node_info.params...),
            state = InputNodeState(),
        )

        #Add it to the dictionary
        nodes_dict[node.name] = node
    end

    ## State nodes
    #For each specified state node
    for node_info in state_nodes

        #Initialize it, passing global params and specific params
        node = StateNode(
            name = node_info.name,
            params = NodeParams(; default_params.params..., node_info.params...),
            state = NodeState(;
                default_params.starting_state...,
                node_info.starting_state...,
            ),
        )

        #Add it to the dictionary
        nodes_dict[node.name] = node
    end

    ### Set up child-parent relations ###
    #For each child
    for relationship_set in child_parent_relations

        #Find corresponding child node
        child_node = nodes_dict[relationship_set.child_node]

        #For each value parent
        for parent_info in relationship_set.value_parents

            #Check if it is a Tuple or a strind and find corresponding parent node 
            if typeof(parent_info) == String
                parent = nodes_dict[parent_info]
            else
                parent = nodes_dict[parent_info[1]]
            end

            #Add the parent to the child node
            push!(child_node.value_parents, parent)

            #Add the child node to the parent node
            push!(parent.value_children, child_node)

            #Add coupling strength to child node
            if typeof(parent_info) == String
                child_node.params.value_coupling[parent_info] = 1
            else
                child_node.params.value_coupling[parent_info[1]] = parent_info[2]
            end
        end

        #For each volatility parent
        for parent_info in relationship_set.volatility_parents

            #Check if it is a Tuple or a strind and find corresponding parent node 
            if typeof(parent_info) == String
                parent = nodes_dict[parent_info]
            else
                parent = nodes_dict[parent_info[1]]
            end
            #Add the parent to the child node
            push!(child_node.volatility_parents, parent)

            #Add the child node to the parent node
            push!(parent.volatility_children, child_node)

            #Add coupling strengths
            if typeof(parent_info) == String
                child_node.params.volatility_coupling[parent_info] = 1
            else
                child_node.params.volatility_coupling[parent_info[1]] = parent_info[2]
            end
        end
    end

    ### Update order ###
    ## Determine Update order
    #If update order has not been specified
    if .!update_order
        #Initialize empty vector for storing the update order
        update_order = []
        #For each state node, in the order inputted
        for node_info in state_nodes
            #Add the node name to the vector
            push!(update_order, nodes_dict[node_info.name])
        end
    end

    ## Order input nodes
    #Initialize empty vector for storing properly ordered input nodes
    ordered_input_nodes = []

    #For each specified input node, in the order inputted by the user 
    for node_info in input_nodes
        #Add the node to the vector
        push!(ordered_input_nodes, nodes_dict[node_info.name])
    end

    ## Order state nodes
    #Initialize empty vector for storing properly ordered state nodes
    ordered_state_nodes = []

    #For each specified state node, in the order inputted by the user 
    for node_info in state_nodes
        #Add the node to the vector
        push!(ordered_state_nodes, nodes_dict[node_info.name])
    end

    ### Create HGF structure ###
    #Initialize lists
    input_nodes_dict = Dict{String,InputNode}()
    state_nodes_dict = Dict{String,StateNode}()

    #Go through each node
    for node in nodes_dict
        #Put input nodes in one dictionary
        if typeof(node[2]) == InputNode
            input_nodes_dict[node[1]] = node[2]

            #Put state nodes in another
        elseif typeof(node[2]) == StateNode
            state_nodes_dict[node[1]] = node[2]
        end
    end

    #Create HGF structure containing the lists of nodes
    HGF = HGFStruct(
        update_HGF,
        input_nodes_dict,
        state_nodes_dict,
        ordered_input_nodes,
        ordered_state_nodes,
    )

    return HGF
end