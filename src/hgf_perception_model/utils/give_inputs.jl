"""
    give_inputs!(hgf::HGFStruct, inputs::Number)

Function for inputting multiple observations to an hgf. Input is a single value.
"""
function give_inputs!(hgf::HGFStruct, inputs::Number)

    ### Input data ###
    #Input the value to the hgf
    update_hgf!(hgf, inputs)

    return nothing
end


"""
    give_inputs!(hgf::HGFStruct, inputs::Array)

Function for inputting multiple observations to an hgf. Input is structured as an array, with one column per input node and one row per timestep.
"""
function give_inputs!(hgf::HGFStruct, inputs::Array)

    ### Checks ###
    #If number of column in input is diffferent from amount of input nodes
    if size(inputs, 2) != length(hgf.input_nodes)
        #Raise an error
        throw(
            ArgumentError(
                "the number of columns in the input is different from the number of input nodes in the model",
            ),
        )
    end

    ### Input data ###
    #Take each row in the array
    for rownr = 1:size(inputs, 1)
        #Input it to the hgf
        update_hgf!(hgf, inputs[rownr, :])
    end

    return nothing
end


"""
    give_inputs!(hgf::HGFStruct, inputs::Dict{String,Vector})

Function for inputting multiple observations to an hgf. Input is structured as a dictionary with a vector for each input node.
"""
function give_inputs!(hgf::HGFStruct, inputs::Dict{String,Vector})

    ### Checks ###
    #If specified input destinations do not match input nodes
    if keys(input) != keys(hgf.input_nodes)
        #Raise an error
        throw(
            ArgumentError(
                "the input nodes specified in the input do not match the input nodes in the model",
            ),
        )
    end

    #Make empty list for populating
    lengths_list = []
    #Go through each input
    for input_list in values(inputs)
        #Add the length of the input to the list
        push!(lengths_list, length(input_list))
    end

    #If all lengths are not equal
    if .!all(y -> y == lengths_list[1], lengths_list)
        #Raise an error
        throw(ArgumentError("the lists of inputs to each node are not of the same length"))
    end

    ### Input data ###
    #Create empty dictionary for a single input
    input = Dict()

    #Go through a sequence as long as the length of the first entry
    for input_nr = 1:length(first(inputs)[2])
        #Go through each node in the input
        for input_node in keys(inputs)
            #Save the input from the corresponding input number
            input[input_node] = inputs[input_node][input_nr]
        end
        #And input it to the hgf
        update_hgf!(hgf, input)
    end

    return nothing
end