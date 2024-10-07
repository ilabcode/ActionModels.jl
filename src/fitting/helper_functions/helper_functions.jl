#####################################################################################################
####### FUNCTIONS FOR EXTRACTING A VALUE WHICH WORKS WITH DIFFERENT AUTODIFFERENTIATION BACKENDS ####
#####################################################################################################
function ad_val(x::ReverseDiff.TrackedReal)
    return ReverseDiff.value(x)
end
function ad_val(x::ReverseDiff.TrackedArray)
    return ReverseDiff.value(x)
end
function ad_val(x::ForwardDiff.Dual)
    return ForwardDiff.value(x)
end
function ad_val(x::Real)
    return x
end
