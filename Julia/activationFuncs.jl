## A selection of functions passable to calcOj in the NetworkGameFuncs.jl game loop

function linear(x)
    return x
end

##JVC's exponential
function jvc_exp(x)
    return (1/(1+exp(-x)))
end

##LeNagard's exponential
function lenagard_exp(x)
    return (1 - exp(-(x^2)))
end

# struct activationFunctions
#     linear::Function
#     jvc_exp::Function
#     lenagard_exp::Function
# end

# activationFuncs = activationFunctions(linear, jvc_exp, lenagard_exp)
# # function_dict = Dict([("linear", linear),
# #                  ("jvc_exp", jvc_exp),
# #                  ("lenagard_exp", lenagard_exp)]
# #                  )