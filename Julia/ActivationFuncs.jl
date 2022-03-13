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