## A selection of functions passable to calcOj in the NetworkGameFuncs.jl game loop

function linear_unbounded(x)
    return x
end

function linear(x)
    return minimum([1, maximum([x, 0])])
end

##JVC's exponential
function jvc_exp(x)
    return (1/(1+exp(-x)))
end

##LeNagard's exponential
function lenagard_exp(x)
    return (1 - exp(-(x^2)))
end