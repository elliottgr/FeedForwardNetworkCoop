## A selection of functions passable to calcOj in the NetworkGameFuncs.jl game loop

function heaviside(x)
    if x > 0
        return 1
    elseif x <= 0
        return 0
    end
end

function ELU(x)
    if x > 0
        return 1
    elseif x <= 0
        return exp(x) - 1
    end
end

function ReLU(x)
    return maximum([0, x])
end

function linear(x)
    return x
end

function bounded_linear(x)
    return minimum([1, maximum([x, 0])])
end

##JVC's exponential AKA Logistic
function jvc_exp(x)
    return (1/(1+exp(-x)))
end

##LeNagard's exponential
function lenagard_exp(x)
    return (1 - exp(-(x^2)))
end