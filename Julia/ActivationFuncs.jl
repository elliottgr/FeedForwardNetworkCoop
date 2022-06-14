## A selection of functions passable to calcOj in the NetworkGameFuncs.jl game loop

function heaviside(x)
    if x > 0
        return 1
    elseif x <= 0
        return 0
    end
end

function ReLU(x)
    return maximum(0, x)
end

function linear(x)
<<<<<<< HEAD
    return x
end

function bounded_linear(x)
    return minimum([1, maximum([x, 0])])
=======
    return minimum([1, maximum([(x + 0.5), 0])])
>>>>>>> 26d081a1d726fc271b1891614c00f193c71c58df
end

##JVC's exponential AKA Logistic
function jvc_exp(x)
    return (1/(1+exp(-x)))
end

##LeNagard's exponential
function lenagard_exp(x)
    return (1 - exp(-(x^2)))
end