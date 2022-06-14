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

##JVC's exponential AKA Logistic/sigmoid
function jvc_exp(x)
    return (1/(1+exp(-x)))
end

##LeNagard's exponential
function lenagard_exp(x)
    return (1 - exp(-(x^2)))
end

## From Richie
function SELU(x, α = 1.67326, λ = 1.0507)
    if (x < 0)
        return λ * α * (exp(x) - 1)
    else
        return λ * x
    end
end

function PReLU(x, α = 1.67326)
    if x < 0
        return α * x
    else
        return x
    end
end

function tanh(x)
    num = exp(x) - exp(-x)
    den = exp(x) + exp(-x)
    return num/den
end

function softplus(x)
    return log(1 + exp(x))
end

function LeakyReLU(x)
    if (x < 0)
        return .01 * x
    else
        return x
    end
end

function SiLU(x)
    return (x / 1 + exp(-x))
end

function gaussian(x)
    return exp(-(x^2))
end