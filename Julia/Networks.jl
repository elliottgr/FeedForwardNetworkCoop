module Networks
export Network
export netinit!, layerout!, netout!, buildweights

"""
Network object that contains the edge and node weights within the array
`weights`. Node weights (or bias or intercept) values are on the diagonal and
edge weights between each layer and its preceding layers are below the block
diagnoal.

NOTE: Node weights for the first (input) layer are the initial input values
"""
mutable struct Network
    nnodes::Int64
    nlayer::Array{Int64,1}
    sumlayer::Array{Int64,1}
    weights::Array{<:Real,2}
    nodevals::Array{<:Real,1}
    activef::Function

    # Base constructor. Check that weight matrix, input, and output are right size
    function Network(nlayer::Array{Int64,1}, weights::Array{<:Real,2}, activef::Function,
                     nodevals::Array{<:Real,1}, sumlayer::Array{Int64,1}, nnodes::Int64)
        # error checking
        if ndims(weights) > 2
            throw(ArgumentError("Weight matrix must be 2D"))
        end
        if sum(size(weights) .== [nnodes, nnodes]) < 2
            throw(ArgumentError(string("Weight matrix must be ",
                                        nnodes, "x", nnodes)))
        end
        if length(nodevals) != nnodes
            throw(ArgumentError(string("nodevals must be ", nnodes, " elements")))
        end

        # copy the initial input in the weight matrix to input nodes
        # (bare for loops are TEH FASTESTSSS)
        for i in 1:nlayer[1]
            nodevals[i] = weights[i,i]
        end

        new(nnodes, nlayer, sumlayer, weights, nodevals, activef)
    end

    # Construct network with given weight matrix
    function Network(nlayer::Array{Int64,1}, weights::Array{<:Real,2}, activef::Function)
        nnodes = sum(nlayer)
        sumlayer = cumsum(nlayer)
        nodevals = zeros(eltype(weights), nnodes)

        Network(nlayer, weights, activef, nodevals, sumlayer, nnodes)
    end

end

"""
"Initialize" network by copying initial values from weight matrix to nodevals
"""
function netinit!(net::Network)
    for i in 1:net.nlayer[1]
        net.nodevals[i] = net.weights[i,i]
    end
end

"""
Calculate feedforward output of layer l based on outputs of previous layers
"""
function layerout!(l::Int64, net::Network)

    # calculate `weights * prev_outs + bias` using bare loops for speeeeeed...
    lindex = net.sumlayer[l]+1:net.sumlayer[l+1]
    for i = lindex
        net.nodevals[i] = 0.0
        for j = 1:net.sumlayer[l]
            net.nodevals[i] += net.weights[i,j] * net.nodevals[j]
        end
        net.nodevals[i] += net.weights[i,i]
    end

    # apply activation function to layer l "input" to calculate layer l output
    @. @views net.nodevals[lindex] = net.activef(net.nodevals[lindex])

    nothing
end

"""
Evaluate the network. The input vector is the external stimuli.
The output vector contains the input vector, all the node outputs, and the
output vector. In other words, if we have N_i inputs, N_s nodes,
and N_o outputs, the output vector is of length N_i+N_s+N_o.
"""
function netout!(net::Network)

    # Calculate the outout of each layer (layer 1 through output layer)
    for l in 1:length(net.nlayer)-1
        layerout!(l, net)
    end

    nothing
end

"""
Construct network object from list of layer sizes, list of node weights, and
list of weights matrices.

Each list of weight matrices corresponds to the matrices for a specific layer
where the first matrix connects that layer to the first layer, the second to the
second layer, etc.

Final weight matrix has is weights = [w_ij] where w_ij is weight of edge from
node j to node i

NOTE: node weights for the input nodes are the initial inputs
"""
function buildweights(nlayer::Array{Int64,1}, nodew, layerw...)

    nnodes = sum(nlayer)
    sumlayer = cumsum([0; nlayer])
    weights = zeros(nnodes, nnodes)

    if length(nlayer)-1 != length(layerw)
        throw(ArgumentError(string("layerw must have ", length(nlayer)-1, " elements")))
    end

    if length(nodew) != nnodes
        throw(ArgumentError(string("nodew must have ", nnodes,
                                   " elements (input nodes + hidden layers + output nodes)")))
    end

    for i in 1:length(nlayer)-1
        if length(layerw[i]) != i
            throw(ArgumentError(string("element ", i, " of layerw must have ", i, " elements")))
        end

        # for each weight matrix j for the ith layer, save it in the weight matrix
        rindex = sumlayer[i+1]+1:sumlayer[i+2]
        for j in 1:length(layerw[i])
            cindex = sumlayer[j]+1:sumlayer[j+1]
            if sum([length(rindex), length(cindex)] .== size(layerw[i][j])) < 2
                throw(DimensionMismatch(string("weight matrix ", j,
                                               " of layer ", i,
                                               " is ", size(layerw[i][j]),
                                               " and should be ",
                                               [length(rindex), length(cindex)])))
           end
            weights[rindex, cindex] = layerw[i][j]
        end
        weights[rindex, rindex] = diagm(nodew[rindex-nlayer[1]])
    end

    weights
end

end
