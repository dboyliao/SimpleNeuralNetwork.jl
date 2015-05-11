module SimpleNeuralNetwork

import Base: repr, show

export NeuralLayer, NeuralNetwork, train!, predict, repr, show

function derivative(fun::Function; epsilon = 1e-3)
    function (x)
        (fun(x + epsilon) - fun(x - epsilon)) / (2*epsilon)
    end
end

function sqr_cost(y_predict::Array, y::Array)
    sqrt(sum((y_predict - y).^2))
end

type NeuralLayer
    weight::Union(Matrix, Array{None, 1}) # Let conversion do the work....
    _nodes_value::Array{Float64, 1}
    
    function NeuralLayer(w, nv)
        new(w, nv)
    end
end

type NeuralNetwork
    structure::Array{Int, 1}
    act_fun::Function
    act_diff::Function
    layers::Array{NeuralLayer, 1}
    out_fun::Function
    out_diff::Function
    _deltas::Array{Vector{FloatingPoint}, 1} # Array for storing dels
    
    # Constructor for NeuralNetwork type
    function NeuralNetwork(struct; act_fun = tanh, act_diff = None, out_fun = x-> x, out_diff = None)
        if act_diff == None
            act_diff = derivative(act_fun)
        end
        if out_diff == None
            out_diff = derivative(out_fun)
        end
        layers = NeuralLayer[]
        for ind = 1:(length(struct)-1)
            dim_in = struct[ind]
            dim_out = struct[ind+1]
            b = sqrt(6) / sqrt(dim_in + dim_out)
            w = 2b*rand(dim_out, dim_in + 1) - b
            nodes_value = push!([1.0], rand(dim_in)...)
            temp_layer = NeuralLayer(w, nodes_value)
            push!(layers, temp_layer)
        end
        # append the output layer.
        w = []
        nodes_value = rand(struct[end])
        temp_layer = NeuralLayer(w, nodes_value)
        push!(layers, temp_layer)
        _deltas = Vector{Float64}[[1.] for i = 1:length(struct)]
        nn = new(struct, act_fun, act_diff, layers, out_fun, out_diff, _deltas)
    end    
end

function repr(nn::NeuralNetwork)
    struct = join([string(i) for i in nn.structure], "x")
    msg = join(["It is a ", struct, " NeuralNetwork.\n"] , "")
    msg = join([msg, "Activate Function: ", string(nn.act_fun), '\n'], "")
    msg = join([msg, "Output Function: ", string(nn.out_fun), '\n'], "")
    msg
end

function show(nn::NeuralNetwork)
    print(repr(nn))
    println()
end


function predict(nn::NeuralNetwork, data::Array{Float64, 2})
    predict_results = Array{Float64, 1}[]
    for data_id = 1:size(data)[1]
        v = data[data_id, :][:]
        #println(v)
        forward_prob!(nn, v)
        push!(predict_results, nn.out_fun(nn.layers[end]._nodes_value))
    end
    return predict_results
end

function forward_prob!(nn::NeuralNetwork, x)
    # forward_prob! will update nodes_value for all layers.
    nn.layers[1]._nodes_value = [1, x]
    n_layers = length(nn.structure)
    for layer_id = 1:(n_layers-2)
        current_layer = nn.layers[layer_id]
        next_layer = nn.layers[layer_id + 1]
        temp = current_layer.weight * current_layer._nodes_value
        next_layer._nodes_value = [1., nn.act_fun(temp)]
    end
    # Compute the node values of the last layer without pass through activation function.
    current_layer = nn.layers[end - 1]
    next_layer = nn.layers[end]
    temp = current_layer.weight * current_layer._nodes_value
    next_layer._nodes_value = temp[:]
    return
end

function back_prob!(nn::NeuralNetwork, x, y)
    # back_prob! will update nn._detas.
    forward_prob!(nn, x)
    nn._deltas[end] = -2.*(y - nn.out_fun(nn.layers[end]._nodes_value)).*nn.out_diff(nn.layers[end]._nodes_value)
    
    n_layers = length(nn.structure)
    for layer_id = n_layers-1:-1:1
        delta_next = nn._deltas[layer_id + 1]
        w_this = nn.layers[layer_id].weight[:, 2:end]
        nodes_value_this = nn.layers[layer_id]._nodes_value
        dd = nn.act_diff(nodes_value_this[2:end])
        temp = transpose(w_this) * delta_next
        nn._deltas[layer_id] = temp[:] .* dd
    end
    return
end

function train!(nn::NeuralNetwork, X::Matrix, Y::Array; epos = 10000, cost_fun = sqr_cost, tol = 0.0001, learning_rate = 0.1)
    y_predict = predict(nn, X)
    #return y_predict
    err_p = cost_fun(y_predict, Y)
    #return err_p
    n_obs = size(X)[1]
    #return n_obs
    errors = Float64[]
    for iter = 1:epos
        ind = rand(1:n_obs)
        x = X[ind, :][:]
        y = Y[ind]
        back_prob!(nn, x, y)
        #return nn
        for layer_id in 1:(length(nn.layers)-1)
            next_id = layer_id + 1
            nl = nn.layers[layer_id]
            gradient = nn._deltas[next_id]*nl._nodes_value'
            nl.weight -= learning_rate .* gradient
        end
        
        y_predict = predict(nn, X)
        err_t = cost_fun(y_predict, Y)
        push!(errors, err_t)
        if iter % 1000 == 0
            if err_p - err_t > 0 && err_p - err_t < tol
                println("Terminating training process due to no significant improvement.")
                println("At iteration No. ", iter)
                break
            end
        end
    end
    return errors
end

end # module
