# using Distributed:nprocs, addprocs
# nprocs()
# addprocs(3)

module Synaprune

    using Flux
    using Flux:softmax, sigmoid
    using Statistics:mean
    using Distributed:addprocs, nprocs, @distributed

    abstract type Layers end

    mutable struct Build <: Layers
        I
        W::Array
        O
        f
    end

    mutable struct Last <: Layers
        I
        W::Array
        O
        f
    end        

    struct Freeze
        layer
    end
                
    function linear(x)
        return x
    end

    function sigmoid(x)
        return Flux.sigmoid.(x)
    end

    Devarg(x) = x[argmax(abs.(x), dims=1)]

    Build(in::Integer, out::Integer, f) = Build(Nothing, 2*rand(out, in) .- 1, Nothing, f)
    Build(in::Integer, out::Integer) = Build(Nothing, 2*rand(out, in) .- 1, Nothing, linear)
    Last(in::Integer, out::Integer, f) = Last(Nothing, 2*rand(out, in) .- 1, Nothing, f)
    Last(in::Integer, out::Integer) = Last(Nothing, 2*rand(out, in) .- 1, Nothing, linear)

    function (m::Layers)(x)
        m.O = m.f(m.W*x)
        m.I = x
        return m.O
    end

    function (f::Freeze)(x)
        return f.layer(x)
    end

    function Update!(f::Freeze, loss, loss1, learning_rate)
        return nothing
    end

    function Standardize(x)
        return x ./ sum(abs.(x), dims=1)
    end

    function Update!(b::Build, loss, loss1, learning_rate)
        Local = Standardize(b.O)
        Xn = Standardize(b.I)
        O, I = size(b.W)
        w = zeros(O, I, size(b.I)[2])
        for i in 1:size(b.I)[2]
            w[:,:,i] = matmul(loss1[i], Local[:,i], Xn[:,i])
        end    
        b.W += learning_rate * w[argmax(abs.(w), dims=3)][:,:]
    end

    matmul(loss, Xn) = loss * transpose(Xn)
    matmul(loss1, Local, Xn) = loss1 * (Local * transpose(Xn))


    function Update!(b::Last, loss, loss1, learning_rate)
        Xn = Standardize(b.I)
        O, I = size(L.W)
        w = zeros(O, I, size(L.I)[2])
        for i in 1:size(L.I)[2]
            w[:,:,i] = matmul(loss[:,i], Xn[:,i])
        end
        b.W += learning_rate * w[argmax(abs.(w), dims=3)][:,:]
    end

    function train!(model::Flux.Chain, data, actual, learning_rate)
        pred = model(data)
        error = actual - pred
        error1 = Devarg(error)
        @distributed for m in 1:length(model)
            Update!(model[m], error, error1, learning_rate)
        end
        @show mean(abs.(error))
    end


end
