using SimpleML
using Base.Test

using HDF5
using JLD


function testLogisticOnevsAll()
    f=jldopen(joinpath(Pkg.dir(),"SimpleML/test/mnist.jld"), "r")
    X=read(f, "X")
    y=read(f, "y")
    m = size(X, 1); #number of training set data points
    n = size(X, 2); #number of features
    X = [ones(m, 1) X];

    all_theta = logisticAll(X, y, .1, 10)

    p=logisticPredictAll(all_theta, X)
    acc=mean(int(p.==y))
    print("Accuracy of training set: $acc")
    @assert acc > .95 
    
end


testLogisticOnevsAll()