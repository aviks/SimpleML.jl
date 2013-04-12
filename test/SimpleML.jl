using SimpleML
using Test

using HDF5
using JLD


function testLogisticOnevsAll()
    f=jldopen(joinpath(Pkg.dir(),"SimpleML/test/mnist.jld"), "r")
    X=read(f, "X")
    y=read(f, "y")
    m = size(X, 1); #number of data points
    n = size(X, 2); #number of features
    X = [ones(m, 1) X];

    all_theta = logisticAll(X, y, .1, 10)

    p=logisticPredictAll(all_theta, X)
    @test_approx_eq mean((int(p.==y))) .9646
    

end


testLogisticOnevsAll()