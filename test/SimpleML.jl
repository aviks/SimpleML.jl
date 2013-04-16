using SimpleML
using Base.Test

using HDF5
using JLD

f=jldopen(joinpath(Pkg.dir(),"SimpleML/test/mnist.jld"), "r")

function testLogisticOnevsAll()
    
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

function testNN2()
    input_layer_size  = 400
    hidden_layer_size = 25
    num_labels = 10
    X=read(f, "X")
    y=read(f, "y")
    m = size(X, 1); #number of training set data points
    n = size(X, 2); #number of features
    fwts=jldopen(joinpath(Pkg.dir(),"SimpleML/test/wts.jld"), "r")
    theta1 = read(fwts, "theta1")
    theta2 = read(fwts, "theta2")
    params = [theta1[:] ; theta2[:] ]

    theta1_grad = zeros(size(theta1))
    theta2_grad = zeros(size(theta2))

    grad = [theta1_grad[:] ; theta2_grad[:]]

    lambda=0
    @test_approx_eq SimpleML.nn2Cost(params, input_layer_size,  hidden_layer_size, num_labels, X, y, lambda, grad) 0.287629

    lambda=1
    @test_approx_eq SimpleML.nn2Cost(params, input_layer_size,  hidden_layer_size, num_labels, X, y, lambda, grad) 0.383770


end


#testLogisticOnevsAll()
@test_approx_eq_eps SimpleML.sigmoidGradient([1 -0.5 0 0.5 1])  [0.196612 0.235004 0.250000 0.235004 0.196612] 1e-6