using SimpleML
using Base.Test

using HDF5
using JLD
using Calculus

f=jldopen(joinpath(Pkg.dir(),"SimpleML/test/mnist.jld"), "r")

function testLogisticOnevsAll()
    
    X=read(f, "X")
    y=read(f, "y")
    m = size(X, 1); #number of training set data points
    n = size(X, 2); #number of features
    X = [ones(m, 1) X]

    all_theta = learnLogisticAll(X, y, .1, 10)

    p=predictLogisticAll(all_theta, X)
    acc=mean(int(p.==y))
    println("Accuracy of training set for Logistic: $acc")
    @assert acc > .95 
    
end

function testNN2()
    input_sz  = 400
    hidden_sz = 25
    num_labels = 10

    #Read input data and saved coefficients
    X=read(f, "X")
    y=read(f, "y")
    m = size(X, 1); #number of training set data points
    n = size(X, 2); #number of features
    fwts=jldopen(joinpath(Pkg.dir(),"SimpleML/test/wts.jld"), "r")
    theta1 = read(fwts, "theta1")
    theta2 = read(fwts, "theta2")

    #Test prediction function
    p=SimpleML.predict2nn(theta1, theta2, X)
    acc=mean(int(p.==y))
    @test_approx_eq acc 0.9758

    #Test Cost Function
    params = [theta1[:] ; theta2[:] ]
    theta1_grad = zeros(size(theta1))
    theta2_grad = zeros(size(theta2))
    grad = [theta1_grad[:] ; theta2_grad[:]]
    
    lambda=0
    @test_approx_eq_eps SimpleML.nn2Cost(params, X, y, input_sz,  hidden_sz, num_labels,  lambda, grad) 0.2382217 1e-6

    lambda=1
    @test_approx_eq_eps SimpleML.nn2Cost(params, X, y, input_sz,  hidden_sz, num_labels, lambda, grad) 0.4602118 1e-6

    #Test learning
    theta1, theta2 = learn2nn(X, y, input_sz, hidden_sz, num_labels, lambda)
    p=predict2nn(theta1, theta2, X)
    acc=mean(int(p.==y))
    println("Accuracy of training set for NN: $acc")
    @assert acc > .97

end

#test gradient calculations for dummy data
function testNNGradient(lambda)
    lambda=0
    input_sz = 3
    hidden_sz = 5
    num_labels = 3
    m = 5
    theta1 = debugInitialWeights(hidden_sz, input_sz)
    theta2 = debugInitialWeights(num_labels, hidden_sz)
    
    X  = debugInitialWeights(m, input_sz - 1)
    y  = 1 + mod([1:m], num_labels)'
    params = [theta1[:] ; theta2[:]]
    grad = zeros(size(params))

    J=SimpleML.nn2Cost(params, X, y, input_sz, hidden_sz, num_labels, lambda, grad)
    costFunc = (p) -> SimpleML.nn2Cost(p, X, y, input_sz, hidden_sz, num_labels, lambda, nothing)
    numgrad = derivative(costFunc, params, :central)
    diff = norm(numgrad-grad)/norm(numgrad+grad)
    @assert diff < 1e-9
end

#generate a well distributed, but deterministic values for initial weights, for testing
function debugInitialWeights(l_out, l_in)
    W = zeros(l_out, 1 + l_in)
    W = reshape(sin(1:length(W)), size(W)) / 10
    return W
end


#Run the tests
testLogisticOnevsAll()
@test_approx_eq_eps SimpleML.sigmoidGradient([1 -0.5 0 0.5 1])  [0.196612 0.235004 0.250000 0.235004 0.196612] 1e-6
testNNGradient(0)
testNNGradient(3)
testNN2()
