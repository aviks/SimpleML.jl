module SimpleML

using Optim
using OptionsMod


export predictLogistic, predictLogisticAll, learnLogistic, learnLogisticAll, predict2nn, learn2nn 

sigmoid(x) = (1 + exp (-1 * x)) .^ -1

sigmoidGradient(z) = sigmoid(z) .* (1-sigmoid(z))

#Cost function for a logistic regression, with regularisation
function logisticCost(theta, X, y, grad, lambda)
    m = length(y) #size of training set
    n = length(theta) #number of features

    h = sigmoid(X * theta)

    sum =0
    for i = 1:m 
        sum = sum + ( -1 * y[i] * log(h[i]) - (1-y[i]) * log (1-h[i]))
    end    
    for j=2:n 
        sum = sum + (lambda/2) * (theta[j] ^ 2)
    end

    if !(grad === nothing)
        gptr = pointer(grad)
        for j=1:n
            grad[j]=0
            for i = 1:m
                grad[j] =  grad[j]  + (h[i] - y[i])*X[i,j]
            end
            
           if j==1
                grad[j] = grad[j]/m
           else 
                grad[j] = ( grad[j] + lambda* theta[j]) / m
           end
        end
        @assert gptr == pointer(grad)
    end

    return sum / m
end



#Learn via a logistic regression of two states
#  X : features
#  y : lables (binary)
#  lambda : regularisation parameter
function learnLogistic(X, y, lambda)
    initial_theta = zeros(size(X,2))
    ops = @options itermax=50 tol=1e-5
    
    results, fval, fcount, converged = cgdescent((g,t)->logisticCost(t, X, y, g, lambda),  initial_theta, ops)
    #println("Converged: $converged @ $(fval[end]) with coefficients $results in $fcount iterations. ")

    return results
end

#Learn via a logistic regression
#  X : features
#  y : lables
#  nlables : number of lables
#  lambda : regularisation parameter
function learnLogisticAll(X, y, lambda, nlables)

    m = size(X, 1); #number of data points
    n = size(X, 2); #number of features

    all_theta = zeros(nlables, n)

    for c=1:nlables
        initial_theta = zeros(n , 1)
        all_theta[c,:]=learnLogistic(X,int((y.==c)),lambda)[:]
    end

     return all_theta

end

function predictLogistic(theta, X )
    m = size(X, 1) 
    p = zeros(m, 1)
    h = sigmoid(X * theta)

    for i=1:m
        if h[i] < 0.5 
            p[i] = 0 
        else
            p[i] = 1
        end
    end
    return p
end

function predictLogisticAll(all_theta, X)
    m = size(X, 1) #Number of training examples
    n = size(X, 2) #Number of features
    nlables = size(all_theta, 1)
    all_h = zeros(nlables, m)
    for c = 1: nlables
        theta = all_theta[c,:]
        h = sigmoid(X * theta')
        all_h[c,:] = h[:]
    end    

    p = zeros(m)

    for i = 1:m
        _ , p[i] = findmax(all_h[:,i])
    end

    return p
end

#Learn a 2 layer neural network
#  X : features
#  y : lables
#  input_sz: size of input layer
#  hidden_sz: size of hidden layer
#  nlables : number of lables
#  lambda : regularisation parameter
function learn2nn(X, y, input_sz, hidden_sz, nlables, lambda)

    initial_theta1 = SimpleML.randomInitialWeights(input_sz, hidden_sz)
    initial_theta2 = SimpleML.randomInitialWeights(hidden_sz, nlables)
    initial_params = [initial_theta1[:] ; initial_theta2[:]]


    ops = @options itermax=50 tol=1e-5
    
    results, fval, fcount, converged = cgdescent((g,t)->nn2Cost(t, X, y,input_sz, hidden_sz, nlables, lambda, g),  initial_params, ops)

    #println("Converged: $converged @ $(fval[end]) with coefficients $results in $fcount iterations. ")

    theta1 = reshape(results[1:hidden_sz * (input_sz + 1)], hidden_sz, (input_sz + 1))
    theta2 = reshape(results[(1 + (hidden_sz * (input_sz + 1))):end], nlables, (hidden_sz + 1))
    return theta1 ,theta2

end

#Predict a 2 layer neural network
function predict2nn(theta1, theta2, X)
    m = size(X, 1)
    nlables = size(theta2, 1)
    p = zeros(size(X, 1), 1)
    X = [ones(m, 1) X]
    a2 = sigmoid(X * theta1'); 
    a2 = [ones(m, 1) a2]
    a3 = sigmoid (a2 * theta2')

    for i=1:m
        _, p[i] = findmax(a3[i,:])
    end

    return p
end

#Cost function for a 2 layer neural network
function nn2Cost(params, X, y, input_sz,  hidden_sz, nlables,  lambda, grad)

    theta1 = reshape(params[1:hidden_sz * (input_sz + 1)], hidden_sz, (input_sz + 1))
    theta2 = reshape(params[(1 + (hidden_sz * (input_sz + 1))):end], nlables, (hidden_sz + 1))

    m = size(X, 1)

    J = 0
    theta1_grad = zeros(size(theta1))
    theta2_grad = zeros(size(theta2))

    X = [ones(m, 1) X]

    z2 = X * theta1'
    a2 = sigmoid(z2) 

    a2 = [ones(m, 1) a2]
    a3 = sigmoid (a2 * theta2')
    s=0
    for i=1:m
        yvector = int([1:nlables] .== y[i])
        s=s+sum(-yvector' .* log(a3[i,:]) - (1-yvector') .* log (1-a3[i,:])); 
    end

    temp1 = theta1
    temp2 = theta2
    temp1[:, 1] = zeros(size(temp1, 1), 1)
    temp2[:, 1] = zeros(size(temp2, 1), 1)


    if !(grad === nothing)
        gptr = pointer(grad)
        for t=1:m
            yvector = int([1:nlables] .== y[t])

            delta3 = zeros(1,nlables)
            delta3 = a3[t,:] - yvector'

            delta2 = zeros (1, hidden_sz)
            delta2 = (theta2' * delta3' )[2:end] .* sigmoidGradient(z2[t,:])' 

            theta2_grad = theta2_grad + delta3' * a2[t,:]
            theta1_grad = theta1_grad + delta2 * X[t,:]
        end

        theta1_grad = theta1_grad / m
        theta2_grad = theta2_grad / m

        theta1_grad = theta1_grad + (lambda/m) * temp1
        theta2_grad = theta2_grad + (lambda/m) * temp2

        grad[:] = [theta1_grad[:] ; theta2_grad[:]]
        @assert gptr == pointer(grad)
        
    end
    return s/m  + (lambda/(2*m))*(sum(sum(temp1 .* temp1)) + sum(sum(temp2 .* temp2)))
end

function randomInitialWeights(l_in, l_out)
    ep = 0.12
    rand(l_out, 1 + l_in) * 2 * ep - ep
end

end