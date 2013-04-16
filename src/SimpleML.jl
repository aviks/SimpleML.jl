module SimpleML

using Optim
using OptionsMod


export logisticPredict, logistic, logisticReg, logisticAll, logisticPredictAll

sigmoid(x) = (1 + exp (-1 * x)) .^ -1

sigmoidGradient(z) = sigmoid(z) .* (1-sigmoid(z));



# function logisticCost(theta, X, y)
#     m = length(y) #size of training set

#     h = sigmoid(X * theta)

#     sum =0
#     for i = 1:m 
#         sum = sum + ( -1 * y[i] * log(h[i]) - (1-y[i]) * log (1-h[i]))
#     end

#     return sum / m
# end

function logisticCostGradient(theta, X, y)
    grad = zeros(size(theta))
    m = length(y)
    h = sigmoid(X * theta)
    for i = 1:m
        for j = 1:length(theta)
            grad[j] =  grad[j]  + (h[i] - y[i])*X[i,j]
       end
    end

    map!(x->x/m , grad)

    return grad
end

#theta : coefficients
#X : feature matrix
#y : training labels
#grad : output vector for gradient
function logisticCost(theta, X, y, grad)
    m = length(y) #size of training set
    h = sigmoid(X * theta)

    sum =0
    for i = 1:m 
        sum = sum + ( -1 * y[i] * log(h[i]) - (1-y[i]) * log (1-h[i]))
    end

    if !(grad === nothing)
        for i = 1:m
            for j = 1:length(theta)
                grad[j] =  grad[j]  + (h[i] - y[i])*X[i,j]
           end
        end

        map!(x->x/m , grad)
    end 

    return sum/m

end 

function logisticCostReg(theta, X, y, grad, lambda)
    m = length(y) #size of training set
    n = length(theta) #number of features

    h = sigmoid(X * theta)

    sum =0
    for i = 1:m 
        sum = sum + ( -1 * y[i] * log(h[i]) - (1-y[i]) * log (1-h[i]))
    end    
    for j=2:n 
        sum = sum + (lambda/2) * (theta[j] ^ 2);
    end

    if !(grad === nothing)
        gptr = pointer(grad)
        for j=1:n
            grad[j]=0
            for i = 1:m
                grad[j] =  grad[j]  + (h[i] - y[i])*X[i,j];
            end
            
           if j==1
                grad[j] = grad[j]/m;
           else 
                grad[j] = ( grad[j] + lambda* theta[j]) / m;
           end
        end
        @assert gptr == pointer(grad)
    end

    return sum / m
end

function logisticCostGradientReg(theta, X, y, lambda)
    grad=zeros(size(theta))
    m = length(y) #size of training set
    n = length(theta) #number of features

    h = sigmoid(X * theta)

    return grad
end

function logistic(X, y)
    initial_theta = zeros(size(X,2))

    results, fval, fcount, converged = cgdescent((g,t)->logisticCost(t, X, y, g),  initial_theta)

    #println("Converged: $converged @ $(fval[end]) with coefficients $results in $fcount iterations. ")

    return results
end

function logisticReg(X, y, lambda)
    initial_theta = zeros(size(X,2))


    ops = @options itermax=50 tol=1e-5
    
    results, fval, fcount, converged = cgdescent((g,t)->logisticCostReg(t, X, y, g, lambda),  initial_theta, ops)

    #println("Converged: $converged @ $(fval[end]) with coefficients $results in $fcount iterations. ")

    return results
end

function logisticAll(X, y, lambda, num_labels)

    m = size(X, 1); #number of data points
    n = size(X, 2); #number of features

    all_theta = zeros(num_labels, n)

    for c=1:num_labels
        initial_theta = zeros(n , 1)
        all_theta[c,:]=logisticReg(X,int((y.==c)),lambda)[:]
    end

     return all_theta

end

function logisticPredict(theta, X )
    m = size(X, 1) 
    p = zeros(m, 1)
    h = sigmoid(X * theta);

    for i=1:m
        if h[i] < 0.5 
            p[i] = 0 
        else
            p[i] = 1
        end

    end
    return p
end

function logisticPredictAll(all_theta, X)
    m = size(X, 1) #Number of training examples
    n = size(X, 2) #Number of features
    num_labels = size(all_theta, 1)
    all_h = zeros(num_labels, m)
    for c = 1: num_labels
        theta = all_theta[c,:]
        h = sigmoid(X * theta')
        all_h[c,:] = h[:]
    end    

    ix = zeros(m)

    for i = 1:m
        _ , ix[i] = findmax(all_h[:,i])
    end

    return ix
end

#Predict a 2 layer neural network
function predict2nn(theta1, theta2, X)
    m = size(X, 1);
    num_labels = size(theta2, 1)
    p = zeros(size(X, 1), 1)
    X = [ones(m, 1) X]
    a2 = sigmoid(X * theta1'); 
    a2 = [ones(m, 1) a2]
    a3 = sigmoid (a2 * theta2')

    for i=1:m
        _, p[i] = findmax(a3[:,i])
    end

    return ix
end

#Cost function for a 2 layer neural network
function nn2Cost(params, input_layer_size,  hidden_layer_size, num_labels, X, y, lambda, grad)

    theta1 = reshape(params[1:hidden_layer_size * (input_layer_size + 1)], hidden_layer_size, (input_layer_size + 1))
    theta2 = reshape(params[(1 + (hidden_layer_size * (input_layer_size + 1))):end], num_labels, (hidden_layer_size + 1))

    m = size(X, 1)

    J = 0
    theta1_grad = zeros(size(theta1))
    theta2_grad = zeros(size(theta2))

    X = [ones(m, 1) X];

    z2 = X * theta1'
    a2 = sigmoid(z2) 

    a2 = [ones(m, 1) a2]
    a3 = sigmoid (a2 * theta2')
    s=0
    for i=1:m
        yvector = [1:num_labels] == y[i]
        s=s+sum(-yvector .* log(a3[i,:]) - (1-yvector) .* log (1-a3[i,:])); 
    end

    temp1 = theta1
    temp2 = theta2
    temp1[:, 1] = zeros(size(temp1, 1), 1)
    temp2[:, 1] = zeros(size(temp2, 1), 1)

    J = s/m  + (lambda/(2*m))*(sum(sum(temp1 .* temp1)) + sum(sum(temp2 .* temp2)))

    if !(grad === nothing)
        gptr = pointer(grad)
        for t=1:m
            yvector = [1:num_labels] == y[t]

            delta3 = zeros(1,num_labels)
            delta3 = a3[t,:] - yvector

            delta2 = zeros (1, hidden_layer_size);
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

end

end