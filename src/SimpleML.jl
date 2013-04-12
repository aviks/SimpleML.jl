module SimpleML

using Optim


export logisticPredict, logistic, logisticReg, logisticAll, logisticPredictAll

sigmoid(x) = (1 + exp (-1 * x)) .^ -1


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
        for j=1:n
            for i = 1:m
                grad[j] =  grad[j]  + (h[i] - y[i])*X[i,j];
            end

           if j==1
                grad[j] = grad[j]/m;
           else 
                grad[j] = ( grad[j] + lambda* theta[j]) / m;
           end
        end
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

    println("Converged: $converged @ $(fval[end]) with coefficients $results in $fcount iterations. ")

    return results
end

function logisticReg(X, y, lambda)
    initial_theta = zeros(size(X,2))
    
    results, fval, fcount, converged = cgdescent((g,t)->logisticCostReg(t, X, y, g, lambda),  initial_theta)

    println("Converged: $converged @ $(fval[end]) with coefficients $results in $fcount iterations. ")

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



end