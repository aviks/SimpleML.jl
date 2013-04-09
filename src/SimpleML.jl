module SimpleML

using Optim


export logisticPredict, logistic, logisticReg

sigmoid(x) = (1 + exp (-1 * x)) .^ -1

#theta : coefficients
#X : feature matrix
#y : training labels
#grad : output vector for gradient
function logisticCost(theta, X, y)
    m = length(y) #size of training set

    h = sigmoid(X * theta)

    sum =0
    for i = 1:m 
        sum = sum + ( -1 * y[i] * log(h[i]) - (1-y[i]) * log (1-h[i]))
    end

    return sum / m
end

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

function logisticCostReg(theta, X, y, lambda)
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

    return sum / m
end

function logisticCostGradientReg(theta, X, y, lambda)
    grad=zeros(size(theta))
    m = length(y) #size of training set
    n = length(theta) #number of features

    h = sigmoid(X * theta)
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
    return grad
end

function logistic(X, y)
    initial_theta = zeros(size(X,2))
    grad = zeros(size(initial_theta));
    initial_cost = logisticCost(initial_theta, X, y)
    println("Cost Function at intial theta: $initial_cost")

    results = optimize((t)->logisticCost(t, X, y),
                        (t)->logisticCostGradient(t, X, y),  initial_theta)

    println(results)

    return results.minimum
end

function logisticReg(X, y, lambda)
    initial_theta = zeros(size(X,2))
    grad = zeros(size(initial_theta))
    initial_cost = logisticCostReg(initial_theta, X, y, lambda)
    println("Cost Function at intial theta: $initial_cost")
    results = optimize((t)->logisticCostReg(t, X, y, lambda),
                        (t)->logisticCostGradientReg(t, X, y, lambda),  initial_theta)

    println(results)

    return results.minimum
end

function logisticPredict(theta, X )
    m = size(X, 1) #Number of training examples
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



end