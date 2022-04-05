set.seed(100)

# Predictor variable as a set of uniform rv's over the interval [-2, 2]
x <- runif(60, min=-2, max=2)

# Response variable
y <- function(x) { 
  
  Y = (cos(2*x + 1))
  
  return(Y) 
  
}


# Initialize the number of hidden neurons
hidden_neurons = 5

# Activation Function
sigmoid <- function(x) { 
  
  z = (1 / (1 + exp(-x)))
  
  return(z) 
  
}

# Randomly initializing the weights as i.i.d. N(0,1) rv's...
w1 = matrix(rnorm(2*hidden_neurons), nrow=hidden_neurons, ncol=2)
w2 = matrix(rnorm(hidden_neurons + 1), nrow=1, ncol=(hidden_neurons+1))

# Function to obtain predicted outputs
feedForward <- function(x, w1, w2, activation) {
  
  output <- rep(0, length(x))
  
  for (i in 1:length(x)) {
    
    a1 = w1 %*% matrix(rbind(1, x[i]), ncol=1)
    
    z1 = activation(a1)
    
    a2 = w2 %*% matrix(rbind(1, z1), ncol=1)
    
    output[i] = a2
  }
  
  
  return(output)
}

# Defining the derivative of our activation function needed for computing the gradient
derivativeActivation <- function(x) {
  
  g = (sigmoid(x) * (1 - sigmoid(x)))
  
  return(g)
  
}

# Function for computing model error 
modelError <- function(x, y, w1, w2, activation) {
  
  # obtaining predictions
  preds <- feedForward(x, w1, w2, activation)
  
  # Calclating the error
  SSE <- sum((y - preds) ** 2)
  
  return (SSE)
  
}

#Function for computing the gradients 
backPropagation <- function(x, y, w1, w2, activation, derivativeActivation) {
  
  preds <- feedForward(x, w1, w2, activation) #predicted values
  
  derivCost <- -2*(y - preds) #Derivative of the cost function (first term)
  
  dW1 <- matrix(0,ncol=2,nrow=nrow(w1)) #Gradient for w1
  dW2 <- matrix(rep(0,length(x)*(dim(w2)[2])),nrow=length(x)) #Gradient matrix for w2
  
  # Computing the Gradient for W2
  for (i in 1:length(x)) {
    
    a1 = w1 %*% matrix(rbind(1, x[i]), ncol=1)
    da2dW2 = matrix(rbind(1, activation(a1)), nrow=1)
    dW2[i,] = derivCost[i] * da2dW2
    
  }
  
  #Computing the gradient for W1
  for (i in 1:length(x)) {
    
    a1 = w1 %*% matrix(rbind(1, x[i]), ncol=1)
    da2da1 = derivativeActivation(a1) * matrix(w2[,-1], ncol=1)
    da2dW1 = da2da1 %*% matrix(rbind(1, x[i]), nrow=1)
    
    dW1 = dW1 + derivCost[i] * da2dW1
  }
  
  # Storing gradients for w1, w2 in a list
  gradient <- list(dW1, colSums(dW2))
  
  return (gradient)
}


# Defining our Stochastic Gradient Descent algorithm which will adjust our weight matrices
SGD <- function(x, y, w1, w2, activation, derivative, learnRate, epochs) {
  
  SSEvec <- rep(NA, epochs) #Empty array to store SSE values after each epoch
  SSEvec[1] = modelError(x, y, w1, w2, activation)
  
  for (j in 1:epochs) {
    
    for (i in 1:length(x)) {
      
      gradient <- backPropagation(x[i], y[i], w1, w2, activation, derivative)
      
      
      #Adjusting model parameters for a given number of epochs
      w1 <- w1 - learnRate * gradient[[1]] 
      
      w2 <- w2 - learnRate * gradient[[2]]
      
    }
    
    SSEvec[j+1] <- modelError(x, y, w1, w2, activation)#Storing SSE values after each iteration
    
  }
  
  #Beta vector holding model parameters
  B <- list(w1,w2) 
  result <- list(B, SSEvec)
  
  return(result)
  
}


# Running SGD function to obtain our optimized mode and parameters
model <- SGD(x, y(x), w1, w2, sigmoid, derivativeActivation, learnRate = 0.01, epochs = 200)


# Obtaining our adjusted SSE's for each epoch..
SSE <- model[[2]]

#Plotting the SSE  from each epoch vs # epochs
plot(seq(0,200,1),SSE, main="Model SSE by Number of Epochs",xlab = "Epochs", ylab = "Error",col="red",type = "b")


#Extracting our new parameters from our model...
new_W1 <- model[[1]][[1]]
new_W2 <- model[[1]][[2]]

# Comparing our old weight matrices against the new ones
print(w1)
print(new_W1)

print(w2)
print(new_W2)

#Obtaining our new predictions using our optimized parameters
y_pred <- feedForward(x, new_W1, new_W2, sigmoid)

#Plotting training data against our model predictions
plot(x,y(x), main = "Target Response vs. Predictions",xlab="Observations", ylab="Responses", col="red")
lines(x, y_pred, col="blue", type="p")
legend(y=1,x=0.7, legend=c('Data points','Fitted Values'), col=c('red','blue') , lty=c(0,0), pch=c(1,1))
