# load R packages and functions
library(fda.usc)
library(fda)
library(ggplot2)

source("generate_data.R")
source("interval_score.R")

# Data generating process: "FLM", "FPLM", or "FNPM"
#   - FLM: functional linear model
#   - FPLM: functional partial linear model 
#   - FNPM: functional non-parametric model 
dgp <- "FNPM"

# Estimation methods: "basis", "pc", "FPLM", or "FNPM"
#   - basis: B-spline expansion (only for FLM)
#   - pc: PCA method (only for FLM)
#   - FPLM: functional partial linear model 
#   - FNPM: functional non-parametric model 
est_method <- "FNPM"

n1 <- 2000 # number of observations for the training data
n2 <- 800 # number of observations for the testing data
n_block <- 40 # number of blocks for distributed learning
n_mc <- 200 # number of Monte Carlo experiments
alpha <- 0.05 # significance level 

if (est_method == "basis") {
  n_basis_x <- 20 # number of basis for functional predictor
  n_basis_b <- 5 # number of basis for beta function
} else if (est_method == "pc") {
  n_pc <- 5 # number of PCs used in PCA method 
}

# Grid points
dt <- 0.01
if (dgp == "FLM") {
  t <- seq(from = 0, to = 1, by = dt) 
} else if (dgp == "FPLM" | dgp == "FNPM") {
  t <- seq(from = -1, to = 1, by = dt) 
}

# Criteria
rmse1 <- numeric(n_mc) # RMSE for training data
re1 <- numeric(n_mc) # relative error for training data
mae1 <- numeric(n_mc) # mean absolute error for training data
cp1 <- numeric(n_mc) # coverage probability for training data
is1 <- numeric(n_mc) # interval score for training data
rmse2 <- numeric(n_mc) # RMSE for testing data
re2 <- numeric(n_mc) # relative error for testing data
mae2 <- numeric(n_mc) # mean absolute error for testing data
cp2 <- numeric(n_mc) # coverage probability for testing data
is2 <- numeric(n_mc) # interval score for testing data
exe_time <- matrix(0, nrow = n_mc, ncol = n_block) # execution time
rescaled_frob <- numeric(n_mc) # rescaled Frobenius norm 
beta_hat_all <- matrix(0, nrow = length(t), ncol = n_mc) # estimated beta for all Monte Carlo experiments

# Monte Carlo simulations
for (mc in 1: n_mc) {
  #########################
  ##### Simulate data #####
  #########################
  print(mc)
  
  set.seed(1234 + mc)
  
  # Training data
  dat1 <- generate_data(n1, t, dgp)
  y1 <- dat1$y
  beta <- dat1$beta
  x1 <- dat1$x
  z1 <- dat1$z
  
  # Testing data
  dat2 <- generate_data(n2, t, dgp)
  y2 <- dat2$y
  x2 <- dat2$x
  z2 <- dat2$z
  
  beta_hat_block <- matrix(0, nrow = length(t), ncol = n_block) # estimated beta in each block
  y_hat <- c() # a vector of all estimated responses
  y_real <- c() # a vector of all real responses
  predict_y <- matrix(0, nrow = n2, ncol = n_block) # predicted responses for testing data in each block
  sd_train_block <- numeric(n_block) # standard deviation of training residuals in each block
  gamma_block <- numeric(n_block) # gamma in each block
  
  # Split data into different blocks
  set.seed(1111 + mc)
  index <- sample(1: n1, replace = FALSE)
  
  if (n_block == 1) {
    y_split <- y1
    x_split <- x1
    if (est_method == "FPLM") z_split <- z1
  } else {
    y_split <- split(y1[index], cut(1: n1, breaks = n_block, labels = FALSE))
    x_split <- split(data.frame(x1$data[index, ]), cut(1: n1, breaks = n_block, labels = FALSE))
    if (est_method == "FPLM") z_split <- split(z1[index], cut(1: n1, breaks = n_block, labels = FALSE))
  }
  
  ################################
  ##### Distributed learning #####
  ################################
  for (k in 1: n_block) {
    # Get data
    if (n_block == 1) {
      y_block <- y_split
      x_block <- x_split
      if (est_method == "FPLM") z_block <- z_split
    } else {
      y_block <- y_split[[k]]
      x_block <- fdata(x_split[[k]], argvals = t)
      if (est_method == "FPLM") z_block <- z_split[[k]]
    }
    
    start_time <- Sys.time()
    
    # Functional regression
    if (est_method =="basis") {
      # functional linear model using B-spline expansion
      basis_x <- create.bspline.basis(rangeval = range(x_block[['argvals']]), nbasis = n_basis_x)
      basis_b <- create.bspline.basis(rangeval = range(x_block[['argvals']]), nbasis = n_basis_b)  
      fr_model <- fregre.basis(fdataobj = x_block, y = y_block, basis.x = basis_x, basis.b = basis_b)
      basis_value <- eval.basis(evalarg = t, basisobj = basis_b)
      beta_hat <- basis_value %*% fr_model$coefficients[2: (n_basis_b+1)]
    } else if (est_method == "pc") {
      # functional linear model using PCA method
      fr_model <- fregre.pc(fdataobj = x_block, y = y_block, l = 1:n_pc)
      beta_hat <- t(fr_model$beta.est$data)
    } else if (est_method == "FNPM") {
      # functional non-parametric model
      fr_model <- fregre.np.cv(fdataobj = x_block, y = y_block, type.CV = "CV.S")
      beta_hat <- NULL
    } else if (est_method == "FPLM") {
      # functional partial linear model
      dat <- list("df" = data.frame(y = y_block, z = z_block), "x" = x_block)
      fr_model <- fregre.plm(y ~ z + x, data = dat, type.CV = "CV.S")
      beta_hat <- NULL
    }
    
    end_time <- Sys.time()
    
    exe_time[mc, k] <- end_time - start_time
    
    # Estimation  
    if (est_method == "basis" | est_method == "pc") beta_hat_block[, k] <- beta_hat
    
    y_hat <- c(y_hat, fr_model$fitted.values)
    y_real <- c(y_real, y_block)
    
    # Testing data
    if (est_method == "FPLM") {
      dat_test <- list("df" = data.frame(y = y2, z = z2), "x" = x2)
      pred <- predict(fr_model, dat_test)
    } else {
      pred <- predict(fr_model, x2)
    }
    
    # Coverage probability
    residual <- fr_model$residuals # residuals of training data
    sd_train_block[k] <- sd(residual)
    gamma_block[k] <- quantile(abs(residual), 1-alpha) / sd_train_block[k]
    
    predict_y[, k] <- pred
  } # end of distributed learning
  
  ##############################
  ##### Calculate criteria #####
  ##############################
  # Estimation of beta
  if (est_method == "basis" | est_method == "pc") {
    beta_hat <- apply(beta_hat_block, 1, mean)
    rescaled_frob[mc] <- mean( (beta - beta_hat)^2 )
    beta_hat_all[, mc] <- beta_hat
  }
  
  # Estimation of y
  rmse1[mc] <- sqrt(mean( (y_real - y_hat)^2 ))
  re1[mc] <- mean(abs( (y_real - y_hat) / y_real ))
  mae1[mc] <- mean( abs(y_real - y_hat) )
  
  # Prediction error
  y2_hat <- apply(predict_y, 1, mean)
  rmse2[mc] <- sqrt(mean( (y2 - y2_hat)^2 ))
  re2[mc] <- mean(abs( (y2 - y2_hat) / y2 ))
  mae2[mc] <- mean( abs(y2 - y2_hat) )
  
  # Coverage probability
  sd_train <- mean(sd_train_block)
  gamma <- mean(gamma_block)
  cp1[mc] <- mean(y_real >= y_hat - gamma * sd_train
                  & y_real <= y_hat + gamma * sd_train)  
  cp2[mc] <- mean(y2 >= y2_hat - gamma * sd_train
                  & y2 <= y2_hat + gamma * sd_train)
  
  # intercal score
  is1[mc] <- interval_score(x = y_real, 
                            lb = y_hat - gamma * sd_train, 
                            ub = y_hat + gamma * sd_train, 
                            alpha = alpha)
  is2[mc] <- interval_score(x = y2, 
                            lb = y2_hat - gamma * sd_train, 
                            ub = y2_hat + gamma * sd_train, 
                            alpha = alpha)
} # end of Monte Carlo

if (est_method == "basis" | est_method == "pc") {
  bias <- mean( (apply(beta_hat_all, 1, mean) - beta)^2 )
  est_std <- sqrt( mean( apply(beta_hat_all, 1, var) ) )
}

print( paste("Execution time:", mean(apply(exe_time, 1, mean))) )
print( paste("RMSE (training data):", round(mean(rmse1), 4)) )
print( paste("RE (training data):", round(100*mean(re1), 2)) )
print( paste("MAE (training data):", round(mean(mae1), 4)) )
print( paste("Coverage probability (training data):", round(100*mean(cp1), 2)) )
print( paste("Interval score (training data):", round(mean(is1), 4)) )
print( paste("RMSE (testing data):", round(mean(rmse2), 4)) )
print( paste("RE (testing data):", round(100*mean(re2), 2)) )
print( paste("MAE (testing data):", round(mean(mae2), 4)) )
print( paste("Coverage probability (testing data):", round(100*mean(cp2), 2)) )
print( paste("Interval score (testing data):", round(mean(is2), 4)) )

if (est_method == "basis" | est_method == "pc") {
  print( paste("F.norm:", round( mean(rescaled_frob), 4 ) ) )
  print( paste("Bias:", round( mean(bias), 4) ) )
  print( paste("ST.DEV:", round(mean(est_std), 4) ) )
}



