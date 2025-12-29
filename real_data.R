# load R packages and functions
library(fda.usc)
library(fda)
library(fsemipar)
library(ggplot2)

source("generate_data.R")
source("interval_score.R")

# load data
data(Tecator)

# Data generating process: "FLM", "FPLM", or "FNPM"
#   - FLM: functional linear model
#   - FPLM: functional partial linear model 
#   - FNPM: functional non-parametric model 
dgp <- "FPLM"

# Estimation methods: "basis", "pc", "FPLM", or "FNPM"
#   - basis: B-spline expansion (only for FLM)
#   - pc: PCA method (only for FLM)
#   - FPLM: functional partial linear model 
#   - FNPM: functional non-parametric model 
est_method <- "FPLM"

n1 <- 150 # number of observations for the training data
n2 <- 65 # number of observations for the testing data
n_block <- 2 # number of blocks for distributed learning
n_mc <- 200 # number of Monte Carlo experiments
alpha <- 0.2 # significance level 

if (est_method == "basis") {
  n_basis_x <- 20 # number of basis for functional predictor
  n_basis_b <- 5 # number of basis for beta function
} else if (est_method == "pc") {
  n_pc <- 5 # number of PCs used in PCA method 
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

# Data
spectra <- Tecator$absor.spectra
fat <- Tecator$fat
protein <- Tecator$protein
moisture <- Tecator$moisture
grid <- seq(850, 1050, length.out = 100) # grid points

for (mc in 1: n_mc) {
  print(mc)
  set.seed(1234 + mc)
  
  ####################
  ##### Get data #####
  ####################
  index <- sample(1:215, replace = FALSE)
  
  # Resampling
  x <- spectra[index,] # functional covariate
  y <- fat[index] # response variable 
  z <- moisture[index] # non-functional covariate 
  
  # Training data
  x1 <- x[1: n1, ]
  y1 <- y[1: n1]
  z1 <- z[1: n1]
  
  # Testing data
  x2 <- x[(n1+1): 215, ]
  y2 <- y[(n1+1): 215]
  z2 <- z[(n1+1): 215]
  
  beta_hat_block <- matrix(0, nrow = length(grid), ncol = n_block) # estimated beta in each block
  y_hat <- c() # a vector of all estimated responses
  y_real <- c() # a vector of all real responses
  predict_y <- matrix(0, nrow = n2, ncol = n_block) # predicted responses for testing data in each block
  sd_train_block <- numeric(n_block) # standard deviation of training residuals in each block
  gamma_block <- numeric(n_block) # gamma in each block
  
  # Split data
  if (n_block == 1) {
    y_split <- y1
    x_split <- x1
    z_split <- z1
  } else {
    y_split <- split(y1, cut(1: n1, breaks = n_block, labels = FALSE))
    x_split <- split(data.frame(x1), cut(1: n1, breaks = n_block, labels = FALSE))
    z_split <- split(z1, cut(1: n1, breaks = n_block, labels = FALSE))
  }
  
  ################################
  ##### Distributed learning #####
  ################################
  for (k in 1: n_block) {
    # Get data for each block 
    if (n_block == 1) {
      y_block <- y_split
      x_block <- fdata(x_split, argvals = grid)
      z_block <- z_split
    } else {
      y_block <- y_split[[k]]
      x_block <- fdata(x_split[[k]], argvals = grid)
      z_block <- z_split[[k]]
    }
    
    start_time <- Sys.time()
    
    # Functional regression
    if (est_method =="basis") {
      # functional linear model using B-spline expansion
      basis_x <- create.bspline.basis(rangeval = range(x_block[['argvals']]), nbasis = n_basis_x)
      basis_b <- create.bspline.basis(rangeval = range(x_block[['argvals']]), nbasis = n_basis_b)  
      fr_model <- fregre.basis(fdataobj = x_block, y = y_block, basis.x = basis_x, basis.b = basis_b)
    } else if (est_method == "pc") {
      # functional linear model using PCA method
      fr_model <- fregre.pc(fdataobj = x_block, y = y_block, l = 1:n_pc)
    } else if (est_method == "FNPM") {
      # functional non-parametric model
      fr_model <- fregre.np.cv(fdataobj = x_block, y = y_block, type.CV = "CV.S")
    } else if (est_method == "FPLM") {
      # functional partial linear model
      dat <- list("df" = data.frame(y = y_block, z = z_block), "x" = x_block)
      fr_model <- fregre.plm(y ~ z + x, data = dat, type.CV = "CV.S")
    }
    
    end_time <- Sys.time()
    
    exe_time[mc, k] <- end_time - start_time
    
    # Estimation 
    y_hat <- c(y_hat, fr_model$fitted.values)
    y_real <- c(y_real, y_block)
    
    # Testing data
    if (est_method == "FPLM") {
      dat_test <- list("df" = data.frame(y = y2, z = z2), "x" = fdata(x2, argvals = grid))
      pred <- predict(fr_model, dat_test)
    } else {
      pred <- predict(fr_model, fdata(x2, argvals = grid))
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