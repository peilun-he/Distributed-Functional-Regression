generate_data <- function(n ,t, dgp)
{
  # Generate functional regression data. For FLM, we use the DGP as in 
  # Beyaztas, Tez and Shang (2024). For FPLM and FNPM, we use the DGP as in
  # Ferraty, Mas and Vieu (2007).
  #
  # Inputs: 
  #   n: number of observations.
  #   t: a set of grid points.
  #   dgp: which DGP to use, "FLM", "FPLM", or "FNPM".
  #     - FLM: functional linear model
  #     - FPLM: functional partial linear model
  #     - FNPM: functional non-parametric model
  #
  # Outputs: 
  #   y: a n-by-1 matrix of scalar response.
  #   beta: a T-by-1 matrix of functional coefficient, where T is the length of t.
  #   x: functional predictor. A "fdata" class object. 
  
  x <- matrix(0, nrow = n, ncol = length(t)) # functional predictor
  dt <- t[2] - t[1]
  
  if (dgp == "FLM") {
    # functional linear model
    for (i in 1:n)
    {
      for (j in 1:5)
      {
        k <- rnorm( 1, mean = 0, sd = sqrt(4*j^(-3/2)) )
        v <- sin(j * pi * t) - cos(j * pi * t)
        x[i, ] <- x[i, ] + k*v
      }
      #x[i, ] <- x[i, ] + rnorm(length(t), mean = 0, sd = 0.1*sd(x[i, ]))
    }
    beta <- matrix(2 * sin(2*pi*t), nrow = length(t), ncol = 1) # functional coefficient
    y <- x %*% beta %*% dt + rnorm(n, mean = 0, sd = 0.01) # functional response
    z <- NULL
    
  } else if (dgp == "FNPM") {
    # functional non-parametric model
    m <- numeric(n)
    
    for (i in 1:n) {
      a <- runif(1, min = 0, max = 1)
      b <- runif(1, min = 0, max = 1)
      omega <- runif(1, min = 0, max = 2*pi)
      x[i, ] <- cos(omega*t) + (a + 2*pi)*t + b
      m[i] <- abs(-omega*sin(omega*t)+a+2*pi) %*% (1 - cos(pi*t)) %*% dt
    }
    beta <- NULL
    y <- m + rnorm(n, mean = 0, sd = sqrt(2)) # functional response
    z <- NULL
    
  } else if (dgp == "FPLM") {
    # functional partial linear model
    m <- numeric(n)
    beta_NF <- 0.5 # beta for non-functional covariate
    z <- rnorm(n, mean = 0, sd = 1)
    
    for (i in 1:n) {
      a <- runif(1, min = 0, max = 1)
      b <- runif(1, min = 0, max = 1)
      omega <- runif(1, min = 0, max = 2*pi)
      x[i, ] <- cos(omega*t) + (a + 2*pi)*t + b
      m[i] <- abs(-omega*sin(omega*t)+a+2*pi) %*% (1 - cos(pi*t)) %*% dt
    }
    beta <- NULL
    y <- beta_NF * z + m + rnorm(n, mean = 0, sd = sqrt(2)) # functional response
    
  }
  
  x <- fdata(x, argvals = t)
  
  return(list(y = y, beta = beta, x = x, z = z))
}
