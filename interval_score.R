interval_score <- function(x, lb, ub, alpha) {
  # Calculate the interval score. 
  #
  # Inputs: 
  #   x: a vector of observed values
  #   lb: a vector of lower bounds
  #   ub: a vector of upper bounds
  #   alpha: significance level 
  #
  # Outputs: 
  #   score: interval score
  
  score <- (ub - lb) + 2/alpha * (lb - x) * (x < lb) + 2/alpha * (x - ub) * (x > ub)
  
  return(mean(score))
}