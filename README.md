# realoptions
Real Options Analysis in Go

This model is an example of how approximate dynamic programming can be used to value 
real options using Monte Carlo simulation. The example here is based on Schwartz (2004), see
the reference below. The algorithm uses a variant of the Least Squares Monte Carlo (LSM) algorithm
from Longstaff & Schwartz (2001). LSM is a parametric value function approximation approach that estimates
the value function of the dynamic program using Ordinary Least Squares.

For a more extensive blog post discussing the R&D model see <a href="https://freeholdfinance.com/2021/04/01/valuing-rd-and-patents-with-real-options-analysis/">here</a>.

## References
Longstaff, F.A., Schwartz, E.S. Valuing American options by simulation: a simple least-squares approach. 
The Review of Financial Studies, Volume 14, Issue 1, January 2001, Pages 113–147, 

Schwartz, E.S., Patents and R&D as real options. Economic Notes, 33(1), 23-54


