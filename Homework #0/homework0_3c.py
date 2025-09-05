## Ravi Raghavan, Homework #0, Problem 3(c)

from scipy.optimize import minimize

def f(vars):
    x, y = vars
    return (x)**2 + (y)**2 - 6*x*y - 4*x - 5*y

# Unperturbed problem
# Initial guess
x0 = [2, 3]

# Constraints
constraints = [
    {'type': 'ineq', 'fun': lambda vars: -(vars[0] - 2)**2 - vars[1] + 4},  
    {'type': 'ineq', 'fun': lambda vars: vars[0] + vars[1] - 1}       
]

# Minimize using SLSQP
result = minimize(f, x0, method='SLSQP', constraints=constraints)

print("--- Unperturbed Problem ---\n")

# Results
print("Converged:", result.success)
print("Message:", result.message)
print("Optimal variables:", result.x)
print("Minimum value:", result.fun)

print("\n--- Perturbed Problem ---\n")

# Perturbed Problem
# Initial guess
x0 = [2, 3]

# Constraints
constraints = [
    {'type': 'ineq', 'fun': lambda vars: -(vars[0] - 2)**2 - vars[1] + 4.1},  
    {'type': 'ineq', 'fun': lambda vars: vars[0] + vars[1] - 1}       
]

# Minimize using SLSQP
result = minimize(f, x0, method='SLSQP', constraints=constraints)

# Results
print("Converged:", result.success)
print("Message:", result.message)
print("Optimal variables:", result.x)
print("Minimum value:", result.fun)
