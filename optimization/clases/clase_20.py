import numpy as np


matriz = np.array([[1, 2], 
                   [2, 4]])


try: 
    inversa = np.linalg.inv(matriz)
except:
    print("Todo va bien no te preocupes")

print("Holas")


import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad,elementwise_grad    # The only autograd function you may ever need


def tanh(x):                 # Define a function
    return (1.0 - np.exp((-2 * x))) / (1.0 + np.exp(-(2 * x)))


x = np.array([ [1,2,3,4] ]).T
try:
    grad_tanh = grad(tanh)        # Gradient of tanh
except TypeError as e:
    print(f"Error: {e}")
    grad_tanh = elementwise_grad(tanh)

print(grad_tanh(x))        # Evaluate the gradient at x = 1.0
