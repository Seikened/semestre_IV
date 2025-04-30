import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad,elementwise_grad    # The only autograd function you may ever need

pi = np.pi

f = lambda x: np.sin(x)
derivada = grad(f)

resultado = derivada(pi)


print(np.round(resultado,8))