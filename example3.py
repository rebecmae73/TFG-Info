import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Simulamos algunos datos: lanzamientos de un dado donde "6" ocurre más a menudo
torch.manual_seed(6)
data_nofilter = torch.randint(1, 7, (50,))
print(data_nofilter)
data = torch.tensor([1 if x == 6 else 0 for x in torch.randint(1, 7, (50,))], dtype=torch.float)

# Definimos el modelo generativo: cómo se producen los datos a partir de z
def model(data):
    # Probabilidad latente de obtener un "6" (z ~ Uniform(0, 1))
    alpha_q = torch.tensor(2.0)
    beta_q = torch.tensor(2.0)
   
    z = pyro.sample("z", dist.Beta(alpha_q, beta_q))
    
    # Para cada lanzamiento, el resultado depende de z
    for i in range(len(data)):
        pyro.sample(f"obs_{i}", dist.Bernoulli(z), obs=data[i])

# Definimos la guía (q(z)): una aproximación a la posterior p(z | x)
def guide(data):
    # Usamos dos parámetros aprendidos para definir una distribución Beta
    alpha_q = pyro.param("alpha_q", torch.tensor(2.0), constraint=dist.constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(2.0), constraint=dist.constraints.positive)
    pyro.sample("z", dist.Beta(alpha_q, beta_q))

# Configuramos el optimizador y el algoritmo SVI (Stochastic Variational Inference)
optimizer = Adam({"lr": 1e-3})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Entrenamos la guía
num_steps = 2000
for step in range(num_steps):
    loss = svi.step(data)
    if step % 500 == 0:
        print(f"Step {step} - Loss: {loss:.4f}")

# Resultado: los valores de los parámetros alpha y beta de la distribución Beta
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()
print(f"\nEstimación final de alpha_q: {alpha_q:.4f}")
print(f"Estimación final de beta_q: {beta_q:.4f}")

# Calculamos la media y la varianza de la distribución Beta aproximada
mean_q = alpha_q / (alpha_q + beta_q)
variance_q = (alpha_q * beta_q) / ((alpha_q + beta_q)**2 * (alpha_q + beta_q + 1))
print(f"\nEstimación final de z (media de Beta): {mean_q:.4f}")
print(f"Varianza asociada a la estimación de z: {variance_q:.4f}")
