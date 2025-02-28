import time
import torch
import torch.nn as nn
import torch.optim as optim

# Definición de dispositivos
device_cpu = torch.device("cpu")
device_gpu = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Dispositivo CPU: {device_cpu} | Dispositivo GPU: {device_gpu}")

# Generamos datos sintéticos: y = 2x + 1 + ruido (con 100000 muestras)
def generate_data(n_samples=100000, seed=42):
    torch.manual_seed(seed)
    x = torch.rand(n_samples, 1)
    noise = torch.randn(n_samples, 1) * 0.1
    y = 2 * x + 1 + noise
    return x, y

# Modelo de regresión lineal simple
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Entrenamiento NO optimizado: procesa muestra por muestra
def train_model_non_optimized(model, x, y, device, epochs=10, learning_rate=0.1):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    start = time.time()
    for epoch in range(epochs):
        for i in range(x.size(0)):
            optimizer.zero_grad()
            xi = x[i].unsqueeze(0).to(device)  # [1,1]
            yi = y[i].unsqueeze(0).to(device)
            output = model(xi)
            loss = criterion(output, yi)
            loss.backward()
            optimizer.step()
    end = time.time()
    print(f"[NO OPTIMIZADO] Dispositivo: {device} | Loss final: {loss.item():.4f} | Tiempo: {end - start:.4f} seg")
    return model

# Entrenamiento OPTIMIZADO: procesa todo el batch de datos
def train_model_optimized(model, x, y, device, epochs=1000, learning_rate=0.1):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    x = x.to(device)
    y = y.to(device)
    
    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    end = time.time()
    print(f"[OPTIMIZADO] Dispositivo: {device} | Loss final: {loss.item():.4f} | Tiempo: {end - start:.4f} seg")
    return model

# Generamos los datos con 10000 muestras
#x, y = generate_data(n_samples=10000)

# Generamos los datos con 100000 muestras
x, y = generate_data(n_samples=100000)

# -------------------------------
# 1. CPU: Versión no optimizada
print("=== Entrenamiento NO optimizado en CPU ===")
model_cpu_non_opt = SimpleLinearModel()
train_model_non_optimized(model_cpu_non_opt, x, y, device_cpu, epochs=10, learning_rate=0.1)

# 2. CPU: Versión optimizada
print("\n=== Entrenamiento OPTIMIZADO en CPU ===")
model_cpu_opt = SimpleLinearModel()
train_model_optimized(model_cpu_opt, x, y, device_cpu, epochs=1000, learning_rate=0.1)

# -------------------------------
# 3. GPU: Versión no optimizada (usando los mismos datos)
print("\n=== Entrenamiento NO optimizado en GPU ===")
model_gpu_non_opt = SimpleLinearModel()
train_model_non_optimized(model_gpu_non_opt, x, y, device_gpu, epochs=10, learning_rate=0.1)

# 4. GPU: Versión optimizada
print("\n=== Entrenamiento OPTIMIZADO en GPU ===")
model_gpu_opt = SimpleLinearModel()
train_model_optimized(model_gpu_opt, x, y, device_gpu, epochs=1000, learning_rate=0.1)

