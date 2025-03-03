import pandas as pd
import matplotlib.pyplot as plt

ruta = '/Users/ferleon/Documents/GitHub/semestre_IV/graphic_visualization/clases/datasaurus.csv'
ruta2 = '/Users/ferleon/Documents/GitHub/semestre_IV/graphic_visualization/clases/datos_inscripcion.csv'

df = pd.read_csv(ruta)
grupos = df.groupby('dataset')
nombres = df["dataset"].unique()

print(nombres)

for n in nombres:
    grupo = grupos.get_group(n)  
    plt.figure()
    plt.scatter(grupo['x'], grupo['y'])
    plt.title(n)  
    plt.show()


print(grupos.agg(['mean', 'std', 'count']))



df2 = pd.read_csv(ruta2)
grupos2 = df2.groupby('curso')
print(grupos2.agg(['mean', 'std', 'count']))
