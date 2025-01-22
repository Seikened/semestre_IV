import pandas as pd
import re

ruta_archivo = '/Users/ferleon/Documents/GitHub/semestre_IV/machine_learning/tareas/tarea_1/df_hosp.csv'

data_frame = pd.read_csv(ruta_archivo, sep=',', header=0)
data_frame.rename(columns=lambda x: x.strip(), inplace=True)
data_frame["anio"] = None

anio_actual = None
filas_para_eliminar = []

for i in range(len(data_frame)):
    valor_primera_columna = str(data_frame.iloc[i, 0]).strip()
    if "AÑO" in valor_primera_columna.upper():
        coincidencia = re.search(r"(\d{4})", valor_primera_columna)
        if coincidencia:
            anio_actual = int(coincidencia.group(1))
        filas_para_eliminar.append(i)
    else:
        data_frame.loc[i, "anio"] = anio_actual

data_frame.drop(filas_para_eliminar, inplace=True)
data_frame.reset_index(drop=True, inplace=True)

data_frame["anio"] = data_frame["anio"].astype(float)
data_frame["anio"] = data_frame["anio"].fillna(2018).astype(int)
data_frame.rename(columns={"AÑO 2018": "indicador"}, inplace=True)
data_frame.to_csv("df_hosp_limpio.csv", index=False, encoding='utf-8-sig')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data_frame)
