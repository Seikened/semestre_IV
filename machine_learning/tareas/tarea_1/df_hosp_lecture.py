import pandas as pd


dataframe = pd.read_csv("/Users/ferleon/Documents/GitHub/semestre_IV/machine_learning/tareas/tarea_1/df_hosp_limpio.csv",sep=',', header=0)


#print(dataframe.head())



dataframe_mejorado = pd.melt(dataframe, id_vars=[
    'indicador', 'anio'],
    value_vars=['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO',
                'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE'],
    var_name='mes',
    value_name='valor'
    )

print(dataframe_mejorado.head(32))

print("Meses unico posibles:")
print(dataframe_mejorado['mes'].unique())



# Mapeo de meses a números
mes_map = {
    'ENERO': '01', 'FEBRERO': '02', 'MARZO': '03', 'ABRIL': '04',
    'MAYO': '05', 'JUNIO': '06', 'JULIO': '07', 'AGOSTO': '08',
    'SEPTIEMBRE': '09', 'OCTUBRE': '10', 'NOVIEMBRE': '11', 'DICIEMBRE': '12'
}


dataframe_mejorado['mes_num'] = dataframe_mejorado['mes'].map(mes_map)

dataframe_mejorado['fecha'] = dataframe_mejorado['anio'].astype(str) + '-' + dataframe_mejorado['mes_num']


print(dataframe_mejorado[['anio', 'mes', 'mes_num', 'fecha']].head(1000))

dataframe_mejorado.to_csv("/Users/ferleon/Documents/GitHub/semestre_IV/machine_learning/tareas/tarea_1/df_hosp_apilado.csv", index=False, encoding="utf-8-sig")
