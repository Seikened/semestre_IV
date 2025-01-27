import pandas as pd
import plotly.graph_objects as go

# Cargar el dataset limpio
dataframe = pd.read_csv(
    "/Users/ferleon/Documents/GitHub/semestre_IV/machine_learning/tareas/tarea_1/df_hosp_apilado.csv", 
    sep=',', 
    header=0
)

# Normalizar los datos
dataframe['indicador'] = (
    dataframe['indicador']
    .str.lower()
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)

# Variable de interés
variable = 'numero de pacientes que ingresan a admision hospitalaria'

# Filtrar el dataframe por el indicador deseado
filtered_df = dataframe[dataframe['indicador'] == variable]

# Crear columna de meses y años
filtered_df['mes_num'] = pd.to_datetime(filtered_df['fecha']).dt.month
filtered_df['anio'] = pd.to_datetime(filtered_df['fecha']).dt.year

# Agrupar por año y mes, y sumar valores
grouped_df = filtered_df.groupby(['anio', 'mes_num'], as_index=False).agg({'valor': 'sum'})

# Crear un gráfico interactivo
fig = go.Figure()

# Añadir una línea para cada año
for anio in grouped_df['anio'].unique():
    data_anio = grouped_df[grouped_df['anio'] == anio]
    fig.add_trace(
        go.Scatter(
            x=data_anio['mes_num'],
            y=data_anio['valor'],
            mode='lines+markers',
            name=f'Año {anio}'
        )
    )

# Personalizar diseño
fig.update_layout(
    title=f'{variable.capitalize()} (Por Mes y Año)',
    xaxis_title='Mes',
    yaxis_title='Valor',
    xaxis=dict(
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=[
            'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
        ]
    ),
    legend_title='Años',
    template='plotly_white',
    hovermode='x unified'
)

# Mostrar el gráfico interactivo
fig.show()