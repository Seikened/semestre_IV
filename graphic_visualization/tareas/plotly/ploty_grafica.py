from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import dask.dataframe as dd
import networkx as nx
import plotly.graph_objects as go

app = Dash()

df = dd.read_csv(
    "https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv"
).compute()

def filter_df(pop_thresh, continent):
    filt_df = df[df["population"] > pop_thresh]
    if continent != "All":
        filt_df = filt_df[filt_df["continent"] == continent]
    return filt_df

def build_scatter(filt_df):
    fig = px.scatter(
        filt_df,
        x="gdp per capita",
        y="life expectancy",
        size="population",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=60,
    )
    return fig

def build_bar_chart(filt_df):
    avg_life_exp = filt_df.groupby("continent")["life expectancy"].mean().reset_index()
    fig = px.bar(
        avg_life_exp,
        x="continent",
        y="life expectancy",
        title="Esperanza de vida promedio por continente"
    )
    return fig

def build_network(filt_df):
    G = nx.Graph()

    for _, row in filt_df.iterrows():
        G.add_node(row["country"], gdp=row["gdp per capita"])

    for i, row1 in filt_df.iterrows():
        for j, row2 in filt_df.iterrows():
            if i != j and abs(row1["gdp per capita"] - row2["gdp per capita"]) < 5000:
                G.add_edge(row1["country"], row2["country"])

    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="gray"),
        hoverinfo="none",
        mode="lines"
    )

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        marker=dict(size=10, color="blue"),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Conexiones entre países con GDP similar", showlegend=False)
    return fig

app.layout = html.Div(
    [
        html.H1("Visualización Interactiva con Plotly Dash"),
        dcc.Slider(
            id="pop-slider",
            min=df["population"].min(),
            max=df["population"].max(),
            value=5 * 10**6,
            marks={i: str(i) for i in range(0, int(df["population"].max()), 500000000)},
            step=1000000,
        ),
        dcc.Dropdown(
            id="continent-dropdown",
            options=[{"label": "All", "value": "All"}] +
            [{"label": c, "value": c} for c in df["continent"].unique()],
            value="All",
        ),
        dcc.Graph(id="scatter-plot"),
        dcc.Graph(id="bar-chart"),
        dcc.Graph(id="network-graph"),
    ]
)

@app.callback(
    [Output("scatter-plot", "figure"), Output("bar-chart", "figure"), Output("network-graph", "figure")],
    [Input("pop-slider", "value"), Input("continent-dropdown", "value")],
)
def update_graphs(pop_thresh, continent):
    filt_df = filter_df(pop_thresh, continent)
    return build_scatter(filt_df), build_bar_chart(filt_df), build_network(filt_df)

if __name__ == "__main__":
    app.run(debug=True)
