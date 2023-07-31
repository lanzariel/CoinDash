import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

app = dash.Dash(__name__)

app.layout = html.Div(
    style={'max-width': '800px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1("Coin Toss", style={'text-align': 'center'}),
        html.Div(
            style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
            children=[
                html.H2("Enter Probability:"),
                dcc.Slider(
                    id='probability-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.55,
                    marks={i/10: str(i/10) for i in range(11)},
                ),
                html.Div(
                    style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                    children=[
                        html.H2("Sequential:"),
                        dcc.Input(id='sequential-input', type='number', value=3),
                        html.H2("Contemporaneous:"),
                        dcc.Input(id='contemporaneous-input', type='number', value=5),
                        html.H2("Product:"),
                        html.Div(id='product-output'),
                    ],
                ),
            ],
        ),
        html.Div(
            style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'},
            children=[
                dcc.Graph(
                    id='grid-scatter',
                    style={'width': '50%', 'background-color': 'rgba(0, 0, 0, 0)'},
                ),
                dcc.Graph(
                    id='barplot',
                    style={'width': '50%'},
                ),
            ],
        ),
    ],
)

# Callback to update the product output
@app.callback(
    Output('product-output', 'children'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value')
)
def update_product_output(sequential, contemporaneous):
    product = sequential * contemporaneous
    return f"Product: {product}"

# Callback to update the grid scatter plot
@app.callback(
    Output('grid-scatter', 'figure'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value')
)
def update_grid_scatter(sequential, contemporaneous):
    points = [(x, y) for x in range(1, sequential + 1) for y in range(1, contemporaneous + 1)]
    x_vals = [el[0] for el in points]
    y_vals = [el[1] for el in points]
    grid_scatter = go.Figure(
        data=[go.Scatter(x=x_vals, y=y_vals, mode='markers', marker=dict(size=12, opacity=0.7))],
        layout=go.Layout(
            title='Grid of Points',
            xaxis=dict(title='Sequential'),
            yaxis=dict(title='Contemporaneous'),
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        )
    )
    return grid_scatter

# Callback to update the bar plot
@app.callback(
    Output('barplot', 'figure'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value')
)
def update_barplot(sequential, contemporaneous):
    x_vals = list(range(1, contemporaneous * sequential + 1))
    y_vals = [contemporaneous * sequential - x for x in x_vals]
    barplot_figure = go.Figure(
        data=[go.Bar(x=x_vals, y=y_vals)],
        layout=go.Layout(
            title='Bar Plot',
            xaxis=dict(title='x'),
            yaxis=dict(title='y')
        )
    )
    return barplot_figure

if __name__ == '__main__':
    app.run_server(debug=True)