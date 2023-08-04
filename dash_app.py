import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from scipy.stats import binom

import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor


app = dash.Dash(__name__)

layout1 = html.Div(
    style={'max-width': '1600px', 'margin': 'auto', 'padding': '20px'},
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
        html.Div(id='probability-value'),  # To display the value of the slider
        html.Div(
            style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'justify-content': 'center'},
            children=[
                html.Div(
                    style={'width': '25%', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                    children=[
                        html.H3("Sequential:"),
                        dcc.Input(id='sequential-input', type='number', value=3),
                        html.H3("Contemporaneous:"),
                        dcc.Input(id='contemporaneous-input', type='number', value=5),
                        # html.H2("Total Bets:"),
                        html.Div(id='product-output'),
                    ],
                ),
                dcc.Graph(
                    id='grid-scatter',
                    style={'width': '74%', 'height': '35vh','background-color': 'rgba(0, 0, 0, 0)'},
                ),
            ],
        ),
        html.Div(
            style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'},
            children=[
                dcc.Graph(
                    id='barplot',
                    style={ 'height': '35vh','width': '80%'},
                ),
            ]
        ),

        html.Div(
            style={'display': 'flex', 'flex-direction': 'column'},
            children=[
                html.H2("Bet Size:"),
                dcc.Slider(
                    id='bet-size-slider',
                    min=0,
                    max=10,
                    step=0.1,
                    value=1,
                    marks={i: f"{i} %" for i in range(0, 11, 1)},
                ),
                html.Div(id='bet-size-value'),  # To display the value of the bet size slider
                html.Div(
                    style={'display': 'flex', 'align-items': 'center'},
                    children=[
                        dcc.Checklist(
                            id='absorbing-state-checkbox',
                            options=[{'label': 'Absorbing State:', 'value': 'enabled'}],
                            value=['enabled'],  # Initially unchecked
                        ),
                        html.Div(id='absorbing-state-input-container', children=[
                            html.H3("Absorbing State: "),
                            # dcc.Input(id='absorbing-state-input', type='number', value=None, disabled=True),
                            dcc.Input(id='absorbing-state-input', type='number', value=0, disabled=False),
                        ]),
                    ],
                ),
            ],
        ),
    ],
)

layout2 = app.layout = html.Div([
    html.H1("Set Parameters"),
    html.Div([
        html.Div([
            html.Label("x_min:"),
            dcc.Input(id='x-min', type='number', value=0.001),
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),
        html.Div([
            html.Label("x_max:"),
            dcc.Input(id='x-max', type='number', value=0.1),
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),
        html.Div([
            html.Label("y_min:"),
            dcc.Input(id='y-min', type='number', value=0),
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),
        html.Div([
            html.Label("y_max:"),
            dcc.Input(id='y-max', type='number', value=1),
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div([
        html.Div([
            html.Label("Exponential Factor:"),
            dcc.Input(id='exp-factor', type='number', value=1),
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),
        html.Div([
            html.Label("Fractional Kelly Factor:"),
            dcc.Input(id='fractional-kelly-factor', type='number', value=1),
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div(id='output-plot')
])


app.layout = html.Div(
    style={'max-width': '1800px', 'margin': 'auto', 'padding': '20px', 'display': 'flex', 'flex-direction': 'column'},
    children=[
        html.H1("Coin Toss", style={'text-align': 'center'}),
        html.Div(
            style={'max-width': '1600px', 'margin': 'auto', 'padding': '20px', 'display': 'flex', 'flex-direction': 'row'},
            children=[
                html.Div(style={'width': '50%'}, children=layout1),  # First column for layout1
                html.Div(style={'width': '50%'}, children=layout2),  # Second column for layout2
            ]
        )
    ]
)
# Simple callback for probability value
@app.callback(
    Output('probability-value', 'children'),
    Input('probability-slider', 'value')
)
def update_probability_value(value):
    return f"Probability: {value:.2f}"

# Callback for absorbing state

@app.callback(
    Output('absorbing-state-input', 'disabled'),
    Output('absorbing-state-input', 'value'),
    Input('absorbing-state-checkbox', 'value')
)
def update_absorbing_state_input(checklist_value):
    if 'enabled' in checklist_value:
        return False, 0
    else:
        return True, None


# Callback to update the product output
@app.callback(
    Output('product-output', 'children'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value')
)
def update_product_output(sequential, contemporaneous):
    product = sequential * contemporaneous
    return f"Total Bets: {product}"

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
        data=[go.Scatter(x=x_vals, y=y_vals, mode='markers', marker=dict(size=10, opacity=0.7))],
        layout=go.Layout(
            # title='Grid of Points',
            xaxis=dict(title='Sequential'),
            yaxis=dict(title='Contemporaneous'),
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        )
    )
    return grid_scatter

# Callback to update the bar plot
@app.callback(
    Output('barplot', 'figure'),
    Input('probability-slider', 'value'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value'),
    Input('bet-size-slider', 'value'),
    Input('absorbing-state-input', 'value')
)
def update_barplot(probability,
                   sequential,
                   contemporaneous,
                   betsize,
                   absorbing_state
                   ):
    # x_vals = list(range(1, contemporaneous * sequential + 1))
    # y_vals = [contemporaneous * sequential - x for x in x_vals]
    # barplot_figure = go.Figure(
    #     data=[go.Bar(x=x_vals, y=y_vals)],
    #     layout=go.Layout(
    #         # title='Bar Plot',
    #         xaxis=dict(title='x'),
    #         yaxis=dict(title='y')
    #     )
    # )
    ready_to_bar = create_ready_to_bar_df(
        p=probability,
        nsl=sequential,
        ncs=contemporaneous,
        s=betsize/100,
        absorbing=absorbing_state
        )
    fig = go.Figure(
            data = go.Bar(
            x=ready_to_bar.index,
            y=ready_to_bar['probability'],
            # width=widths # customize width here
        ),
        layout=go.Layout(
            title='Final Distribution',
            xaxis=dict(title='Final Dollar Value'),
            yaxis=dict(title='Probability'),
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        )
    )
    # print(ready_to_bar)
    # fig.show()
    return fig

@lru_cache(maxsize=600)
def create_nonrecombining_distribution(p=0.55, ncs=10, nsl=3, s=0.09, absorbing=0.0):
    def compute_distribution():
        rv = binom(ncs,p)
        distribution = {1: 1}

        for _ in range(nsl):
            next_distribution = {}
            for cur_el, cur_prob in distribution.items():
                for cur_x in range(ncs+1):
                    trans_prob = rv.pmf(cur_x)
                    if absorbing is not None and cur_el*(1-s*ncs)<absorbing:
                        next_el = cur_el
                    else:
                        next_el = cur_el * (1+s*(cur_x*2-ncs))
                    next_prob = trans_prob * cur_prob
                    next_distribution[next_el] = next_distribution.get(next_el, 0) + next_prob
            distribution = next_distribution
        # print(distribution)
        return distribution

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(compute_distribution)
        try:
            result = future.result(timeout=0.01)  # Set timeout of 0.2 seconds
        except TimeoutError:
            print("timeout")
            result = {}
        except Exception as e:
            print("An exception occurred:", str(e))
            result = {}
    return result

def create_ready_to_bar_df(
        p=0.55,
        ncs = 10,
        nsl = 3,
        s=0.09,
        absorbing = 0.000
        ):
    my_d = create_nonrecombining_distribution(
        p=p,
        ncs = ncs,
        nsl = nsl,
        s=s,
        absorbing = absorbing
        )
    my_d_l = {key: [el] for key, el in my_d.items()}
    pd_distribution = pd.DataFrame(my_d_l).T
    pd_distribution.columns = ['probability']
    bins = np.linspace(min(pd_distribution.index)-0.0001, max(pd_distribution.index)+0.0001, 20)
    pd_binned = pd.cut(pd_distribution.index, bins, labels=False)

    # Step 2: Calculate the bin centers
    bin_centers = (np.array(bins[1:]) + np.array(bins[:-1])) / 2

    pd_distribution['base'] = bin_centers[pd_binned]
    return pd_distribution.groupby('base').sum()



@app.callback(
    Output('output-plot', 'children'),
    Input('x-min', 'value'),
    Input('x-max', 'value'),
    Input('y-min', 'value'),
    Input('y-max', 'value'),
    Input('exp-factor', 'value'),
    Input('fractional-kelly-factor', 'value'),
    Input('probability-slider', 'value'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value'),
    Input('bet-size-slider', 'value'),
    Input('absorbing-state-input', 'value')
)
def update_plot(x_min, x_max, y_min, y_max, exp_factor, fractional_kelly_factor, probability, sequential, contemporaneous, betsize, absorbing_state):


    def util_exp(val):
        return 1-np.exp(-(val-1)/0.1)
    def util_log(val):
        if val <=0:
            return -np.inf #1e10
        else:
            return np.log(val)
        
    def distributionalize(old_util):
        def new_util(d):
            expectation = 0
            for it, el in d.items():
                expectation += el*old_util(it)
            return expectation
        return new_util

    util_exp_dist = distributionalize(util_exp)
    util_log_dist = distributionalize(util_log)

    fractions = np.linspace(x_min,x_max,60)
    utility_vector_exp = np.zeros(len(fractions))
    utility_vector_log = np.zeros(len(fractions))
    for it, cur_fraction in np.ndenumerate(fractions):
        cur_d = create_nonrecombining_distribution(
            ncs = contemporaneous,
            nsl = sequential,
            p=probability,
            s=cur_fraction,
            absorbing = absorbing_state
        )
        utility_vector_exp[it] = util_exp_dist(cur_d)
        utility_vector_log[it] = util_log_dist(cur_d)
    # print("frac", fractions)
    # print("utility", utility_vector_exp)
    # print(absorbing_state)
    # Use the inputs to create the plot
    trace_exp = go.Scatter(x=fractions, y=utility_vector_exp , mode='lines', name='exponential')
    trace_log = go.Scatter(x=fractions, y=utility_vector_log, mode='lines', name='kelly')

    layout = go.Layout(
        title='Utility Comparison',
        xaxis=dict(title='Fractions', range=[x_min, x_max]),
        yaxis=dict(title='Utility', range=[y_min, y_max]),
    )

    data = [trace_exp, trace_log]

    fig = go.Figure(data=data, layout=layout)
    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)