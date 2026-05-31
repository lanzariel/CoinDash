import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from scipy.stats import binom

from functools import lru_cache


app = dash.Dash(__name__)

# ── shared style tokens ──────────────────────────────────────────────────────
SECTION = {'marginBottom': '24px'}
LABEL   = {'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}

# ── Left panel: controls + histogram ────────────────────────────────────────
layout1 = html.Div(
    style={'padding': '20px 24px'},
    children=[
        # Probability
        html.Div(style=SECTION, children=[
            html.Label("Win probability (p)", style=LABEL),
            dcc.Slider(
                id='probability-slider',
                min=0, max=1, step=0.01, value=0.55,
                marks={i / 10: f'{i / 10:.1f}' for i in range(11)},
                tooltip={'placement': 'bottom', 'always_visible': True},
            ),
        ]),

        # Sequential / Contemporaneous side-by-side with small grid preview
        html.Div(
            style={**SECTION, 'display': 'flex', 'alignItems': 'center', 'gap': '16px'},
            children=[
                html.Div(
                    style={'display': 'flex', 'flexDirection': 'column', 'gap': '12px', 'minWidth': '170px'},
                    children=[
                        html.Div([
                            html.Label("Sequential rounds", style=LABEL),
                            dcc.Input(id='sequential-input', type='number', value=3, min=1,
                                      style={'width': '80px'}),
                        ]),
                        html.Div([
                            html.Label("Contemporaneous bets", style=LABEL),
                            dcc.Input(id='contemporaneous-input', type='number', value=5, min=1,
                                      style={'width': '80px'}),
                        ]),
                        html.Div(id='product-output', style={'color': '#666', 'fontSize': '13px'}),
                    ],
                ),
                dcc.Graph(
                    id='grid-scatter',
                    style={'flex': '1', 'height': '22vh'},
                    config={'displayModeBar': False},
                ),
            ],
        ),

        # Bet size slider (max updated dynamically from ncs)
        html.Div(style={**SECTION, 'overflow': 'visible', 'paddingBottom': '32px'}, children=[
            html.Label("Bet size (% of wealth per bet)", style=LABEL),
            dcc.Slider(
                id='bet-size-slider',
                min=0, max=19, step=0.5, value=1,
                marks={i: f'{i}%' for i in range(0, 20, 2)},
                tooltip={'placement': 'bottom', 'always_visible': True},
            ),
        ]),

        # Absorbing / ruin barrier
        html.Div(style=SECTION, children=[
            html.Div(
                style={'display': 'flex', 'alignItems': 'center', 'gap': '12px'},
                children=[
                    dcc.Checklist(
                        id='absorbing-state-checkbox',
                        options=[{'label': 'Ruin barrier', 'value': 'enabled'}],
                        value=['enabled'],
                    ),
                    dcc.Input(id='absorbing-state-input', type='number', value=0,
                              style={'width': '80px'}),
                ],
            ),
        ]),

        # Histogram: wealth distribution at current bet size
        html.Label("Final wealth distribution (at current bet size)", style=LABEL),
        dcc.Loading(
            dcc.Graph(id='barplot', style={'height': '30vh'},
                      config={'displayModeBar': False}),
        ),
    ],
)

# ── Right panel: utility sweep ───────────────────────────────────────────────
layout2 = html.Div(
    style={'padding': '20px 24px'},
    children=[
        html.Label("Bet size sweep range", style=LABEL),
        html.Div(
            style={'overflow': 'visible', 'paddingBottom': '36px'},
            children=[
                dcc.RangeSlider(
                    id='fraction-range',
                    min=0, max=0.19, step=0.005,
                    value=[0.001, 0.19],
                    marks={v / 100: f'{v}%' for v in range(0, 20, 5)},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                ),
            ],
        ),
        dcc.Loading(html.Div(id='output-plot')),
    ],
)

# ── Root layout ──────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={'maxWidth': '1800px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1("Coin Toss — Kelly Analysis", style={'textAlign': 'center', 'marginBottom': '24px'}),
        html.Div(
            style={'display': 'flex', 'flexDirection': 'row', 'gap': '8px'},
            children=[
                html.Div(style={'width': '42%', 'overflow': 'visible'}, children=layout1),
                html.Div(style={'width': '58%', 'overflow': 'visible'}, children=layout2),
            ],
        ),
    ],
)


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output('absorbing-state-input', 'disabled'),
    Output('absorbing-state-input', 'value'),
    Input('absorbing-state-checkbox', 'value'),
)
def toggle_absorbing(checklist_value):
    enabled = 'enabled' in checklist_value
    return not enabled, (0 if enabled else None)


@app.callback(
    Output('product-output', 'children'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value'),
)
def update_product_output(sequential, contemporaneous):
    if sequential and contemporaneous:
        return f'Total bets per scenario: {sequential * contemporaneous}'
    return ''


@app.callback(
    Output('grid-scatter', 'figure'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value'),
)
def update_grid_scatter(sequential, contemporaneous):
    sequential = sequential or 1
    contemporaneous = contemporaneous or 1
    points = [(x, y) for x in range(1, sequential + 1) for y in range(1, contemporaneous + 1)]
    fig = go.Figure(
        data=[go.Scatter(
            x=[p[0] for p in points],
            y=[p[1] for p in points],
            mode='markers',
            marker=dict(size=8, opacity=0.7),
        )],
        layout=go.Layout(
            xaxis=dict(title='Round', dtick=1),
            yaxis=dict(title='Bet slot', dtick=1),
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=10, t=10, b=40),
        ),
    )
    return fig


# Dynamically cap slider maxima to 1/ncs − 1% (ruin boundary)
@app.callback(
    Output('bet-size-slider', 'max'),
    Output('bet-size-slider', 'marks'),
    Output('bet-size-slider', 'value'),
    Output('fraction-range', 'max'),
    Output('fraction-range', 'marks'),
    Output('fraction-range', 'value'),
    Input('contemporaneous-input', 'value'),
    State('bet-size-slider', 'value'),
    State('fraction-range', 'value'),
)
def update_slider_bounds(ncs, cur_bet, cur_range):
    ncs = max(ncs or 1, 1)
    # one bet at max_s loses exactly 100% of wealth when all ncs bets lose
    max_pct  = int(100 / ncs) - 1          # e.g. 19 for ncs=5
    max_frac = max_pct / 100               # e.g. 0.19

    bet_step  = max(1, max_pct // 8)
    bet_marks = {i: f'{i}%' for i in range(0, max_pct + 1, bet_step)}
    bet_val   = min(cur_bet or 1, max_pct)

    frac_step  = max(0.01, round(max_frac / 4, 2))
    frac_marks = {}
    v = 0.0
    while v <= max_frac + 0.001:
        frac_marks[round(v, 2)] = f'{round(v * 100)}%'
        v += frac_step
    cur_hi  = (cur_range or [0.001, max_frac])[1]
    cur_lo  = (cur_range or [0.001, max_frac])[0]
    frac_hi = min(cur_hi, max_frac)
    frac_lo = min(cur_lo, frac_hi - 0.005)

    return max_pct, bet_marks, bet_val, max_frac, frac_marks, [max(frac_lo, 0.001), frac_hi]


@app.callback(
    Output('barplot', 'figure'),
    Input('probability-slider', 'value'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value'),
    Input('bet-size-slider', 'value'),
    Input('absorbing-state-input', 'value'),
)
def update_barplot(probability, sequential, contemporaneous, betsize, absorbing_state):
    ready_to_bar = create_ready_to_bar_df(
        p=probability,
        nsl=sequential,
        ncs=contemporaneous,
        s=betsize / 100,
        absorbing=absorbing_state,
    )
    return go.Figure(
        data=go.Bar(x=ready_to_bar.index, y=ready_to_bar['probability']),
        layout=go.Layout(
            xaxis=dict(title='Final wealth (× starting wealth)', tickformat='.2f'),
            yaxis=dict(title='Probability'),
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=10, t=10, b=50),
        ),
    )


@app.callback(
    Output('output-plot', 'children'),
    Input('probability-slider', 'value'),
    Input('sequential-input', 'value'),
    Input('contemporaneous-input', 'value'),
    Input('bet-size-slider', 'value'),
    Input('absorbing-state-input', 'value'),
    Input('fraction-range', 'value'),
)
def update_plot(probability, sequential, contemporaneous, betsize, absorbing_state, fraction_range):

    def util_exp(val):
        return 1 - np.exp(-(val - 1) / 0.1)

    def util_log(val):
        return -np.inf if val <= 0 else np.log(val)

    def distributionalize(old_util):
        def new_util(d):
            return sum(prob * old_util(wealth) for wealth, prob in d.items())
        return new_util

    util_exp_dist = distributionalize(util_exp)
    util_log_dist = distributionalize(util_log)

    f_min, f_max = fraction_range or [0.001, 0.19]
    fractions = np.linspace(max(f_min, 0.001), f_max, 100)
    utility_vector_exp = np.zeros(len(fractions))
    utility_vector_log = np.zeros(len(fractions))
    for it, cur_fraction in np.ndenumerate(fractions):
        cur_d = create_nonrecombining_distribution(
            ncs=contemporaneous,
            nsl=sequential,
            p=probability,
            s=cur_fraction,
            absorbing=absorbing_state,
        )
        utility_vector_exp[it] = util_exp_dist(cur_d)
        utility_vector_log[it] = util_log_dist(cur_d)

    fig = go.Figure(
        data=[
            go.Scatter(x=fractions, y=utility_vector_exp, mode='lines', name='Exponential utility'),
            go.Scatter(x=fractions, y=utility_vector_log, mode='lines', name='Log utility (Kelly)'),
        ],
        layout=go.Layout(
            xaxis=dict(title='Bet size (fraction of wealth)', tickformat='.0%'),
            yaxis=dict(title='Expected utility'),
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=50, r=10, t=40, b=50),
        ),
    )
    fig.add_vline(
        x=betsize / 100,
        line_dash='dot',
        line_color='grey',
        annotation_text='current bet size',
        annotation_position='top right',
    )
    return dcc.Graph(figure=fig, style={'height': '65vh'}, config={'displayModeBar': False})


# ── Pure computation ─────────────────────────────────────────────────────────

@lru_cache(maxsize=600)
def create_nonrecombining_distribution(p=0.55, ncs=10, nsl=3, s=0.09, absorbing=0.0):
    rv = binom(ncs, p)
    distribution = {1: 1}
    for _ in range(nsl):
        next_distribution = {}
        for cur_el, cur_prob in distribution.items():
            for cur_x in range(ncs + 1):
                trans_prob = rv.pmf(cur_x)
                if absorbing is not None and cur_el * (1 - s * ncs) < absorbing:
                    next_el = cur_el
                else:
                    next_el = cur_el * (1 + s * (cur_x * 2 - ncs))
                next_prob = trans_prob * cur_prob
                next_distribution[next_el] = next_distribution.get(next_el, 0) + next_prob
        distribution = next_distribution
    return distribution


def create_ready_to_bar_df(p=0.55, ncs=10, nsl=3, s=0.09, absorbing=0.000):
    my_d = create_nonrecombining_distribution(p=p, ncs=ncs, nsl=nsl, s=s, absorbing=absorbing)
    pd_distribution = pd.DataFrame.from_dict(my_d, orient='index', columns=['probability'])
    bins = np.linspace(pd_distribution.index.min() - 0.0001,
                       pd_distribution.index.max() + 0.0001, 20)
    bin_indices = pd.cut(pd_distribution.index, bins, labels=False)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    pd_distribution['base'] = bin_centers[bin_indices]
    return pd_distribution.groupby('base').sum()


if __name__ == '__main__':
    app.run(debug=True)
