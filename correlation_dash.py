"""
Parallel Bets: Optimal Fraction vs Correlation

Shows how the optimal bet size per bet changes as pairwise correlation
between parallel bets increases from 0 (independent) to ~1 (co-moving),
for Kelly/CRRA and CARA utility functions.

Correlation model: one-factor Gaussian copula.
  Bet i wins iff  sqrt(rho)*Z + sqrt(1-rho)*eps_i  <  Phi^{-1}(p)
where Z ~ N(0,1) is a common factor and eps_i are i.i.d. N(0,1).
Given Z, wins are conditionally independent with probability
  p(Z) = Phi( (Phi^{-1}(p) - sqrt(rho)*Z) / sqrt(1-rho) )
Marginal win probability is exactly p and pairwise correlation is exactly rho.

Runs on port 8052.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy.stats import binom as scipy_binom
from scipy.special import ndtr, ndtri
from numpy.polynomial.hermite import hermgauss

app = dash.Dash(__name__)

LABEL   = {'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}
SECTION = {'marginBottom': '28px'}

# Gauss-Hermite nodes/weights for E_Z[f(Z)] = (1/sqrt(pi)) sum_i w_i f(sqrt(2)*z_i)
_GH_NODES, _GH_WEIGHTS = hermgauss(32)
_GH_W_NORM = _GH_WEIGHTS / np.sqrt(np.pi)


# ── Probability model ─────────────────────────────────────────────────────────

def correlated_pmf(ncs: int, p: float, rho: float) -> np.ndarray:
    """
    Return PMF array of length ncs+1: P(X=k) for k wins out of ncs bets
    with win probability p and pairwise correlation rho (one-factor copula).
    """
    k_arr = np.arange(ncs + 1)

    if rho < 1e-9:
        return scipy_binom.pmf(k_arr, ncs, p)

    if rho > 1 - 1e-9:
        # Perfect correlation: all win or all lose together
        probs = np.zeros(ncs + 1)
        probs[0]   = 1 - p
        probs[ncs] = p
        return probs

    thresh = ndtri(p)                                                          # Phi^{-1}(p)
    z_vals = np.sqrt(2) * _GH_NODES                                            # (n_quad,)
    p_cond = ndtr((thresh - np.sqrt(rho) * z_vals) / np.sqrt(1 - rho))        # (n_quad,)

    # binom_mat[i, k] = P(X=k | Z=z_i) — shape (n_quad, ncs+1)
    binom_mat = scipy_binom.pmf(k_arr[None, :], ncs, p_cond[:, None])

    probs = _GH_W_NORM @ binom_mat                                             # (ncs+1,)
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()
    return probs


# ── Utility functions ─────────────────────────────────────────────────────────

def crra(gamma: float):
    """CRRA utility W^(1-γ)/(1-γ); log(W) at γ=1 (Kelly)."""
    if abs(gamma - 1.0) < 1e-9:
        return lambda w: np.log(w) if w > 0 else -np.inf
    return lambda w: (w ** (1 - gamma) / (1 - gamma)) if w > 0 else -np.inf


def cara(alpha: float):
    """CARA utility 1 − exp(−α(W−1)).  Note: fixed-fraction is NOT the true
    CARA optimum (which is a fixed dollar amount); this shows the best
    constrained fixed-fraction under CARA."""
    return lambda w: 1.0 - np.exp(-alpha * (w - 1.0))


# ── Optimisation ──────────────────────────────────────────────────────────────

def expected_utility(probs: np.ndarray, ncs: int, s: float, util_fn) -> float:
    k_arr = np.arange(ncs + 1)
    W_arr = 1.0 + s * (2.0 * k_arr - ncs)
    u_arr = np.array([util_fn(w) for w in W_arr])
    # Any outcome with U=-inf and positive probability makes EU=-inf
    if np.any(~np.isfinite(u_arr) & (probs > 1e-15)):
        return -np.inf
    mask = np.isfinite(u_arr)
    return float(np.dot(probs[mask], u_arr[mask]))


def optimal_fraction(probs: np.ndarray, ncs: int, s_grid: np.ndarray, util_fn) -> float:
    """Grid-search for the s in s_grid that maximises E[U(W)]."""
    best_s, best_eu = s_grid[0], -np.inf
    for s in s_grid:
        eu = expected_utility(probs, ncs, s, util_fn)
        if eu > best_eu:
            best_eu = eu
            best_s  = s
    return best_s


# ── Layout helpers ────────────────────────────────────────────────────────────

def _slider_block(label_text, slider_id, **kw):
    return html.Div(style=SECTION, children=[
        html.Label(label_text, style=LABEL),
        html.Div(
            style={'overflow': 'visible', 'paddingBottom': '32px'},
            children=[dcc.Slider(id=slider_id, **kw)],
        ),
    ])


# ── App layout ────────────────────────────────────────────────────────────────

app.layout = html.Div(
    style={'maxWidth': '1800px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1("Parallel Bets — Optimal Fraction vs Pairwise Correlation",
                style={'textAlign': 'center', 'marginBottom': '8px'}),
        html.P(
            "One-factor Gaussian copula. Correlation ρ = 0: independent bets. "
            "ρ → 1: all bets move together. "
            "Ruin boundary: s = 1/ncs (all-loss wipes out wealth).",
            style={'textAlign': 'center', 'color': '#555', 'marginBottom': '24px'},
        ),
        html.Div(
            style={'display': 'flex', 'gap': '24px'},
            children=[
                # ── Controls ──────────────────────────────────────────────
                html.Div(
                    style={'width': '30%', 'padding': '8px 20px'},
                    children=[
                        _slider_block(
                            "Win probability p", 'p-slider',
                            min=0.5, max=0.99, step=0.01, value=0.55,
                            marks={v / 100: f'{v}%' for v in range(50, 100, 10)},
                            tooltip={'placement': 'bottom', 'always_visible': True},
                        ),
                        _slider_block(
                            "Parallel bets per round (ncs)", 'ncs-slider',
                            min=1, max=20, step=1, value=5,
                            marks={i: str(i) for i in [1, 5, 10, 15, 20]},
                            tooltip={'placement': 'bottom', 'always_visible': True},
                        ),
                        _slider_block(
                            "CRRA risk-aversion γ  (γ=1 = Kelly)", 'gamma-slider',
                            min=0.1, max=5, step=0.1, value=2,
                            marks={g: str(g) for g in [0.1, 1, 2, 3, 5]},
                            tooltip={'placement': 'bottom', 'always_visible': True},
                        ),
                        _slider_block(
                            "CARA risk-aversion α  (1 − e^{−α(W−1)})", 'alpha-slider',
                            min=1, max=50, step=1, value=10,
                            marks={a: str(a) for a in [1, 10, 20, 30, 50]},
                            tooltip={'placement': 'bottom', 'always_visible': True},
                        ),
                    ],
                ),
                # ── Charts ────────────────────────────────────────────────
                html.Div(
                    style={'flex': '1'},
                    children=[
                        dcc.Loading(html.Div(id='corr-opt-plot')),
                        html.Hr(style={'margin': '12px 0'}),
                        html.P(
                            "Utility landscape at selected ρ",
                            style={**LABEL, 'marginBottom': '0'},
                        ),
                        html.Div(
                            style={'overflow': 'visible', 'paddingBottom': '32px'},
                            children=[
                                dcc.Slider(
                                    id='rho-inspect-slider',
                                    min=0, max=0.97, step=0.01, value=0,
                                    marks={round(v, 2): f'{round(v*100)}%'
                                           for v in np.linspace(0, 0.97, 6)},
                                    tooltip={'placement': 'bottom', 'always_visible': True},
                                ),
                            ],
                        ),
                        dcc.Loading(html.Div(id='corr-landscape-plot')),
                    ],
                ),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('corr-opt-plot', 'children'),
    Input('p-slider',     'value'),
    Input('ncs-slider',   'value'),
    Input('gamma-slider', 'value'),
    Input('alpha-slider', 'value'),
)
def update_optimal_chart(p, ncs, gamma, alpha):
    ncs   = max(int(ncs or 5), 1)
    gamma = float(gamma or 1.0)
    alpha = float(alpha or 10.0)

    max_s  = 1.0 / ncs - 0.002
    s_grid = np.linspace(0.001, max_s, 300)
    rho_vals = np.linspace(0.0, 0.97, 50)

    u_kelly = crra(1.0)
    u_crra  = crra(gamma)
    u_cara  = cara(alpha)

    opt_kelly, opt_crra, opt_cara = [], [], []
    for rho in rho_vals:
        probs = correlated_pmf(ncs, p, rho)
        opt_kelly.append(optimal_fraction(probs, ncs, s_grid, u_kelly) * 100)
        opt_crra.append( optimal_fraction(probs, ncs, s_grid, u_crra)  * 100)
        opt_cara.append( optimal_fraction(probs, ncs, s_grid, u_cara)  * 100)

    ruin_pct = 100.0 / ncs
    kelly_label = 'Kelly / Log (CRRA γ=1)'
    crra_label  = f'CRRA γ={gamma:.1f}' + (' = Kelly' if abs(gamma - 1) < 0.05 else '')
    cara_label  = f'CARA α={alpha:.0f}  [best fixed fraction, not true optimum]'

    fig = go.Figure(
        data=[
            go.Scatter(x=rho_vals, y=opt_kelly, mode='lines', name=kelly_label),
            go.Scatter(x=rho_vals, y=opt_crra,  mode='lines', name=crra_label,
                       line=dict(dash='dash')),
            go.Scatter(x=rho_vals, y=opt_cara,  mode='lines', name=cara_label,
                       line=dict(dash='dot')),
        ],
        layout=go.Layout(
            xaxis=dict(title='Pairwise correlation ρ', range=[0, 1],
                       tickformat='.0%'),
            yaxis=dict(title='Optimal bet size per bet (%)'),
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
            margin=dict(l=55, r=80, t=50, b=50),
        ),
    )
    fig.add_hline(
        y=ruin_pct, line_dash='dot', line_color='red', opacity=0.5,
        annotation_text=f'Ruin limit 1/ncs = {ruin_pct:.1f}%',
        annotation_position='right',
    )
    return dcc.Graph(figure=fig, style={'height': '42vh'},
                     config={'displayModeBar': False})


@app.callback(
    Output('corr-landscape-plot', 'children'),
    Input('p-slider',            'value'),
    Input('ncs-slider',          'value'),
    Input('gamma-slider',        'value'),
    Input('alpha-slider',        'value'),
    Input('rho-inspect-slider',  'value'),
)
def update_landscape_chart(p, ncs, gamma, alpha, rho):
    ncs   = max(int(ncs or 5), 1)
    gamma = float(gamma or 1.0)
    alpha = float(alpha or 10.0)
    rho   = float(rho or 0.0)

    max_s  = 1.0 / ncs - 0.002
    s_grid = np.linspace(0.001, max_s, 200)
    probs  = correlated_pmf(ncs, p, rho)

    u_kelly = crra(1.0)
    u_crra  = crra(gamma)
    u_cara  = cara(alpha)

    eu_kelly = [expected_utility(probs, ncs, s, u_kelly) for s in s_grid]
    eu_crra  = [expected_utility(probs, ncs, s, u_crra)  for s in s_grid]
    eu_cara  = [expected_utility(probs, ncs, s, u_cara)  for s in s_grid]

    kelly_label = 'Kelly / Log (CRRA γ=1)  [left]'
    crra_label  = f'CRRA γ={gamma:.1f}  [left]'
    cara_label  = f'CARA α={alpha:.0f}  [right]'

    fig = go.Figure(
        data=[
            go.Scatter(x=s_grid * 100, y=eu_kelly, mode='lines', name=kelly_label),
            go.Scatter(x=s_grid * 100, y=eu_crra,  mode='lines', name=crra_label,
                       line=dict(dash='dash')),
            go.Scatter(x=s_grid * 100, y=eu_cara,  mode='lines', name=cara_label,
                       yaxis='y2', line=dict(dash='dot')),
        ],
        layout=go.Layout(
            title=f'E[U] vs bet size at ρ = {rho:.2f}',
            xaxis=dict(title='Bet size per bet (%)'),
            yaxis =dict(title='Expected utility (CRRA)', side='left'),
            yaxis2=dict(title='Expected utility (CARA)', overlaying='y',
                        side='right', showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
            margin=dict(l=55, r=80, t=50, b=50),
        ),
    )
    return dcc.Graph(figure=fig, style={'height': '38vh'},
                     config={'displayModeBar': False})


if __name__ == '__main__':
    app.run(debug=True, port=8052)
