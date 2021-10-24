import streamlit as st
from streamlit_plotly_events import plotly_events
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

st.set_page_config(page_title="TC's Momentum Viz", layout="wide")
st.sidebar.header("TC's Momentum Viz :sunglasses:")

snsgreen, snsorange, snsred, snsblue, snsgrey = ['#55a868', '#dd8452', '#c44e52', '#4c72b0', '#8c8c8c']

level = [-0.86, -0.25, 0.25, 0.86]
color = [snsred, snsred, snsgreen, snsgreen]
opacity = [1, 0.8, 0.8, 1]
width = [1.5, 0.9, 0.9, 1.5]

@st.cache
def get_hist():
    tickers = df_groups[df_groups['groups'] == sel_group]['yf_ticker']
    px_hist = yf.download(
        tickers=tickers.to_list(),
        period="2y",
        interval="1d",
        group_by='column',
        auto_adjust=True)

    px_hist = px_hist['Close'].fillna(method='ffill')
    logret = np.log(px_hist / px_hist.shift(1))
    logret.iloc[0] = 0
    cumret = np.exp(logret.cumsum()) - 1
    logret = logret.iloc[1:]
    return px_hist, cumret, logret


def get_rol():
    rol_ret = np.exp(logret.rolling(win).sum()) - 1
    rol_vol = logret.rolling(win).std() * np.sqrt(win)
    rol_rar = rol_ret / rol_vol
    rol_rar = rol_rar.dropna()
    # HERE
    # rol_rar = pd.DataFrame(norm.cdf(rol_rar), columns=rol_rar.columns, index=rol_rar.index)
    rol_rar_avg = rol_rar.rolling(avg_win, center=True, min_periods=0).mean()
    return rol_ret, rol_vol, rol_rar, rol_rar_avg


def get_spline():
    spline = []
    derivative = []
    arrow = []

    for column in rol_rar_avg:
        f = UnivariateSpline(rol_rar_avg[column].reset_index().index, rol_rar_avg[column],
                             s=len(rol_rar_avg) * s_factor)
        spl = pd.DataFrame({column: f(rol_rar_avg[[column]].reset_index().index)})
        spline.append(spl)
        drv = pd.DataFrame({column: f.derivative()(rol_rar_avg[[column]].reset_index().index)})
        derivative.append(drv)

    spline = pd.concat(spline, axis=1)
    spline.index = rol_rar.index
    spline = spline.sort_values(spline.last_valid_index(), ascending=True, axis=1)
    derivative = pd.concat(derivative, axis=1)
    derivative.index = rol_rar.index
    arrow = pd.concat([spline.iloc[-1], (spline.iloc[-1]+derivative.iloc[-1]*tail)], axis=1, ignore_index=True).T
    arrow.index = [spline.index[-1], spline.index[-1]+pd.tseries.offsets.BusinessDay(n=tail)]

    return spline, derivative, arrow


def get_quad():
    quad = pd.DataFrame(data={'Pos Spl': [True, True, False, False],
                              'Pos Deriv': [True, False, True, False],
                              'Quadrant': ['Leading', 'Weakening', 'Improving', 'Lagging'],
                              'Trade': ['Long Call','Short Call','Short Put','Long Put'],
                              'Color': [snsgreen, snsorange, snsblue, snsred]})

    xclass = pd.concat([spline.tail(1).transpose() >= 0, derivative.tail(1).transpose() >= 0], axis=1)
    xclass.columns = ['Pos Spl', 'Pos Deriv']
    quad = pd.merge(xclass.reset_index(), quad, on=['Pos Spl', 'Pos Deriv']).set_index('index')
    return quad


def plot_all():

    fig_all = make_subplots(rows=1, cols=1,
                            shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.05, horizontal_spacing=0.02,
                            subplot_titles=('<b>TC Momentum',''))

    for column in spline:
        fig_all.add_trace(go.Scatter(x=spline[column].tail(tail),
                                     y=derivative[column].tail(tail),
                                     mode='lines',
                                     marker=dict(color=quad['Color'][column], line=dict(color='White', width=1)),
                                     opacity=0.7, showlegend=False, name=column),
                          row=1, col=1)

        fig_all.add_trace(go.Scatter(x=spline[column].tail(1),
                                     y=derivative[column].tail(1),
                                     mode='markers+text',
                                     marker=dict(color='white', size=10,
                                                 line=dict(color=quad['Color'][column], width=2)),
                                     text=column, textposition='bottom right', name=column),
                          row=1, col=1)

    spline_max = spline.tail(tail).abs().max().max()
    derivative_max = derivative.tail(tail).abs().max().max()

    fig_all.add_trace(go.Scatter(x=[-spline_max, spline_max, -spline_max, spline_max],
                                 y=[-derivative_max, derivative_max, derivative_max, -derivative_max],
                                 mode='markers', opacity=0),
                      row=1, col=1)

    for l, c, o, w in zip(level, color, opacity, width):
        fig_all.add_vline(x=l, line_color=c, opacity=o, line_width=w, line_dash="dash", row=1, col=1)

    fig_all.update_layout(showlegend=False, margin=dict(l=0, r=10, t=50, b=30), plot_bgcolor='#f0f2f6',
                          width=500, height=650)  #paper_bgcolor='lightyellow'

    fig_all.update_xaxes(zerolinecolor='white', zerolinewidth=3)
    fig_all.update_yaxes(zerolinecolor='white', zerolinewidth=3)

    return fig_all


def plot_one():

    fig_one = make_subplots(rows=2, cols=1, row_heights=[3, 2],
                            shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.05, horizontal_spacing=0.02,
                            subplot_titles=(f'<b>Close prices for {names.loc[tkr]["name"]}', f'<b>Rolling Sharpe Ratio & TC Momentum ({win} days)'))

    fig_one.add_trace(go.Scatter(x=px_hist.index, y=px_hist[tkr],
                                 mode='lines', line=dict(color=snsblue, width=3),
                                 name='Close Price'),
                      row=1, col=1)

    fig_one.add_trace(go.Scatter(x=px_hist.index, y=px_hist[tkr].rolling(win).mean(),
                                 mode='lines', line=dict(color=snsgrey, width=1.5), opacity=0.8,
                                 name='Close Price'),
                      row=1, col=1)

    fig_one.add_trace(go.Scatter(x=arrow.index, y=px_hist[tkr].shift(win).iloc[-1] * (1 + arrow[tkr] * rol_vol[tkr].iloc[-1]),
                                 mode='lines',  line=dict(color=quad['Color'][tkr], width=5), opacity=0.5,
                                 name='Close Price'),
                      row=1, col=1)

    # fig_one.add_trace(go.Scatter(x=arrow.index, y=px_hist[tkr].shift(win).iloc[-1] * (1 + arrow[tkr] * rol_vol[tkr].iloc[-1]),
    #                              mode='markers',  line=dict(color=quad['Color'][tkr], width=3),
    #                              marker=dict(color='white', size=10,
    #                                          line=dict(color=quad['Color'][tkr], width=2)),
    #                              name='Close Price'),
    #                   row=1, col=1)

    # fig_one.add_trace(go.Scatter(x=px_hist.index, y=px_hist[tkr].shift(win)*(1+spline[tkr]*rol_vol[tkr].iloc[-1]),
    #                              mode='lines', line=dict(color='red', width=3),
    #                              name='Spline'),
    #                   row=1, col=1)

    fig_one.add_trace(go.Scatter(x=rol_rar.index, y=rol_rar[tkr],
                                    mode='lines', line=dict(color=snsgrey, width=2), opacity=0.8,
                                    name=f'Rolling Sharpe Ratio {win} trading days', fill='tozeroy'),
                      row=2, col=1)

    fig_one.add_trace(go.Scatter(x=spline.index, y=spline[tkr],
                                    mode='lines', line=dict(color=snsgrey, width=2),
                                    name='TC Momentum'),
                      row=2, col=1)

    fig_one.add_trace(go.Scatter(x=spline.tail(tail).index, y=spline[tkr].tail(tail),
                                    mode='lines', line=dict(color=quad['Color'][tkr], width=4),
                                    name='TC Momentum'),
                      row=2, col=1)

    fig_one.add_trace(go.Scatter(x=arrow.index, y=arrow[tkr],
                                    mode='lines', line=dict(color=quad['Color'][tkr], width=5), opacity=0.5,
                                    name='TC Momentum'),
                      row=2, col=1)

    fig_one.add_trace(go.Scatter(x=spline.tail(1).index, y=spline[tkr].tail(1),
                                    mode='markers', line=dict(color=quad['Color'][tkr], width=3),
                                    marker=dict(color='white', size=10,
                                             line=dict(color=quad['Color'][tkr], width=2)),
                                    name='TC Momentum'),
                      row=2, col=1)


    for l, c, o, w in zip(level, color, opacity, width):
        fig_one.add_hline(y=l, line_color=c, opacity=o, line_width=w, line_dash="dash", row=2, col=1)

    fig_one.update_layout(showlegend=False, margin=dict(l=0, r=10, t=50, b=30), plot_bgcolor='#f0f2f6',
                          width=880, height=650)    # paper_bgcolor = 'lightyellow'

    fig_one.update_xaxes(zerolinecolor='white', zerolinewidth=3)
    fig_one.update_yaxes(zerolinecolor='white', zerolinewidth=3)
    fig_one.update_yaxes(range=[-5,5], row=2, col=1)

    return fig_one


# BODY

df_groups = pd.read_csv('yf_groups.csv')
sel_group = st.sidebar.selectbox('Choose group',df_groups.groups.unique())
names = df_groups[df_groups['groups']==sel_group][['yf_ticker','name']]
names = names.set_index('yf_ticker')
# st.write(names)


param_expander = st.sidebar.expander(label='Customize TC Mom. parameters')
with param_expander:
    win = st.select_slider('Calculation win', options=[10, 21, 63, 126], value=63)
    avg_win = st.slider('Averaging win', min_value=1, max_value=63, value=5)
    s_factor = st.slider('Smoothing factor', 0.0, 0.2, 0.05)
    tail = st.slider('Tail length', 0, 21, 10)

px_hist, cumret, logret = get_hist()
rol_ret, rol_vol, rol_rar, rol_rar_avg = get_rol()
spline, derivative, arrow = get_spline()
quad = get_quad()

rec = pd.concat([quad[['Trade','Color']], arrow.iloc[-1] - spline.iloc[-1]], axis=1)
rec.columns = ['Trade','Color','Improvement']
rec['Score'] = abs(rec['Improvement'])
rec = rec.sort_values('Score', ascending=False)
rec_dict = rec.to_dict()
tkr = st.selectbox('Recommendations:', options=rec.index,
                           format_func=lambda x: f"{x} - {rec_dict['Trade'][x]} on {names.loc[x]['name']} (Score={rec_dict['Score'][x]:.1f})")



col1, col2 = st.columns([3, 5])
with col1:
    fig_all = plot_all()
    st.plotly_chart(fig_all)

with col2:
    fig_one = plot_one()
    st.plotly_chart(fig_one)