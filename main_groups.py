import streamlit as st
from streamlit_plotly_events import plotly_events
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import UnivariateSpline
import datetime
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
    fig_all.update_xaxes(title_text="Velocity", row=1, col=1)
    fig_all.update_yaxes(title_text="Acceleration", row=1, col=1)

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
    fig_one.update_yaxes(title_text="Velocity", row=2, col=1)

    return fig_one


def get_exp_dates(tkr):
    weekly = False
    D2Emin = 0
    D2Emax = 120
    min_day, max_day = (1, 31) if weekly else (15, 21)
    filt_dates = lambda x: (D2Emin < ((x - datetime.datetime.now()).days) < D2Emax) \
                           and (min_day <= x.day <= max_day)
    exp_dates = list(map(lambda x: str(x.date()), filter(filt_dates, pd.to_datetime(yf.Ticker(tkr).options))))
    return exp_dates


def get_chains(tkr):
    call_chain, put_chain = yf.Ticker(tkr).option_chain(str(exp_date))
    call_chain['pcf'] = 1
    put_chain['pcf'] = -1
    return call_chain, put_chain


def BlackSholes(CallPutFlag, S, X, T, r, v):
    d1 = (np.log(S/X)+(r+v*v/2)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if CallPutFlag=='Call':
        return S*norm.cdf(d1)-X*np.exp(-r*T)*norm.cdf(d2)
    else:
        return X*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)


def calc_opt_hist(strategy, strike):
    minimize_me = lambda x: np.sqrt((BlackSholes(strategy, ref_price, strike, (exp_date - ref_date).days / 365, riskfree, x) - lastPrice) ** 2)
    min_result = minimize(minimize_me, 0.15, method='Nelder-Mead')
    if min_result.success:
        solver_vol = min_result.x[0]
    else:
        st.warning('Solver could not determine impl vol!')
        st.stop()
    opt_hist = stock_hist.copy()
    opt_hist['cd2e'] = (exp_date - stock_hist.index.date)
    opt_hist['cd2e'] = opt_hist['cd2e'].dt.days
    opt_hist['bs'] = opt_hist.apply(lambda row: BlackSholes(strategy, row[0], strike, row.cd2e / 365, riskfree, solver_vol), axis=1)
    return opt_hist, solver_vol


def get_fig():
    fig = go.Figure()

    col_title_2 = f'<b>Current P&L = {pnl_last:,.0f}'
    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.05, horizontal_spacing=0.01,
                        row_heights=[2, 1], column_widths=[8, 2, 0.5],
                        column_titles=[col_title_1, col_title_2, '<b>Prob'],
                        subplot_titles = ('', '', '', '<b>Rol Ret / Rol Vol / Backtest'))

    fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist[ticker],
                             name=ticker+' close price', connectgaps=True, line={'color': snsblue, 'width': 2.5}, opacity=0.8),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=[tx_date, ref_date], y=[ref_price_tx_date, ref_price], mode='markers',
                             showlegend=False, line={'color': snsblue, 'width': 2.5}, opacity=1),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist[ticker].rolling(td2e).mean(),
                             name=str(td2e)+' td SMA',line={'color':snsgrey,'width':1}),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=opt_hist[opt_hist['cd2e'] <= (cd2e + 20)].index,
                             y=opt_hist[opt_hist['cd2e'] <= (cd2e + 20)]['breakeven'],
                             name='BS approx', connectgaps=True, mode='lines', line={'color': snsorange, 'width': 2.5},
                             opacity=0.4),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=bell, y=range, name='norm dist', mode='lines',
                             line={'color': snsblue, 'width': 0.75}, fill='tozerox'),
                  row=1, col=3)

    fig.add_trace(go.Scatter(x=[max(bell)], y=[ref_price_tx_date], mode='markers',
                             showlegend=False, line={'color': snsblue, 'width': 2.5}, opacity=1),
                  row=1, col=3)

    for l, p in zip(levels_short, pnl_short):

        width = 1.25

        if p == 0:
            color = snsgrey
            o = 1
        elif p > 0:
            color = snsgreen
            o = (p/max(pnl))*0.66 + 0.34
        else:
            color = snsred
            o = (p/min(pnl))*0.66 + 0.34

        fig.add_trace(go.Scatter(x=[tx_date, exp_date], y=[l, l], mode='lines+text', opacity=o,
                                 text=['', f'<b>{l:,.2f}  {(l/ref_price_tx_date-1):+,.1%}'],
                                 textposition='bottom center', textfont=dict(color=color), showlegend=False,
                                 line={'color': color, 'width': width, 'dash': 'dashdot'}),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=[min(pnl), max(pnl)], y=[l, l], mode='lines+text', name='',
                                 text=['', f'<b>${p:,.0f}  ({p/(tx_price*lots*mult)*ls:.1f}x)'], textposition='bottom center', textfont=dict(color=color),
                                 showlegend=False, opacity=o, line={'color': color, 'width': width, 'dash': 'dashdot'}),
                      row=1, col=2)

        fig.add_trace(go.Scatter(x=[stock_hist.index.min(), exp_date],
                                 y=[l / stock_hist[ticker].iloc[-1] - 1, l / stock_hist[ticker].iloc[-1] - 1],
                                 showlegend=False, mode='lines', opacity=o,
                                 line={'color': color, 'width': width, 'dash': 'dashdot'}),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=[0, max(bell)*1.25], y=[l, l],
                                 text=['', f'<b>p{1 - norm.cdf((abs(l / ref_price_tx_date - 1)) / (solver_vol * np.sqrt(td2e / 252))):.0%}'],
                                 textfont=dict(color=color), textposition='bottom left',
                                 showlegend=False, mode='lines+text', opacity=o, name='',
                                 line={'color': color, 'width': width, 'dash': 'dashdot'}),
                      row=1, col=3)

    fig.add_trace(go.Scatter(x=pnl, y=levels, name='payoff diagram',
                             line={'color': snsblue, 'width': 2}, opacity=0.7),
                  row=1, col=2)

    padding = (max(pnl)-min(pnl))/5
    fig.add_trace(go.Scatter(x=[min(pnl), max(pnl)+padding],
                             y=[levels[0], levels[0]],
                             name='', opacity=0.0, showlegend=False),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=[tx_date, ref_date], y=[level_tx, level_last],
                             showlegend=False, mode='markers', marker=dict(color=snsorange), opacity=1),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=[0, pnl_last], y=[level_tx, level_last],
                             showlegend=False, mode='lines+markers', marker=dict(color=snsorange), opacity=1),
                  row=1, col=2)

    if strategy in ['Straddle', 'Strangle']:
        fig.add_trace(go.Scatter(x=[tx_date, ref_date], y=[put_strikes[0]-tx_price, put_strikes[0]-lastPrice],
                                 showlegend=False, mode='markers', marker=dict(color=snsorange), opacity=1),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, pnl_last], y=[put_strikes[0]-tx_price, put_strikes[0]-lastPrice],
                                 showlegend=False, mode='lines+markers', marker=dict(color=snsorange), opacity=1),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=opt_hist[opt_hist['cd2e'] <= (cd2e + 20)].index,
                                 y=opt_hist[opt_hist['cd2e'] <= (cd2e + 20)]['breakeven lower'],
                                 name='BS approx', connectgaps=True, mode='lines',
                                 line={'color': snsorange, 'width': 2.5}, opacity=0.4),
                      row=1, col=1)

    rol_ret = stock_hist[ticker] / stock_hist[ticker].shift(td2e) - 1
    fig.add_trace(
        go.Scatter(x=rol_ret.index, y=rol_ret,
                   name=str(td2e) + ' td Rol Ret', connectgaps=True, line={'color': snsgrey, 'width': 1},
                   fill='tozeroy'),
        row=2, col=1)

    rol_vol = np.log(stock_hist[ticker]/stock_hist[ticker].shift(1)).rolling(td2e).std()*np.sqrt(td2e)
    fig.add_trace(go.Scatter(x=rol_vol.index, y=rol_vol,
                             name=str(td2e) + ' td Rol Vol', connectgaps=True, line={'color': snsorange, 'width': 1.25}),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=rol_vol.index, y=-rol_vol,
                             name='', connectgaps=True, showlegend=False,
                             line={'color': snsorange, 'width': 1.25}),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=[tx_date,tx_date], y=[solver_vol*np.sqrt(td2e/252), -solver_vol*np.sqrt(td2e/252)],
                             name='ivol solver', mode='markers',
                             line={'color': snsorange, 'width': 1.25}),
                  row=2, col=1)


    fig.update_xaxes(zerolinecolor='grey', zerolinewidth=1.25, col=2, row=1)
    fig.update_yaxes(zerolinecolor='grey', zerolinewidth=1.25, tickformat='%', col=1, row=2)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), template='seaborn', plot_bgcolor='#F0F2F6')
    fig.update_layout(height=700, width=1200)  #, paper_bgcolor='yellow')
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    return fig


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

ref_date = px_hist.index.max().date()
ref_price = px_hist.iloc[-1][0]

rec = pd.concat([quad[['Trade','Color']], arrow.iloc[-1] - spline.iloc[-1]], axis=1)
rec.columns = ['Trade','Color','Improvement']
rec['Score'] = abs(rec['Improvement'])
rec = rec.sort_values('Score', ascending=False)

m, n = st.sidebar.slider("Filter # of securities", 1, len(rec), (1, len(rec)), 5)
f_tickers = rec.iloc[m-1:n-1].index.tolist()
spline = spline[f_tickers]
derivative = derivative[f_tickers]
arrow = arrow[f_tickers]

rec_dict = rec.to_dict()
tkr = st.selectbox('Recommendations:', options=f_tickers,
                   format_func=lambda x: f"{x} - {rec_dict['Trade'][x]} on {names.loc[x]['name']} (Score={rec_dict['Score'][x]:.1f})")


opt = st.sidebar.checkbox('Options?', False)
if not opt:
    col1, col2 = st.columns([3, 5])
    with col1:
        fig_all = plot_all()
        st.plotly_chart(fig_all)

    with col2:
        fig_one = plot_one()
        st.plotly_chart(fig_one)
else:
    st.header('welcome to options')

    ticker = tkr
    stock_hist = px_hist
    st.write(stock_hist)
    ref_date = stock_hist.index.max().date()
    ref_price = stock_hist.iloc[-1][0]
    st.write(ref_date, ref_price)

    exp_dates = get_exp_dates(tkr)
    exp_date = st.sidebar.selectbox('Pick exp date', exp_dates, index=len(exp_dates)-1)
    exp_date = datetime.datetime.strptime(exp_date, '%Y-%m-%d').date()
    call_chain, put_chain = get_chains(tkr)
    hide_itm = st.sidebar.checkbox('Hide ITM strikes', value=True)
    if hide_itm:
        call_chain = call_chain[call_chain['inTheMoney'] == False]
        put_chain = put_chain[put_chain['inTheMoney'] == False]
    st.write(call_chain, put_chain)
    st.write(BlackSholes("Call", 100, 100, 1, 0.05, 0.2))

