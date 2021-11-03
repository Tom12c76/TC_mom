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
from scipy.optimize import minimize

st.set_page_config(page_title="TC's Momentum Viz", layout="wide")
st.sidebar.header("TC's Momentum Viz :sunglasses:")

snsgreen, snsorange, snsred, snsblue, snsgrey = ['#55a868', '#dd8452', '#c44e52', '#4c72b0', '#8c8c8c']

riskfree = 0.005

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
    if len(exp_dates) == 0:
        st.error(f'Holy :poop: no options available for ticker **{tkr}** on Yahoo!')
        st.stop()
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
    opt_hist['bs'] = opt_hist.apply(lambda row: BlackSholes(strategy, row[tkr], strike, row.cd2e / 365, riskfree, solver_vol), axis=1)
    return opt_hist, solver_vol


def get_fig():
    fig = go.Figure()

    global rol_vol

    col_title_2 = f'<b>Current P&L = {pnl_last:,.0f}'
    fig = make_subplots(rows=1, cols=3, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.05, horizontal_spacing=0.01,
                        row_heights=[1], column_widths=[8, 2, 0.5],
                        column_titles=[col_title_1, col_title_2, '<b>Prob'],
                        subplot_titles = ('', '', '', '<b>Rol Ret / Rol Vol / Backtest'))

    fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist[ticker],
                             name=ticker+' close price', connectgaps=True, line={'color': snsblue, 'width': 2.5}, opacity=1),
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

    fig.add_trace(go.Scatter(x=arrow.index, y=px_hist[tkr].shift(win).iloc[-1] * (1 + arrow[tkr] * rol_vol[tkr].iloc[-1]),
                                 mode='lines',  line=dict(color=quad['Color'][tkr], width=5), opacity=0.5,
                                 name='Close Price'),
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

        # fig.add_trace(go.Scatter(x=[stock_hist.index.min(), exp_date],
        #                          y=[l / stock_hist[ticker].iloc[-1] - 1, l / stock_hist[ticker].iloc[-1] - 1],
        #                          showlegend=False, mode='lines', opacity=o,
        #                          line={'color': color, 'width': width, 'dash': 'dashdot'}),
        #               row=2, col=1)

        fig.add_trace(go.Scatter(x=[0, max(bell)*1.25], y=[l, l],
                                 text=['', f'<b>p{1 - norm.cdf((abs(l / ref_price_tx_date - 1)) / (solver_vol * np.sqrt(td2e / 252))):.0%}'],
                                 textfont=dict(color=color), textposition='bottom left',
                                 showlegend=False, mode='lines+text', opacity=o, name='',
                                 line={'color': color, 'width': width, 'dash': 'dashdot'}),
                      row=1, col=3)

    fig.add_trace(go.Scatter(x=pnl, y=levels, name='payoff diagram',
                             line={'color': snsblue, 'width': 2},
                             marker=dict(color='white', line=dict(color=snsblue, width=2), size=8),
                             opacity=1),
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

    # rol_ret = stock_hist[ticker] / stock_hist[ticker].shift(td2e) - 1
    # fig.add_trace(
    #     go.Scatter(x=rol_ret.index, y=rol_ret,
    #                name=str(td2e) + ' td Rol Ret', connectgaps=True, line={'color': snsgrey, 'width': 1},
    #                fill='tozeroy'),
    #     row=2, col=1)
    #
    # rol_vol = np.log(stock_hist[ticker]/stock_hist[ticker].shift(1)).rolling(td2e).std()*np.sqrt(td2e)
    # fig.add_trace(go.Scatter(x=rol_vol.index, y=rol_vol,
    #                          name=str(td2e) + ' td Rol Vol', connectgaps=True, line={'color': snsorange, 'width': 1.25}),
    #               row=2, col=1)
    #
    # fig.add_trace(go.Scatter(x=rol_vol.index, y=-rol_vol,
    #                          name='', connectgaps=True, showlegend=False,
    #                          line={'color': snsorange, 'width': 1.25}),
    #               row=2, col=1)
    #
    # fig.add_trace(go.Scatter(x=[tx_date,tx_date], y=[solver_vol*np.sqrt(td2e/252), -solver_vol*np.sqrt(td2e/252)],
    #                          name='ivol solver', mode='markers',
    #                          line={'color': snsorange, 'width': 1.25}),
    #               row=2, col=1)


    fig.update_xaxes(zerolinecolor='grey', zerolinewidth=1.25, showticklabels=False, col=2, row=1)
    fig.update_xaxes(showticklabels=False, col=3, row=1)
    fig.update_yaxes(zerolinecolor='grey', zerolinewidth=1.25, tickformat='%', col=1, row=2)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), template='seaborn', plot_bgcolor='#F0F2F6')
    fig.update_layout(height=620, width=1200)  #, paper_bgcolor='yellow')
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    return fig


# BODY

df_groups = pd.read_csv('yf_groups.csv')
sel_group = st.sidebar.selectbox('Choose group',df_groups.groups.unique())
names = df_groups[df_groups['groups']==sel_group][['yf_ticker','name']]
names = names.set_index('yf_ticker')
# st.write(names)


# param_expander = st.sidebar.expander(label='Customize TC Mom. parameters')
# with param_expander:
#     win = st.select_slider('Calculation win', options=[10, 21, 63, 126], value=63)
#     avg_win = st.slider('Averaging win', min_value=1, max_value=63, value=5)
#     s_factor = st.slider('Smoothing factor', 0.0, 0.2, 0.05)
#     tail = st.slider('Tail length', 0, 21, 10)
win = 63
avg_win = 5
s_factor = 0.05
tail = 10

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

with st.container():
    col1, col2, col3 = st.columns((5, 3, 2))
    with col2:
        m, n = st.slider("Keep top # of securities", 1, len(rec), (1, min(len(rec),31)), 5)
        f_tickers = rec.iloc[m-1:n-1].index.tolist()
        spline = spline[f_tickers]
        derivative = derivative[f_tickers]
        arrow = arrow[f_tickers]
    with col1:
        rec_dict = rec.to_dict()
        tkr = st.selectbox('Recommendations:', options=f_tickers,
                           format_func=lambda x: f"{x} - {rec_dict['Trade'][x]} on {names.loc[x]['name']} (Score={rec_dict['Score'][x]:.1f})")
    with col3:
        st.write('')
        st.write('')
        opt = st.checkbox('Switch to Options Viz?', False)

if not opt:
    col11, col12 = st.columns([5, 3])
    with col11:
        fig_one = plot_one()
        st.plotly_chart(fig_one)

    with col12:
        fig_all = plot_all()
        st.plotly_chart(fig_all)

else:
    ticker = tkr
    stock_hist = px_hist[[tkr]]
    rol_vol = rol_vol[[tkr]]
    ref_date = stock_hist.index.max().date()
    ref_price = stock_hist.iloc[-1][0]

    exp_dates = get_exp_dates(tkr)
    exp_date = st.sidebar.selectbox('Pick exp date', exp_dates, index=len(exp_dates)-1)
    exp_date = datetime.datetime.strptime(exp_date, '%Y-%m-%d').date()

    call_chain, put_chain = get_chains(tkr)
    hide_itm = st.sidebar.checkbox('Hide ITM strikes', value=True)
    if hide_itm:
        call_chain = call_chain[call_chain['inTheMoney'] == False]
        put_chain = put_chain[put_chain['inTheMoney'] == False]

    if rec_dict['Trade'][tkr].split()[1] == 'Call':
        call_strikes_dflt = call_chain['strike'].iloc[0]
    else:
        call_strikes_dflt = []
    call_strikes = st.sidebar.multiselect('Call strikes (max 2)', call_chain['strike'],
                                          call_strikes_dflt)
    call_strikes = sorted(call_strikes)
    # format_func = lambda x: f'{x} ({x / ref_price:.0%})'
    if rec_dict['Trade'][tkr].split()[1] == 'Put':
        put_strikes_dflt = put_chain[::-1]['strike'].iloc[0]
    else:
        put_strikes_dflt = []
    put_strikes = st.sidebar.multiselect('Put strikes (max 2)', put_chain[::-1]['strike'],
                                         put_strikes_dflt)
    put_strikes = sorted(put_strikes, reverse=True)

    if rec_dict['Trade'][tkr].split()[0] == "Long":
        long_short_index = 0
    else:
        long_short_index = 1
    long_short = st.sidebar.radio('Long or Short', ('Long', 'Short'), index=long_short_index)
    if long_short == 'Long':
        ls = 1
    else:
        ls = -1

    lots = st.sidebar.select_slider('N of lots', [1, 5, 10, 20, 50, 100], 10) * ls
    mult = 100

    tx_date = ref_date
    # tx_date = st.sidebar.date_input('Transaction date override', ref_date)
    # if tx_date > ref_date:
    #     tx_date = ref_date

    cd2e = (exp_date - tx_date).days
    td2e = cd2e // 7 * 5 + cd2e % 7

    if len(call_strikes) == 1 and len(put_strikes) == 0:
        strategy = 'Call'
        lastPrice, impliedVolatility, pcf = \
            call_chain[call_chain['strike'] == call_strikes[0]][
                ['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]
        opt_hist, solver_vol = calc_opt_hist(strategy='Call', strike=call_strikes[0])
        opt_hist['breakeven'] = call_strikes[0] + opt_hist['bs']
        i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
        tx_price = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
        #                                    value=max(tx_price_suggest, 0.01))
        levels = call_strikes[0] + np.multiply([-4, 0, 1, 2, 3, 4], tx_price)
        levels_short = levels[1:]
        level_tx = call_strikes[0] + tx_price
        level_last = call_strikes[0] + lastPrice
        pnl = np.multiply([-1, -1, 0, 1, 2, 3], tx_price) * lots * mult
        pnl_short = pnl[1:]
        pnl_last = (lastPrice - tx_price) * lots * mult
        col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {call_strikes[0]}  strike  {strategy}s  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
        pos_strikes = {'call_0': call_strikes[0], 'call_1': np.NaN, 'put_0': np.NaN, 'put_1': np.NaN}
    elif len(call_strikes) == 2 and len(put_strikes) == 0:
        strategy = 'Call'
        # lower strike
        lastPrice, impliedVolatility, pcf = \
        call_chain[call_chain['strike'] == call_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[
            0]
        opt_hist_lower, solver_vol = calc_opt_hist(strategy='Call', strike=call_strikes[0])

        # higher strike
        lastPrice, impliedVolatility, pcf = \
        call_chain[call_chain['strike'] == call_strikes[1]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[
            0]
        opt_hist_higher, solver_vol = calc_opt_hist(strategy='Call', strike=call_strikes[1])

        # combine
        opt_hist = opt_hist_lower
        opt_hist['bs'] = opt_hist['bs'] - opt_hist_higher['bs']
        opt_hist['breakeven'] = call_strikes[0] + opt_hist['bs']

        i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
        tx_price = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
        #                                    value=max(tx_price_suggest, 0.01))

        strategy = 'Call Spread'
        spread = call_strikes[1] - call_strikes[0]
        lastPrice = call_chain[call_chain['strike'] == call_strikes[0]]['lastPrice'].item() - \
                    call_chain[call_chain['strike'] == call_strikes[1]]['lastPrice'].item()

        levels = [call_strikes[0] - spread, call_strikes[0], call_strikes[0] + tx_price, call_strikes[1],
                  call_strikes[1] + spread]
        levels_short = levels[1:4]
        level_tx = call_strikes[0] + tx_price
        level_last = call_strikes[0] + lastPrice
        pnl = np.multiply([-tx_price, -tx_price, 0, spread - tx_price, spread - tx_price], lots * mult)
        pnl_short = pnl[1:4]
        pnl_last = (lastPrice - tx_price) * lots * mult
        col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {call_strikes[0]}/{call_strikes[1]}  strike  {strategy}  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
        pos_strikes = {'call_0': call_strikes[0], 'call_1': call_strikes[1], 'put_0': np.NaN, 'put_1': np.NaN}
    elif len(call_strikes) == 0 and len(put_strikes) == 1:
        strategy = 'Put'
        lastPrice, impliedVolatility, pcf = \
        put_chain[put_chain['strike'] == put_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]

        opt_hist, solver_vol = calc_opt_hist(strategy='Put', strike=put_strikes[0])
        opt_hist['breakeven'] = put_strikes[0] - opt_hist['bs']
        i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
        tx_price = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
        #                                    value=max(tx_price_suggest, 0.01))
        levels = put_strikes[0] - np.multiply([-4, 0, 1, 2, 3, 4], tx_price)
        pnl = np.multiply([-1, -1, 0, 1, 2, 3], tx_price) * lots * mult
        levels_short = levels[1:]
        level_tx = put_strikes[0] - tx_price
        level_last = put_strikes[0] - lastPrice
        pnl_short = pnl[1:]
        pnl_last = (lastPrice - tx_price) * lots * mult
        col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {put_strikes[0]}  strike  {strategy}s  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
        pos_strikes = {'call_0': np.NaN, 'call_1': np.NaN, 'put_0': put_strikes[0], 'put_1': np.NaN}
    elif len(call_strikes) == 0 and len(put_strikes) == 2:
        strategy = 'Put'
        # higher strike
        lastPrice, impliedVolatility, pcf = \
            put_chain[put_chain['strike'] == put_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[
                0]
        opt_hist_higher, solver_vol = calc_opt_hist(strategy='Put', strike=put_strikes[0])
        # lower strike
        lastPrice, impliedVolatility, pcf = \
            put_chain[put_chain['strike'] == put_strikes[1]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[
                0]
        opt_hist_lower, solver_vol = calc_opt_hist(strategy='Put', strike=put_strikes[1])
        # combine
        opt_hist = opt_hist_higher
        opt_hist['bs'] = opt_hist['bs'] - opt_hist_lower['bs']
        opt_hist['breakeven'] = put_strikes[0] - opt_hist['bs']
        i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
        tx_price = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
        #                                    value=max(tx_price_suggest, 0.01))
        strategy = 'Put Spread'
        spread = put_strikes[0] - put_strikes[1]
        lastPrice = put_chain[put_chain['strike'] == put_strikes[0]]['lastPrice'].item() - \
                    put_chain[put_chain['strike'] == put_strikes[1]]['lastPrice'].item()
        levels = [put_strikes[0] + spread, put_strikes[0], put_strikes[0] - tx_price, put_strikes[1],
                  put_strikes[1] - spread]
        levels_short = levels[1:4]
        level_tx = put_strikes[0] - tx_price
        level_last = put_strikes[0] - lastPrice
        pnl = np.multiply([-tx_price, -tx_price, 0, spread - tx_price, spread - tx_price], lots * mult)
        pnl_short = pnl[1:4]
        pnl_last = (lastPrice - tx_price) * lots * mult
        col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {put_strikes[0]}/{put_strikes[1]}  strike  {strategy}  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
        pos_strikes = {'call_0': np.NaN, 'call_1': np.NaN, 'put_0': put_strikes[0], 'put_1': put_strikes[1]}
    elif len(call_strikes) == 1 and len(put_strikes) == 1:
        # higher strike
        strategy = 'Call'
        lastPrice, impliedVolatility, pcf = \
            call_chain[call_chain['strike'] == call_strikes[0]][
                ['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[0]
        opt_hist_higher, solver_vol = calc_opt_hist(strategy='Call', strike=call_strikes[0])
        # lower strike
        lastPrice, impliedVolatility, pcf = \
            put_chain[put_chain['strike'] == put_strikes[0]][['lastPrice', 'impliedVolatility', 'pcf']].values.tolist()[
                0]
        opt_hist_lower, solver_vol = calc_opt_hist(strategy='Put', strike=put_strikes[0])
        # combine
        opt_hist = opt_hist_higher
        opt_hist['bs'] = opt_hist['bs'] + opt_hist_lower['bs']
        opt_hist['breakeven'] = call_strikes[0] + opt_hist['bs']
        opt_hist['breakeven lower'] = put_strikes[0] - opt_hist['bs']
        i = opt_hist.index.get_loc(pd.to_datetime(tx_date), method='nearest')
        tx_price = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price_suggest = round(opt_hist.iloc[i]['bs'], 2)
        # tx_price = st.sidebar.number_input('Transaction  price override', min_value=0.01, max_value=None,
        #                                    value=max(tx_price_suggest, 0.01))
        if call_strikes[0] == put_strikes[0]:
            strategy = 'Straddle'
        elif call_strikes[0] > put_strikes[0]:
            strategy = 'Strangle'

        lastPrice = call_chain[call_chain['strike'] == call_strikes[0]]['lastPrice'].item() + \
                    put_chain[put_chain['strike'] == put_strikes[0]]['lastPrice'].item()
        levels = [call_strikes[0] + 2 * tx_price,
                  call_strikes[0] + tx_price,
                  call_strikes[0],
                  put_strikes[0],
                  put_strikes[0] - tx_price,
                  put_strikes[0] - 2 * tx_price]

        level_tx = call_strikes[0] + tx_price
        level_last = call_strikes[0] + lastPrice
        levels_short = levels
        pnl = np.multiply([tx_price, 0, -tx_price, -tx_price, 0, tx_price], lots * mult)
        pnl_last = (lastPrice - tx_price) * lots * mult
        pnl_short = pnl
        if strategy == "Straddle":
            col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {call_strikes[0]}  strike  {strategy}s  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
        else:
            col_title_1 = f'<b>{long_short} {lots} lots of the {exp_date:%d-%b-%y}<br>{ticker}  {put_strikes[0]}/{call_strikes[0]}  strike  {strategy}  @  {tx_price:.2f}  [now {lastPrice:.2f}]'
        pos_strikes = {'call_0': call_strikes[0], 'call_1': np.NaN, 'put_0': put_strikes[0], 'put_1': np.NaN}
    else:
        st.error('Holy :poop: I can only handle calls, puts, vertical spreads and straddles for now :man-shrugging:')
        st.stop()

    range_from = min(min(stock_hist[ticker]), min(levels))
    range_to = max(max(stock_hist[ticker]), max(levels))
    range = np.linspace(range_from, range_to, num=100)
    ref_price_tx_date = opt_hist.iloc[i][0]
    bell = norm.pdf((range / ref_price_tx_date - 1) / (solver_vol * np.sqrt(cd2e / 365)))

    col1, col2 = st.columns((8, 1))
    with col1:
        fig = get_fig()
        st.plotly_chart(fig)

    with col2:
        st.metric(ticker + ' last', f'${ref_price:.2f}')
        st.metric('option last', f'{lastPrice:.2f}')
        # st.metric('ivol yfinance', f'{impliedVolatility:.0%}')
        st.metric('ivol solver', f'{solver_vol:.0%}')
        st.metric('P&L', f'${pnl_last:,.0f}')

    st.write(f'Go to :runner: stock summary on [yahoo!] (https://finance.yahoo.com/quote/{ticker})')

    pos_details = {'ticker': ticker, 'exp_date': exp_date.isoformat(), 'long_short': long_short, 'lots': lots,
                   'strategy': strategy, 'tx_date': tx_date.isoformat(), 'tx_price': tx_price}
    df_pos = pd.DataFrame(data={**pos_details, **pos_strikes}, index=[0])
    df_pos_cols = ['ticker', 'exp_date', 'long_short', 'lots', 'strategy', 'call_0', 'call_1', 'put_0', 'put_1',
                   'tx_date', 'tx_price']
    df_pos = df_pos[df_pos_cols]
    df_pos.to_clipboard(index=False, header=True)

    st.write('')
    st.write('')
    st.write('Trade details have been copied to clipboard :clipboard:')
    st.write(df_pos)



