#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 01 13:26:28 2024

version 1.1

@author: zqli
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import backtest
import numpy as np
import numba
import time
import scipy
from copy import deepcopy




# region Auxiliary functions

def spearman_corr(a,b,axis=1):
    
    """
    
    a: pd.DataFrame
    b: pd.DataFrame
    
    return: spearman rank correlation among certain axis
    
    """
    
    if a.shape != b.shape:
        
        raise ValueError(f'incompatible shapes: {a.shape} vs {b.shape}')
    
    if axis == 0:
        idx = a.columns
    elif axis == 1:
        idx = a.index
    a = a.values
    b = b.values
    
    
    def sort(x):
        
        # count non_nan number, filter out any larger than it in the sorting below
        non_nan = np.repeat(
            np.nansum(np.where(np.isnan(x),np.nan,1),axis=axis).T,x.shape[1]
            ).reshape(x.shape[0],x.shape[1])
        
        
        
        # spearman corr result is the same for ascending or descending
        rank0 = x.argsort(axis=axis)
        rank = np.empty_like(rank0,dtype=int)
        
        if axis == 1:
            rank[np.arange(x.shape[0])[:,None],rank0] = np.arange(x.shape[1])
        elif axis == 0:
            rank[rank0,np.arange(x.shape[1])] = np.arange(x.shape[0])[:,None]
        
        rank = np.where(rank<non_nan,rank,np.nan)
        
        # used in corrlation calculation: n
        n = non_nan.T[0]
        
        return rank, n
        
    
    # spearman corrlation formula: 1 - 6*sum(d^2) / (n*(n**2-1))
    # d: difference in rank
    
    
    a,n = sort(a)
    b,_ = sort(b)
    
    
    
    d = np.abs(a-b)
    r = 1-np.divide((6 * np.nansum(d**2,axis=axis)), n*(n**2-1))
        
    return pd.Series(r,index=idx)



def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    

    return df.rolling(window).sum()

def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()

def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()

def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)

def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """


    return scipy.stats.rankdata(na)[-1]

def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    # Convert DataFrame to numpy array for faster operations
    
    return df.rolling(window).rank()

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """

    # cumprod = df.values.cumprod(axis=0)

    # if cumprod.shape[0] > window:
    #     cumprod[window:] = np.divide(cumprod[window:], cumprod[:(cumprod.shape[0]-window)])

    # df = pd.DataFrame(cumprod, index=df.index, columns=df.columns)

    # return df


    return df.rolling(window).apply(rolling_prod, engine='numba', raw=True)

    
    


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()



def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()



def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)



def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)



def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    return df.rank(axis=1)
    # return df.rank(pct=True)

def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """

    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: DataFrame with the index of the maximum value in each rolling window
    """
    # Convert DataFrame to numpy array for faster operations
    values = df.values
    
    # Create a 2D array of rolling windows
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    
    # Find the argmax for each window
    argmax_indices = np.argmax(windows, axis=2)
    
    # Adjust indices to match the original DataFrame
    result = pd.DataFrame(argmax_indices + 1, index=df.index[window-1:], columns=df.columns)
    
    # Pad the result with NaN for the first window-1 rows
    return pd.concat([pd.DataFrame(index=df.index[:window-1], columns=df.columns), result])

def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    """
    # Convert DataFrame to numpy array for faster operations
    values = df.values
    
    # Create a 2D array of rolling windows
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    
    # Find the argmax for each window
    argmin_indices = np.argmin(windows, axis=2)
    
    # Adjust indices to match the original DataFrame
    result = pd.DataFrame(argmin_indices + 1, index=df.index[window-1:], columns=df.columns)
    
    # Pad the result with NaN for the first window-1 rows
    return pd.concat([pd.DataFrame(index=df.index[:window-1], columns=df.columns), result])


def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # ------------ original code :: start ------------------
    # Clean data
    # if df.isnull().values.any():
    #     df.fillna(method='ffill', inplace=True)
    #     df.fillna(method='bfill', inplace=True)
    #     df.fillna(value=0, inplace=True)
    # na_lwma = np.zeros_like(df)
    # na_lwma[:period, :] = df.iloc[:period, :] 
    # na_series = df#.as_matrix()
# 
    # divisor = period * (period + 1) / 2
    # y = (np.arange(period) + 1) * 1.0 / divisor
    # # Estimate the actual lwma with the actual close.
    # # The backtest engine should assure to be snooping bias free.
    # for row in range(period - 1, df.shape[0]):
    #     x = na_series[row - period + 1: row + 1, :]
    #     na_lwma[row, :] = (np.dot(x.T, y))
    # return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])
    # ------------ original code :: end ----------------------

    # n = period
    # decay_weights = np.arange(1,n+1,1) 
    # decay_weights = decay_weights / decay_weights.sum()

    # def func_decaylinear(na):
    #     return (na * decay_weights).sum()
    # return df.rolling(period).apply(func_decaylinear,
    #                                 engine='numba',
    #                                 raw=True)



    n = period
    decay_weights = np.arange(1, n+1, 1)
    decay_weights = decay_weights / decay_weights.sum()

    # Create rolling windows
    rolling_windows = np.lib.stride_tricks.sliding_window_view(df.values, n, axis=0)

    # Reshape decay_weights for broadcasting
    decay_weights = np.tile(decay_weights,(rolling_windows.shape[0],rolling_windows.shape[1],1))

    # Apply decay weights and sum
    result = np.sum(rolling_windows * decay_weights, axis=2)
    nan_top = np.full((n-1,result.shape[1]),np.nan)
    result = np.concatenate((nan_top,result),axis=0)

    # Create DataFrame with the result
    return pd.DataFrame(result, index=df.index, columns=df.columns)
# endregion



class Alphas(object):
    def __init__(self, data):

        data_org = data.copy()
        idx = pd.MultiIndex.from_arrays([data_org['ticker'], data_org['open_time']], names=['ticker', 'datetime'])
        data = data_org.set_index(idx).rename(columns={'market_cap': 'marketcap'})
        data['sector'] = np.zeros(len(data))
        by_ticker = data.groupby(level='ticker')
        T = [1, 2, 3, 4, 5, 10, 21, 42, 63, 126, 252]
        for t in T:
            data[f'ret_{t:02}'] = by_ticker['close'].pct_change(t)

        data['ret_fwd'] = by_ticker['ret_01'].shift(-1)
        data.rename(columns={'ret_01': 'returns'}, inplace=True)
        self.open = data['open'].unstack('ticker')
        self.high = data['high'].unstack('ticker')
        self.low = data['low'].unstack('ticker')
        self.close = data['close'].unstack('ticker')
        self.volume = data['volume'].unstack('ticker')
        self.quote_volume = data['quote_volume'].unstack('ticker')
        self.returns = data['returns'].unstack('ticker')
        self.taker_buy_quote_volume = data['taker_buy_quote_volume'].unstack('ticker')
        self.upshadow = (self.high - np.maximum(self.open,self.close)) / self.high
        self.downshadow = (np.minimum(self.open,self.close) - self.low) / self.low
        self.w_upshadow = (self.high - self.close) / self.high
        self.w_downshadow = (self.close - self.low) / self.low
        self.shortcut = 2 * (self.high - self.low) - (self.open - self.close).abs()
        # self.LIX = np.log(self.quote_volume / (self.high - self.low))
        self.vwap = self.open.add(self.high).add(self.low).add(self.close).div(4)        #self.adv20 = self.v.rolling(20).mean()

        self.avg_price = self.open.add(self.high).add(self.low).add(self.close).div(4) 
        
        self.btc_close = self.close['BTCUSDT']
        self.btc_open = self.open['BTCUSDT']


        #非量价，若只有k线数据则跳过

        try:
            self.marketcap = data['marketcap'].unstack('ticker')
        except:
            pass

        try:
            self.circulating_supply = data['circulating_supply'].unstack('ticker')
            self.total_supply = data['total_supply'].unstack('ticker')
            self.funding_rate = data['funding_rate'].unstack('ticker')
            self.open_interest = data['sum_open_interest'].unstack('ticker')
            self.open_interest_value = data['sum_open_interest_value'].unstack('ticker')
            self.count_toptrader_long_short_ratio = data['count_toptrader_long_short_ratio'].unstack('ticker')
            self.sum_toptrader_long_short_ratio = data['sum_toptrader_long_short_ratio'].unstack('ticker')
            self.count_long_short_ratio = data['count_long_short_ratio'].unstack('ticker')
            self.taker_long_short_vol_ratio = data['sum_taker_long_short_vol_ratio'].unstack('ticker')
            self.turnover = self.volume / self.circulating_supply
            self.FDV = self.total_supply * self.close

        except:
            pass

        ###############################

        # self.open = df_data['S_DQ_OPEN'] 
        # self.high = df_data['S_DQ_HIGH'] 
        # self.low = df_data['S_DQ_LOW']   
        # self.close = df_data['S_DQ_CLOSE'] 
        # self.volume = df_data['S_DQ_VOLUME']*100 
        # self.returns = df_data['S_DQ_PCTCHANGE'] 
        # self.vwap = (df_data['S_DQ_AMOUNT']*1000)/(df_data['S_DQ_VOLUME']*100+1) 



# 38 个因子




#动量反转因子
    def alpha_mmt_normal_nP(self, n=10):
        """
        mmt_normal_M = 过去一个月收益率
        """
        return self.close.pct_change(n).stack()

#def mmt_normal_1D(df):
#    return mmt_normal_nP(df, 1).rename(columns={'mmt_normal_M': 'mmt_normal_1D'})
#
#def mmt_normal_3D(df):
#    return mmt_normal_nP(df, 3).rename(columns={'mmt_normal_M': 'mmt_normal_3D'})
#
#def mmt_normal_7D(df):
#    return mmt_normal_nP(df, 7).rename(columns={'mmt_normal_M': 'mmt_normal_7D'})
#
#def mmt_normal_14D(df):
#    return mmt_normal_nP(df, 14).rename(columns={'mmt_normal_M': 'mmt_normal_14D'})
#
#def mmt_normal_1M(df):
#    return mmt_normal_nP(df, 30).rename(columns={'mmt_normal_M': 'mmt_normal_1M'})
#
#def mmt_normal_3M(df):
#    return mmt_normal_nP(df, 90).rename(columns={'mmt_normal_M': 'mmt_normal_3M'})
#
#def mmt_normal_6M(df):
#    return mmt_normal_nP(df, 180).rename(columns={'mmt_normal_M': 'mmt_normal_6M'})
#
#
#def mmt_normal_A(df):
#    """
#    mmt_normal_A = 过去一年收益率 - 过去一个月收益率
#    """
#
#    df_daily = df.copy()
#    df_daily['1Y_ret'] = df_daily.groupby('ticker')['close'].pct_change(365)
#    df_daily['1M_ret'] = df_daily.groupby('ticker')['close'].pct_change(30)
#    df_daily['mmt_normal_A'] = df_daily['1Y_ret'] - df_daily['1M_ret']
#    # 对df进行整理
#    df_daily.sort_values('open_time', inplace=True)
#    df_daily.rename(columns={'open_time':'datetime'}, inplace=True)
#    idx = pd.MultiIndex.from_arrays([df_daily['datetime'], df_daily['ticker']], names=['datetime', 'ticker'])
#    df_daily = df_daily.set_index(idx)
#    return df_daily[['mmt_normal_A']]

    def alpha_mmt_avg_nP(self, n=10):
        """
        mmt_avg_M = 当期收盘价 / 过去一个月交易日均价
        """


        return (self.close / self.close.rolling(n).mean()).stack()





#def mmt_avg_3D(df):
#    a = mmt_avg_nP(df, 3)
#    a.columns = ['mmt_avg_3D']
#    return a
#
#def mmt_avg_7D(df):
#    a = mmt_avg_nP(df, 7)
#    a.columns = ['mmt_avg_7D']
#    return a
#
#def mmt_avg_14D(df):
#    a = mmt_avg_nP(df, 14)
#    a.columns = ['mmt_avg_14D']
#    return a
#
#def mmt_avg_1M(df):
#    a = mmt_avg_nP(df, 30)
#    a.columns = ['mmt_avg_1M']
#    return a
#
#def mmt_avg_3M(df):
#    a = mmt_avg_nP(df, 90)
#    a.columns = ['mmt_avg_3M']
#    return a
#
#def mmt_avg_6M(df):
#    a = mmt_avg_nP(df, 180)
#    a.columns = ['mmt_avg_6M']
#    return a
#
#
#def mmt_avg_A(df):
#    """
#    mmt_avg_A = 一个月前收盘价 / 过去一年交易日均价
#    """
 
#    df = df.copy()
#    a = pd.pivot(df, columns='ticker', index='open_time', values='close').shift(30).stack().to_frame(name='close_one_month_ago')
#    a['rolling_mean'] = pd.pivot(df, columns='ticker', index='open_time', values='close').rolling(365).mean().stack()
#    a['mmt_avg_A'] = a['close_one_month_ago'] / a['rolling_mean']
#    
#    df_daily = a
#
#    df_daily = a.rename_axis(['datetime', 'ticker'], axis=0)
#    
#    return df_daily[['mmt_avg_A']]


#################################################################
    def alpha_mmt_intra_nP(self, m=20, n=10):
        """
        mmt_intraday_M = 过去m个时间单位中的n个时间单位内涨跌幅之和
        """
        
        diff = self.close.diff(n)

        return (diff.rolling(m).sum()).stack()


    def alpha_mmt_intra_abs_nP(self, m=20, n=10):
        """
        mmt_intraday_M = 过去m个时间单位中的n个时间单位内涨跌幅绝对值之和
        """
        
        diff = self.close.diff(n).abs()

        return (diff.rolling(m).sum()).stack()

#def mmt_intraday_1D(df):
#    a = mmt_intraday_nP(df, 1)
#    a.columns = ['mmt_intraday_1D']
#    return a
#
#def mmt_intraday_3D(df):
#    a = mmt_intraday_nP(df, 3)
#    a.columns = ['mmt_intraday_3D']
#    return a
#
#def mmt_intraday_7D(df):
#    a = mmt_intraday_nP(df, 7)
#    a.columns = ['mmt_intraday_7D']
#    return a
#
#def mmt_intraday_14D(df):
#    a = mmt_intraday_nP(df, 14)
#    a.columns = ['mmt_intraday_14D']
#    return a
#
#def mmt_intraday_1M(df):
#    a = mmt_intraday_nP(df, 30)
#    a.columns = ['mmt_intraday_1M']
#    return a
#
#def mmt_intraday_3M(df):
#    a = mmt_intraday_nP(df, 90)
#    a.columns = ['mmt_intraday_3M']
#    return a
#
#def mmt_intraday_6M(df):
#    a = mmt_intraday_nP(df, 180)
#    a.columns = ['mmt_intraday_6M']
#    return a

#def mmt_intraday_A(df):
#    """
#    mmt_intraday_A = 过去一年的日内涨跌幅之和 - 过去一个月日内涨鉄幅之和
#    """
#    df = df.copy()
#    df['open_time_date'] = df['open_time'].dt.date
#    gb = df.groupby('ticker')
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        a = group.groupby('open_time_date').tail(1)
#        a['1DPM'] = a.close.pct_change(1)
#        a['mmt_intraday_M'] = a['1DPM'].rolling(30).sum()
#        a['mmt_intraday_A'] = a['1DPM'].rolling(365).sum()
#        a['mmt_intraday_A'] = a['mmt_intraday_A'] - a['mmt_intraday_M']
#        df_daily = pd.concat([df_daily, a])
#    df_daily.sort_values('open_time_date', inplace=True)
#    idx = pd.MultiIndex.from_arrays([df_daily['open_time_date'], df_daily['ticker']], names=['datetime', 'ticker'])
#    df_daily = df_daily.set_index(idx)
#        
#    return df_daily[['mmt_intraday_A']]

    def alpha_mmt_route_nP(self,n=10):
        """
        mmt_route_M = 过去n个时间单位内收益率 / 过去n个时间单位内日度张跌幅绝对值之和
        """
        diff = self.close.diff(n).abs()
        diff = diff.rolling(n).sum()
        ret = self.close.pct_change(n)


        return (ret/diff).stack()

#def mmt_route_1D(df):
#    return mmt_route_nP(df, 1).rename(columns={'mmt_route_M': 'mmt_route_1D'})
#
#def mmt_route_3D(df):
#    return mmt_route_nP(df, 3).rename(columns={'mmt_route_M': 'mmt_route_3D'})
#
#def mmt_route_7D(df):
#    return mmt_route_nP(df, 7).rename(columns={'mmt_route_M': 'mmt_route_7D'})
#
#def mmt_route_15D(df):
#    return mmt_route_nP(df, 15).rename(columns={'mmt_route_M': 'mmt_route_15D'})
#
#def mmt_route_1M(df):
#    return mmt_route_nP(df, 30).rename(columns={'mmt_route_M': 'mmt_route_1M'})
#
#def mmt_route_3M(df):
#    return mmt_route_nP(df, 90).rename(columns={'mmt_route_M': 'mmt_route_3M'})
#
#def mmt_route_6M(df):
#    return mmt_route_nP(df, 180).rename(columns={'mmt_route_M': 'mmt_route_6M'})



#def mmt_route_A(df):
#    """
#    mmt_route_A = 过去1年内收益率 / 过去1年内日度张跌幅绝对值之和
#    """
#    df = df.copy()
#    df['open_time_date'] = df['open_time'].dt.date
#    gb = df.groupby('ticker')
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        a = group.groupby('open_time_date').tail(1)
#        a['1DPM'] = a.close.pct_change(1)
#        a['1DPM_abs'] = abs(a['1DPM'])
#        a['1YRollingSum_1DPM_abs'] = a['1DPM_abs'].rolling(365).sum()
#        a['close_1YAge'] = a['close'].shift(365)
#        a['ret_0-365'] = a['close']/a['close_1YAge'] - 1
#        a['mmt_route_A'] = a['ret_0-365'] / a['1YRollingSum_1DPM_abs']
#        df_daily = pd.concat([df_daily, a])
#    df_daily.sort_values('open_time_date', inplace=True)
#    idx = pd.MultiIndex.from_arrays([df_daily['open_time_date'], df_daily['ticker']], names=['datetime', 'ticker'])
#    df_daily = df_daily.set_index(idx)
#
#    return df_daily[['mmt_route_A']]

    def alpha_mmt_discrete_nP(self, n=10):
        """
        mmt_discrete_M = 过去n个时间单位内，上涨时间占比 - 下跌时间占比
        """

        high = self.close.diff()
        high = pd.DataFrame(np.where(high>0,1,0),columns=self.close.columns,index=self.close.index)
        
        # (high/window - low/window) = (high-low) / window = (high-10+high) / window

        return (high.rolling(n).sum()*2 / n).stack()

#def mmt_discrete_3D(df):
#    return mmt_discrete_nP(df, 3).rename(columns={'mmt_discrete_M':'mmt_discrete_3D'})
#
#def mmt_discrete_7D(df):
#    return mmt_discrete_nP(df, 7).rename(columns={'mmt_discrete_M':'mmt_discrete_7D'})
#
#def mmt_discrete_15D(df):
#    return mmt_discrete_nP(df, 15).rename(columns={'mmt_discrete_M':'mmt_discrete_15D'})
#
#def mmt_discrete_1M(df):
#    return mmt_discrete_nP(df, 30).rename(columns={'mmt_discrete_M':'mmt_discrete_1M'})
#
#def mmt_discrete_3M(df):
#    return mmt_discrete_nP(df, 90).rename(columns={'mmt_discrete_M':'mmt_discrete_3M'})
#
#def mmt_discrete_6M(df):
#    return mmt_discrete_nP(df, 180).rename(columns={'mmt_discrete_M':'mmt_discrete_6M'})


#def mmt_discrete_A(df):
#    """
#    mmt_discrete_A = 过去1年内，上涨天教占比 - 下跌天数占比
#    """
#    df = df.copy()
#    df['open_time_date'] = df['open_time'].dt.date
#    gb = df.groupby('ticker')
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        a = group.groupby('open_time_date').tail(1)
#        a['1DPM'] = a.close.pct_change(1)
#        a['increase_day_ratio'] = a['1DPM'].rolling(365).agg(lambda x: sum(x>0)/len(x))
#        a['decrease_day_ratio'] = a['1DPM'].rolling(365).agg(lambda x: sum(x<0)/len(x))
#        a['mmt_discrete_A'] = a['increase_day_ratio'] - a['decrease_day_ratio']
#        df_daily = pd.concat([df_daily, a])
#    df_daily.sort_values('open_time_date', inplace=True)
#    idx = pd.MultiIndex.from_arrays([df_daily['open_time_date'], df_daily['ticker']], names=['datetime', 'ticker'])
#    df_daily = df_daily.set_index(idx)
#    return df_daily[['mmt_discrete_A']]

    def alpha_mmt_sec_rank_nP(self, n=20):
        """
        mmt_sec_rank_M = 每日计算个股日收益在横截面的排名，取过去n个时间单位排名均值
        """

        ranks = self.close.rank(pct=True,axis=1)
        ranks = ranks.rolling(n).mean()

        return ranks.stack()

#def mmt_sec_rank_3D(df):
#    return mmt_sec_rank_nP(df, 3).rename(columns={'mmt_sec_rank_M':'mmt_sec_rank_3D'})
#
#def mmt_sec_rank_7D(df):
#    return mmt_sec_rank_nP(df, 7).rename(columns={'mmt_sec_rank_M':'mmt_sec_rank_7D'})
#
#def mmt_sec_rank_15D(df):
#    return mmt_sec_rank_nP(df, 15).rename(columns={'mmt_sec_rank_M':'mmt_sec_rank_15D'})
#
#def mmt_sec_rank_1M(df):
#    return mmt_sec_rank_nP(df, 30).rename(columns={'mmt_sec_rank_M':'mmt_sec_rank_1M'})
#
#def mmt_sec_rank_3M(df):
#    return mmt_sec_rank_nP(df, 90).rename(columns={'mmt_sec_rank_M':'mmt_sec_rank_3M'})
#
#def mmt_sec_rank_6M(df):
#    return mmt_sec_rank_nP(df, 180).rename(columns={'mmt_sec_rank_M':'mmt_sec_rank_6M'})



#def mmt_sec_rank_A(df):
#    """
#    mmt_sec_rank_M = 每日计算个股日收益在横截面的排名，取过去一个月排名均值
#    """
#    df = df.copy()
#    df['open_time_date'] = df['open_time'].dt.date
#    gb = df.groupby('ticker')
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        a = group.groupby('open_time_date').tail(1)
#        a['1DPM'] = a.close.pct_change(1)
#        df_daily = pd.concat([df_daily, a])
#    df_daily.sort_values('open_time_date', inplace=True)
#   
#    df_daily.rename(columns={'open_time_date':'datetime'}, inplace=True)
#    df_daily['ret_rank_daily'] =  df_daily.groupby('datetime')['1DPM'].rank(pct=True)
#
#    gb = df_daily.groupby('ticker')
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        group['mmt_sec_rank_A'] = group['ret_rank_daily'].rolling(365).mean()
#        df_daily = pd.concat([df_daily, group])
#
#    df_daily.sort_values('datetime', inplace=True)
#    df_daily.reset_index(inplace=True)
#    df_daily.drop(['index'],axis=1,inplace=True)
#
#    idx = pd.MultiIndex.from_arrays([df_daily['datetime'], df_daily['ticker']], names=['datetime', 'ticker'])
#    df_daily = df_daily.set_index(idx)
#
#    return df_daily[['mmt_sec_rank_A']]

#def mmt_time_rank_M(df):
#    """
#    mmt_time_rank_M = 毎日计算个股价格在时序(1年内)的排名，取过去20个交易日排名取均值
#    """
#    df = df.copy()
#    df['open_time_date'] = df['open_time'].dt.date
#    gb = df.groupby('ticker')
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        a = group.groupby('open_time_date').tail(1)
#        a['close_rank_among1Y'] = a['close'].rolling(365).rank(pct=True)
#        a['mmt_time_rank_M'] = a['close_rank_among1Y'].rolling(30).mean()
#        df_daily = pd.concat([df_daily, a])
#    df_daily.sort_values('open_time_date', inplace=True)
#    idx = pd.MultiIndex.from_arrays([df_daily['open_time_date'], df_daily['ticker']], names=['datetime', 'ticker'])
#    df_daily = df_daily.set_index(idx)
#    return df_daily[['mmt_time_rank_M']]



    def alpha_mmt_highest_days_nP(self, n=10):
        """
        mmt_highest_days_A = 过去n个时间单位最高价日期距离当前日期的长度
        """

        rolling_windows = np.lib.stride_tricks.sliding_window_view(self.high.values, n, axis=0)
        max_indices = np.argmax(rolling_windows, axis=2) + 1
        days_since_max = n - max_indices
        nan_top = np.full((n-1,days_since_max.shape[1]),np.nan)
        days_since_max = np.concatenate((nan_top,days_since_max),axis=0)

        return pd.DataFrame(days_since_max,columns=self.high.columns,index=self.high.index).stack()
    


#def mmt_highest_days_A(df):
#    return mmt_highest_days_nP(df, 365)
#
#def mmt_highest_days_6M(df):
#    return mmt_highest_days_nP(df, 180).rename(columns={'mmt_highest_days_A':'mmt_highest_days_6M'})
#
#def mmt_highest_days_3M(df):
#    return mmt_highest_days_nP(df, 90).rename(columns={'mmt_highest_days_A':'mmt_highest_days_3M'})
#
#def mmt_highest_days_1M(df):
#    return mmt_highest_days_nP(df, 30).rename(columns={'mmt_highest_days_A':'mmt_highest_days_1M'})



#波动率因子

#def vol_std_nM(df,n):
#    """
#    vol_std_nM = 过去n个月（日收益率）的标准差
#    n -> int, 过去n个月    
#    """
#    df = df.copy()
#    df['open_time_date'] = df['open_time'].dt.date
#    gb = df.groupby('ticker')
#    
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        a = group.groupby('open_time_date').tail(1)
#        a['1DPM'] = a.close.pct_change(1)
#        a[f'vol_std_{n}M'] = a['1DPM'].rolling(int(30*n)).std()
#        df_daily = pd.concat([df_daily, a])
#
#    df_daily.sort_values('open_time_date', inplace=True)
#    idx = pd.MultiIndex.from_arrays([df_daily['open_time_date'], df_daily['ticker']], names=['datetime', 'ticker'])
#    df_daily = df_daily.set_index(idx)
#    return df_daily[[f'vol_std_{n}M']]

    def alpha_vol_std_nP(self,n=10):
        """
        vol_std_nM = 过去n个时间周期（日收益率）的标准差
        n -> int, 过去n个时间周期  
        """


        return (self.returns.rolling(n).std()).stack()


#def vol_std_nM2(df,m,n):
#    """
#    vol_std_nM = 过去n个月（日收益率）的标准差
#    n -> int, 过去n个月
#    """
#    df = df.copy()
#    gb = df.groupby('ticker')
#    
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        a = group
#        a['1DPM'] = a.close.pct_change(1)
#        a[f'vol_std_{n}M'] = a['1DPM'].rolling(int(m*n)).std()
#        df_daily = pd.concat([df_daily, a])
#
#    df_daily.sort_values('open_time', inplace=True)
#    idx = pd.MultiIndex.from_arrays([df_daily['open_time'], df_daily['ticker']], names=['datetime', 'ticker'])
#    df_daily = df_daily.set_index(idx)
#    return df_daily[[f'vol_std_{n}M']]


#def vol_std_nM22(df):
#    
#    a = vol_std_nM2(df,30,2)
#    
#    a = a.unstack()
#    a.columns = a.columns.droplevel()
#    return a
#
#
#
#def vol_std_3D(df):
#    a = vol_std_nM(df, 3/30)
#    a.columns = ['vol_std_3D']
#    return a
#
#def vol_std_7D(df):
#    a = vol_std_nM(df, 7/30)
#    a.columns = ['vol_std_7D']
#    return a
#
#def vol_std_15D(df):
#    a = vol_std_nM(df, 15/30)
#    a.columns = ['vol_std_15D']
#    return a
#
#def vol_std_1M(df):
#    a = vol_std_nM(df, 1)
#    a.columns = ['vol_std_1M']
#    return a
#
#def vol_std_3M(df):
#    a = vol_std_nM(df, 3)
#    a.columns = ['vol_std_3M']
#    return a
#
#def vol_std_6M(df):
#    a = vol_std_nM(df, 6)
#    a.columns = ['vol_std_6M']
#    return a


#def vol_highlow_avg_nM(df,n):
#    """
#    vol_highlow_avg_nM = 过去n个月（最高价/最低价）的均值
#    n -> int, 过去n个月
#    """
#    df = df.copy()
#    df['open_time_date'] = df['open_time'].dt.date
#    gb = df.groupby('ticker')
#
#    # 把分钟频数据整理成日频数据
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        o = group.groupby('open_time_date').head(1)[['open']]
#        c = group.groupby('open_time_date').tail(1)[['close']]
#        h = group.groupby('open_time_date')['high'].max()
#        l = group.groupby('open_time_date')['low'].min()
#        a = group.groupby('open_time_date').head(1)[['open_time_date', 'ticker', 'exchange_name']].rename(columns={'open_time_date':'datetime'})
#        a['open'] = o.values
#        a['high'] = h.values
#        a['low'] = l.values
#        a['close'] = c.values
#        df_daily = pd.concat([df_daily, a])
#    df_daily.sort_values('datetime', inplace=True)
#
#    df_daily['hdl'] = df_daily['high'] / df_daily['low']
#
#    #return df_daily.groupby('ticker').rolling(n*30)[['hdl']].mean()
#    gb = df_daily.groupby('ticker')
#    df_daily = pd.DataFrame()
#    for name, group in gb:
#        group[f'vol_highlow_avg_{n}M'] = group[['hdl']].rolling(int(n*30)).mean()
#        df_daily = pd.concat([df_daily, group])
#    
#    df_daily.sort_values('datetime', inplace=True)
#    idx = pd.MultiIndex.from_arrays([df_daily['datetime'], df_daily['ticker']], names=['datetime', 'ticker'])
#    df_daily = df_daily.set_index(idx)
#
#    
#    return df_daily[[f'vol_highlow_avg_{n}M']]



    def alpha_vol_highlow_avg_nP(self,n=10):
        """
        vol_highlow_avg_nM = 过去n个时间周期（最高价/最低价）的均值
        n -> int, 过去n个时间周期
        """
        
        
        return ((self.high/self.low).rolling(n).mean()).stack()


#def vol_highlow_avg_1D(df):
#    a = vol_highlow_avg_nM(df, 1/30)
#    a.columns = ['vol_highlow_avg_1D']
#    return a
#
#def vol_highlow_avg_7D(df):
#    a = vol_highlow_avg_nM(df, 7/30)
#    a.columns = ['vol_highlow_avg_7D']
#    return a
#
#def vol_highlow_avg_15D(df):
#    a = vol_highlow_avg_nM(df, 15/30)
#    a.columns = ['vol_highlow_avg_15D']
#    return a
#
#def vol_highlow_avg_1M(df):
#    a = vol_highlow_avg_nM(df, 1)
#    a.columns = ['vol_highlow_avg_1M']
#    return a
#
#def vol_highlow_avg_3M(df):
#    a = vol_highlow_avg_nM(df, 3)
#    a.columns = ['vol_highlow_avg_3M']
#    return a
#
#def vol_highlow_avg_6M(df):
#    a = vol_highlow_avg_nM(df, 6)
#    a.columns = ['vol_highlow_avg_6M']
#    return a

#####################################################################################

#def vol_highlow_std_nM(df, n):
#    """
#    vol_highlow_std_nM = 过去n个月（最高价/最低价）的标准差
#    n: int, 过去n个月
#    """
#    df = df.copy()
#
#    df['hdl'] = df['high'] / df['low']
#    df[f'vol_highlow_std_{n}M'] =  df.groupby('ticker').rolling(int(30*n))[['hdl']].std().reset_index().set_index('level_1').rename_axis('', axis=0)[['hdl']]
#
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#    return df[[f'vol_highlow_std_{n}M']]


    def alpha_vol_highlow_std_nM(self, n=10):
        """
        vol_highlow_std_nM = 过去n个时间周期（最高价/最低价）的标准差
        n: int, 过去n个时间周期
        """
    
        return ((self.high/self.low).rolling(n).std()).stack()

#def vol_highlow_std_3D(df):
#    return vol_highlow_std_nM(df, 3/30).rename(columns={f'vol_highlow_std_{3/30}M': 'vol_highlow_std_3D'})
#
#def vol_highlow_std_7D(df):
#    return vol_highlow_std_nM(df, 7/30).rename(columns={f'vol_highlow_std_{7/30}M': 'vol_highlow_std_7D'})
#
#def vol_highlow_std_15D(df):
#    return vol_highlow_std_nM(df, 15/30).rename(columns={f'vol_highlow_std_{15/30}M': 'vol_highlow_std_15D'})
#
#def vol_highlow_std_1M(df):
#    return vol_highlow_std_nM(df, 1)
#
#def vol_highlow_std_3M(df):
#    return vol_highlow_std_nM(df, 3)
#
#def vol_highlow_std_6M(df):
#    return vol_highlow_std_nM(df, 6)



    def alpha_vol_upshadow_avg_nP(self, n=10):
        """
        vol_upshadow_avg_nM = 标准化上影线因子过去n个时间周期的均值
        -- 标准化上影线 = （最高价 - max(开盘价，收盘价)）/最高价
        n: int, 过去n个时间周期
        """

        return ((self.upshadow.rolling(n).mean())).stack()

#def vol_upshadow_avg_nM(df, n):
#    """
#    vol_upshadow_avg_nM = 标准化上影线因子过去n个月的均值
#    -- 标准化上影线 = （最高价 - max(开盘价，收盘价)）/最高价
#    n: int, 过去n个月
#    """
#    df = df.copy()
#    df['upshadow'] = (df['high'] - df[['open', 'close']].max(axis=1))/df['high']
#    df[f'vol_upshadow_avg_{n}M'] = df.groupby('ticker').rolling(int(30*n))['upshadow'].mean().reset_index().set_index('level_1').rename_axis('', axis=0)[['upshadow']]
#
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#    return df[[f'vol_upshadow_avg_{n}M']]

#def vol_upshadow_avg_3D(df):
#    return vol_upshadow_avg_nM(df, 3/30).rename(columns={f'vol_upshadow_avg_{3/30}M': 'vol_upshadow_avg_3D'})
#
#def vol_upshadow_avg_7D(df):
#    return vol_upshadow_avg_nM(df, 7/30).rename(columns={f'vol_upshadow_avg_{7/30}M': 'vol_upshadow_avg_7D'})
#
#def vol_upshadow_avg_15D(df):
#    return vol_upshadow_avg_nM(df, 15/30).rename(columns={f'vol_upshadow_avg_{15/30}M': 'vol_upshadow_avg_15D'})
#
#def vol_upshadow_avg_1M(df):
#    return vol_upshadow_avg_nM(df, 1)
#
#def vol_upshadow_avg_3M(df):
#    return vol_upshadow_avg_nM(df, 3)
#
#def vol_upshadow_avg_6M(df):
#    return vol_upshadow_avg_nM(df, 6)



    def alpha_vol_upshadow_std_nP(self, n=10):
        """
        vol_upshadow_avg_nM = 标准化上影线因子过去n个时间周期的标准差
        -- 标准化上影线 = （最高价 - max(开盘价，收盘价)）/最高价
        n: int, 过去n个时间周期
        """

        return ((self.upshadow.rolling(n).std())).stack()


#def vol_upshadow_std_nM(df, n):
#    """
#    vol_upshadow_std_nM = 标准化上影线因子过去n个月的标准差
#    标准化上影线 =（最高价 - max(开盘价，收盘价)）/最高价
#    n: int, 过去n个月
#    """
#    df = df.copy()
#    df['upshadow'] = (df['high'] - df[['open', 'close']].max(axis=1))/df['high']
#    df[f'vol_upshadow_std_{n}M'] = df.groupby('ticker').rolling(int(30*n))['upshadow'].std().reset_index().set_index('level_1').rename_axis('', axis=0)[['upshadow']]
#
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#    return df[[f'vol_upshadow_std_{n}M']]
#
#def vol_upshadow_std_3D(df):
#    return vol_upshadow_std_nM(df, 3/30).rename(columns={f'vol_upshadow_std_{3/30}M': 'vol_upshadow_std_3D'})
#
#def vol_upshadow_std_7D(df):
#    return vol_upshadow_std_nM(df, 7/30).rename(columns={f'vol_upshadow_std_{7/30}M': 'vol_upshadow_std_7D'})
#
#def vol_upshadow_std_15D(df):
#    return vol_upshadow_std_nM(df, 15/30).rename(columns={f'vol_upshadow_std_{15/30}M': 'vol_upshadow_std_15D'})
#
#def vol_upshadow_std_1M(df):
#    return vol_upshadow_std_nM(df, 1)
#
#def vol_upshadow_std_3M(df):
#    return vol_upshadow_std_nM(df, 3)
#
#def vol_upshadow_std_6M(df):
#    return vol_upshadow_std_nM(df, 6)


    def alpha_vol_downshadow_avg_nP(self, n=10):
        """
        vol_downshadow_avg_nM = 标准化下影线因子过去n个时间周期的均值
        -- 标准化下影线 = （min(开盘价，收盘价) - 最低价）/最低价
        n: int, 过去n个时间周期
        """

        return (self.downshadow.rolling(n).mean()).stack()



    def alpha_vol_downshadow_std_nP(self, n=10):
        """
        vol_downshadow_avg_nM = 标准化下影线因子过去n个时间周期的标准差
        -- 标准化下影线 = （min(开盘价，收盘价) - 最低价）/最低价
        n: int, 过去n个时间周期
        """

        return (self.downshadow.rolling(n).std()).stack()


#def vol_downshadow_stats_nM(df, n):
#    """
#    stats_mode: str, available input: 'avg', 'std'
#    vol_downshadow_avg_nM = 标准化下影线过去n个月的均值
#    vol_downshadow_std_nM = 标准化下影线过去n个月的标准差
#    标准化上影线 =（最高价 - max(开盘价，收盘价)）/最高价
#
#    n: int, 过去n个月
#    """
#    stats_mode == 'avg'
#    df = df.copy()
#    df['downshadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['low']
#
#    if stats_mode == 'avg':
#        df[f'vol_downshadow_{stats_mode}_{n}M'] = df.groupby('ticker').rolling(int(n))['downshadow'].mean().reset_index().set_index('level_1').rename_axis('', axis=0)[['downshadow']]
#    elif stats_mode == 'std':
#        df[f'vol_downshadow_{stats_mode}_{n}M'] = df.groupby('ticker').rolling(int(n))['downshadow'].std().reset_index().set_index('level_1').rename_axis('', axis=0)[['downshadow']]
#    else:
#        print('state_mode input error')
#    
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#    return df[[f'vol_downshadow_{stats_mode}_{n}M']]



#def vol_downshadow_stats_nM(df, n, stats_mode):
#    """
#    stats_mode: str, available input: 'avg', 'std'
#    vol_downshadow_avg_nM = 标准化下影线过去n个月的均值
#    vol_downshadow_std_nM = 标准化下影线过去n个月的标准差
#    标准化上影线 =（最高价 - max(开盘价，收盘价)）/最高价
#
#    n: int, 过去n个月
#    """
#    df = df.copy()
#    df['downshadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['low']
#
#    if stats_mode == 'avg':
#        df[f'vol_downshadow_{stats_mode}_{n}M'] = df.groupby('ticker').rolling(int(30*n))['downshadow'].mean().reset_index().set_index('level_1').rename_axis('', axis=0)[['downshadow']]
#    elif stats_mode == 'std':
#        df[f'vol_downshadow_{stats_mode}_{n}M'] = df.groupby('ticker').rolling(int(30*n))['downshadow'].std().reset_index().set_index('level_1').rename_axis('', axis=0)[['downshadow']]
#    else:
#        print('state_mode input error')
#    
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#    return df[[f'vol_downshadow_{stats_mode}_{n}M']]


#def vol_downshadow_avg_1M(df):
#    return vol_downshadow_stats_nM(df, 1, 'avg')
#
#def vol_downshadow_avg_3M(df):
#    return vol_downshadow_stats_nM(df, 3, 'avg')
#
#def vol_downshadow_avg_6M(df):
#    return vol_downshadow_stats_nM(df, 6, 'avg')
#
#def vol_downshadow_std_1M(df):
#    return vol_downshadow_stats_nM(df, 1, 'std')
#
#def vol_downshadow_std_3M(df):
#    return vol_downshadow_stats_nM(df, 3, 'std')
#
#def vol_downshadow_std_6M(df):
#    return vol_downshadow_stats_nM(df, 6, 'std')
#
#def vol_downshadow_avg_3D(df):
#    return vol_downshadow_stats_nM(df, 3/30, 'avg').rename(columns={f'vol_downshadow_avg_{3/30}M': 'vol_downshadow_avg_3D'})
#
#def vol_downshadow_avg_7D(df):
#    return vol_downshadow_stats_nM(df, 7/30, 'avg').rename(columns={f'vol_downshadow_avg_{7/30}M': 'vol_downshadow_avg_7D'})
#
#def vol_downshadow_avg_15D(df):
#    return vol_downshadow_stats_nM(df, 15/30, 'avg').rename(columns={f'vol_downshadow_avg_{15/30}M': 'vol_downshadow_avg_15D'})
#
#def vol_downshadow_std_3D(df):
#    return vol_downshadow_stats_nM(df, 3/30, 'std').rename(columns={f'vol_downshadow_std_{3/30}M': 'vol_downshadow_std_3D'})
#
#def vol_downshadow_std_7D(df):
#    return vol_downshadow_stats_nM(df, 7/30, 'std').rename(columns={f'vol_downshadow_std_{7/30}M': 'vol_downshadow_std_7D'})
#
#def vol_downshadow_std_15D(df):
#    return vol_downshadow_stats_nM(df, 15/30, 'std').rename(columns={f'vol_downshadow_std_{15/30}M': 'vol_downshadow_std_15D'})



    def alpha_vol_w_upshadow_avg_nP(self, n=10):
        """
        stats_mode: str, available input: 'avg', 'std'
        vol_w_upshadow_avg_nM = 威廉上影线因子过去n个时间单位均值
        vol_w_upshadow_std_nM = 威廉上影线因子过去n个时间单位标准差
        威廉上影线 = （最高价-收盘价）/最高价

        n: int, 过去n个月
        """
        
        return (self.w_upshadow.rolling(n).mean()).stack()



    def alpha_vol_w_upshadow_std_nP(self, n=10):
        """
        stats_mode: str, available input: 'avg', 'std'
        vol_w_upshadow_avg_nM = 威廉上影线因子过去n个时间单位均值
        vol_w_upshadow_std_nM = 威廉上影线因子过去n个时间单位标准差
        威廉上影线 = （最高价-收盘价）/最高价

        n: int, 过去n个月
        """
        
        return (self.w_upshadow.rolling(n).std()).stack()

#def vol_w_upshadow_stats_nM(df, n, stats_mode):
#    """
#    stats_mode: str, available input: 'avg', 'std'
#    vol_w_upshadow_avg_nM = 威廉上影线因子过去n个月均值
#    vol_w_upshadow_std_nM = 威廉上影线因子过去n个月标准差
#    威廉上影线 = （最高价-收盘价）/最高价
#
#    n: int, 过去n个月
#    """
#    df = df.copy()
#    df['w_upshadow'] = (df['high'] - df['close']) / df['high']
#
#    if stats_mode == 'avg':
#        df[f'vol_w_upshadow_{stats_mode}_{n}M'] = df.groupby('ticker').rolling(int(30*n))['w_upshadow'].mean().reset_index().set_index('level_1').rename_axis('', axis=0)[['w_upshadow']]
#    elif stats_mode == 'std':
#        df[f'vol_w_upshadow_{stats_mode}_{n}M'] = df.groupby('ticker').rolling(int(30*n))['w_upshadow'].std().reset_index().set_index('level_1').rename_axis('', axis=0)[['w_upshadow']]
#    else:
#        print('stats_mode input error')
#
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#    
#    return df[[f'vol_w_upshadow_{stats_mode}_{n}M']]

#def vol_w_upshadow_avg_1M(df):
#    return vol_w_upshadow_stats_nM(df, 1, 'avg')
#
#def vol_w_upshadow_std_1M(df):
#    return vol_w_upshadow_stats_nM(df, 1, 'std')
#
#def vol_w_upshadow_avg_3M(df):
#    return vol_w_upshadow_stats_nM(df, 3, 'avg')
#
#def vol_w_upshadow_std_3M(df):
#    return vol_w_upshadow_stats_nM(df, 3, 'std')
#
#def vol_w_upshadow_avg_6M(df):
#    return vol_w_upshadow_stats_nM(df, 6, 'avg')
#
#def vol_w_upshadow_std_6M(df):
#    return vol_w_upshadow_stats_nM(df, 6, 'std')
#
#def vol_w_upshadow_avg_3D(df):
#    return vol_w_upshadow_stats_nM(df, 3/30, 'avg').rename(columns={f'vol_w_upshadow_avg_{3/30}M': 'vol_w_upshadow_avg_3D'})
#
#def vol_w_upshadow_std_7D(df):
#    return vol_w_upshadow_stats_nM(df, 7/30, 'std').rename(columns={f'vol_w_upshadow_std_{7/30}M': 'vol_w_upshadow_std_7D'})
#
#def vol_w_upshadow_avg_15D(df):
#    return vol_w_upshadow_stats_nM(df, 15/30, 'avg').rename(columns={f'vol_w_upshadow_avg_{15/30}M': 'vol_w_upshadow_avg_15D'})
#
#def vol_w_upshadow_std_3D(df):
#    return vol_w_upshadow_stats_nM(df, 3/30, 'std').rename(columns={f'vol_w_upshadow_std_{3/30}M': 'vol_w_upshadow_std_3D'})
#
#def vol_w_upshadow_avg_7D(df):
#    return vol_w_upshadow_stats_nM(df, 7/30, 'avg').rename(columns={f'vol_w_upshadow_avg_{7/30}M': 'vol_w_upshadow_avg_7D'})
#
#def vol_w_upshadow_std_15D(df):
#    return vol_w_upshadow_stats_nM(df, 15/30, 'std').rename(columns={f'vol_w_upshadow_std_{15/30}M': 'vol_w_upshadow_std_15D'})


    def alpha_vol_w_downshadow_avg_nP(self, n=10):
        """
        stats_mode: str, available input: 'avg', 'std'
        vol_w_downshadow_avg_nM = 威廉下影线因子过去n个时间单位的均值
        vol_w_downshadow_std_nM = 威廉下影线因子过去n个时间单位的标准差
        威廉下影线 = （收盘价-最低价）/最低价

        n: int, 过去n个月
        """

        return (self.w_downshadow.rolling(n).mean()).stack()



    def alpha_vol_w_downshadow_std_nP(self, n=10):
        """
        stats_mode: str, available input: 'avg', 'std'
        vol_w_downshadow_avg_nM = 威廉下影线因子过去n个时间单位的均值
        vol_w_downshadow_std_nM = 威廉下影线因子过去n个时间单位的标准差
        威廉下影线 = （收盘价-最低价）/最低价

        n: int, 过去n个月
        """

        return (self.w_downshadow.rolling(n).std()).stack()


#def vol_w_downshadow_stats_nM(df, n, stats_mode):
#    """
#    stats_mode: str, available input: 'avg', 'std'
#    vol_w_downshadow_avg_nM = 威廉下影线因子过去n个月均值
#    vol_w_downshadow_std_nM = 威廉下影线因子过去n个月标准差
#    威廉下影线 = （收盘价-最低价）/最低价
#
#    n: int, 过去n个月
#    """
#    df = df.copy()
#    df['w_downshadow'] = (df['close'] - df['low']) / df['low']
#
#    if stats_mode == 'avg':
#        df[f'vol_w_downshadow_{stats_mode}_{n}M'] = df.groupby('ticker').rolling(int(30*n))['w_downshadow'].mean().reset_index().set_index('level_1').rename_axis('', axis=0)[['w_downshadow']]    
#    elif stats_mode == 'std':
#        df[f'vol_w_downshadow_{stats_mode}_{n}M'] = df.groupby('ticker').rolling(int(30*n))['w_downshadow'].std().reset_index().set_index('level_1').rename_axis('', axis=0)[['w_downshadow']]
#    else:
#        print('stats_mode input error')
#
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#    
#    return df[[f'vol_w_downshadow_{stats_mode}_{n}M']]

#def vol_w_downshadow_avg_1M(df):
#    return vol_w_downshadow_stats_nM(df, 1, 'avg')
#
#def vol_w_downshadow_avg_3M(df):
#    return vol_w_downshadow_stats_nM(df, 3, 'avg')
#
#def vol_w_downshadow_avg_6M(df):
#    return vol_w_downshadow_stats_nM(df, 6, 'avg')
#
#def vol_w_downshadow_std_1M(df):
#    return vol_w_downshadow_stats_nM(df, 1, 'std')
#
#def vol_w_downshadow_std_3M(df):
#    return vol_w_downshadow_stats_nM(df, 3, 'std')
#
#def vol_w_downshadow_std_6M(df):
#    return vol_w_downshadow_stats_nM(df, 6, 'std')
#
#def vol_w_downshadow_avg_3D(df):
#    return vol_w_downshadow_stats_nM(df, 3/30, 'avg').rename(columns={f'vol_w_downshadow_avg_{3/30}M': 'vol_w_downshadow_avg_3D'})
#
#def vol_w_downshadow_avg_7D(df):
#    return vol_w_downshadow_stats_nM(df, 7/30, 'avg').rename(columns={f'vol_w_downshadow_avg_{7/30}M': 'vol_w_downshadow_avg_7D'})
#
#def vol_w_downshadow_avg_15D(df):
#    return vol_w_downshadow_stats_nM(df, 15/30, 'avg').rename(columns={f'vol_w_downshadow_avg_{15/30}M': 'vol_w_downshadow_avg_15D'})
#
#def vol_w_downshadow_std_3D(df):
#    return vol_w_downshadow_stats_nM(df, 3/30, 'std').rename(columns={f'vol_w_downshadow_std_{3/30}M': 'vol_w_downshadow_std_3D'})
#
#def vol_w_downshadow_std_7D(df):
#    return vol_w_downshadow_stats_nM(df, 7/30, 'std').rename(columns={f'vol_w_downshadow_std_{7/30}M': 'vol_w_downshadow_std_7D'})
#
#def vol_w_downshadow_std_15D(df):
#    return vol_w_downshadow_stats_nM(df, 15/30, 'std').rename(columns={f'vol_w_downshadow_std_{15/30}M': 'vol_w_downshadow_std_15D'})

#流动性因子


    def alpha_liq_turn_avg_nP(self, n=10):
        """
        liq_turn_avg_1M = n个时间单位换手率的均值
        liq_turn_std_1M = n个时间单位换手率的标准差
        """
        
        return (self.turnover.rolling(n).mean()).stack()


    def alpha_liq_turn_std_nP(self, n=10):
        """
        liq_turn_avg_1M = n个时间单位换手率的均值
        liq_turn_std_1M = n个时间单位换手率的标准差
        """
        
        return (self.turnover.rolling(n).std()).stack()


#def liq_turn_stats_nP(df, n, stats_mode):
#    """
#    liq_turn_avg_1M = 一个月换手率的均值
#    liq_turn_std_1M = 一个月换手率的标准差
#    """
#    df = df.copy()
#    df['turnover_rate'] = df['volume'] / df['total_supply']
#    if stats_mode == 'avg':
#        df[f'liq_turn_{stats_mode}_{n}P'] = df.groupby('ticker')['turnover_rate'].rolling(n).mean().reset_index().set_index('level_1').turnover_rate
#    elif stats_mode == 'std':
#        df[f'liq_turn_{stats_mode}_{n}P'] = df.groupby('ticker')['turnover_rate'].rolling(n).std().reset_index().set_index('level_1').turnover_rate
#    else:
#        print('stats mode input error')
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#    return df[[f'liq_turn_{stats_mode}_{n}P']]

#def liq_turn_avg_3D(df):
#    return liq_turn_stats_nP(df, 3, 'avg')
#
#def liq_turn_avg_7D(df):
#    return liq_turn_stats_nP(df, 7, 'avg')
#
#def liq_turn_avg_15D(df):
#    return liq_turn_stats_nP(df, 15, 'avg')
#
#def liq_turn_avg_1M(df):
#    return liq_turn_stats_nP(df, 30, 'avg')
#
#def liq_turn_avg_3M(df):
#    return liq_turn_stats_nP(df, 90, 'avg')
#
#def liq_turn_avg_6M(df):
#    return liq_turn_stats_nP(df, 180, 'avg')
#
#def liq_turn_std_3D(df):
#    return liq_turn_stats_nP(df, 3, 'std')
#
#def liq_turn_std_7D(df):
#    return liq_turn_stats_nP(df, 7, 'std')
#
#def liq_turn_std_15D(df):
#    return liq_turn_stats_nP(df, 15, 'std')
#
#def liq_turn_std_1M(df):
#    return liq_turn_stats_nP(df, 30, 'std')
#
#def liq_turn_std_3M(df):
#    return liq_turn_stats_nP(df, 90, 'std')
#
#def liq_turn_std_6M(df):
#    return liq_turn_stats_nP(df, 180, 'std')


    def alpha_liq_vstd_nP(self, n=10):
        """
        liq_vstd_nP = n个时间单位成交额 / 过去n个时间单位每日收益率标准差
        n: int, 过去n个时间单位
        """

        
        return (self.quote_volume.rolling(n).sum() / self.returns.rolling(n).std()).stack()


#def liq_vstd_nM(df, n):
#    """
#    liq_vstd_nM = n个月成交额 / 过去n个月每日收益率标准差
#    n: int, 过去n个月
#    """
#    df = df.copy()
#    df[f'quote_volume_{n}M'] =  df.groupby('ticker')['quote_volume'].rolling(int(30*n)).sum().reset_index().set_index('level_1').rename_axis('', axis=0)['quote_volume']
#    df['1DPM'] = df.groupby('ticker')['close'].pct_change(1)
#    df[f'1DPM_{n}M_std'] =  df.groupby('ticker')['1DPM'].rolling(int(30*n)).std().reset_index().set_index('level_1').rename_axis('', axis=0)['1DPM']
#    df[f'liq_vstd_{n}M'] = df[f'quote_volume_{n}M'] / df[f'1DPM_{n}M_std']
#
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#    
#    return df[[f'liq_vstd_{n}M']]

#def liq_vstd_1M(df):
#    return liq_vstd_nM(df, 1)
#
#def liq_vstd_3M(df):
#    return liq_vstd_nM(df, 3)
#
#def liq_vstd_6M(df):
#    return liq_vstd_nM(df, 6)
#
#def liq_vstd_3D(df):
#    return liq_vstd_nM(df, 3/30).rename(columns={f'liq_vstd_{3/30}M':'liq_vstd_3D'})
#
#def liq_vstd_7D(df):
#    return liq_vstd_nM(df, 7/30).rename(columns={f'liq_vstd_{7/30}M':'liq_vstd_7D'})
#
#def liq_vstd_15D(df):
#    return liq_vstd_nM(df, 15/30).rename(columns={f'liq_vstd_{15/30}M':'liq_vstd_15D'})


    def alpha_liq_amihud_avg_nP(self, n=10):
        """
        stats_mode = 'avg', 'std'
        liq_amihud_avg_nM = 过去n个时间单位（收益率/成交额）的平均值
        liq_amihud_std_nM = 过去n个时间单位（收益率/成交额）的标准差
        n: int, 过去n个月
        """
        
        return (self.returns / self.quote_volume).rolling(n).mean().stack()


    def alpha_liq_amihud_avg_nP(self, n=10):
        """
        stats_mode = 'avg', 'std'
        liq_amihud_avg_nM = 过去n个时间单位（收益率/成交额）的平均值
        liq_amihud_std_nM = 过去n个时间单位（收益率/成交额）的标准差
        n: int, 过去n个月
        """
        
        return (self.returns / self.quote_volume).rolling(n).mean().stack()

#def liq_amihud_stats_nM(df, n, stats_mode):
#    """
#    stats_mode = 'avg', 'std'
#    liq_amihud_avg_nM = 过去n个月（日收益率/成交额）的平均值
#    liq_amihud_std_nM = 过去n个月（日收益率/成交额）的标准差
#    n: int, 过去n个月
#    """
#    df = df.copy()
#    df['1DPM'] = df.groupby('ticker')['close'].pct_change(1)
#    df['amihud'] = df['1DPM'] / df['quote_volume']
#    if stats_mode == 'avg':
#        df[f'liq_amihud_{stats_mode}_{n}M'] = df.groupby('ticker')['amihud'].rolling(int(30*n)).mean().reset_index().set_index('level_1').rename_axis('', axis=0)['amihud']
#    elif stats_mode == 'std':
#        df[f'liq_amihud_{stats_mode}_{n}M'] = df.groupby('ticker')['amihud'].rolling(int(30*n)).std().reset_index().set_index('level_1').rename_axis('',axis=0)['amihud']
#    else:
#        print('stats_mode input error')
#
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#
#    return df[[f'liq_amihud_{stats_mode}_{n}M']]

#def liq_amihud_avg_1M(df):
#    return liq_amihud_stats_nM(df, 1, 'avg')
#
#def liq_amihud_avg_2M(df):
#    return liq_amihud_stats_nM(df, 2, 'avg')
#
#def liq_amihud_avg_3M(df):
#    return liq_amihud_stats_nM(df, 3, 'avg')
#
#
#def liq_amihud_std_1M(df):
#    return liq_amihud_stats_nM(df, 1, 'std')
#
#def liq_amihud_std_2M(df):
#    return liq_amihud_stats_nM(df, 2, 'std')
#
#def liq_amihud_std_3M(df):
#    return liq_amihud_stats_nM(df, 3, 'std')
#
#def liq_amihud_avg_3D(df):
#    return liq_amihud_stats_nM(df, 3/30, 'avg').rename(columns={f'liq_amihud_avg_{3/30}M':'liq_amihud_avg_3D'})
#
#def liq_amihud_avg_7D(df):
#    return liq_amihud_stats_nM(df, 7/30, 'avg').rename(columns={f'liq_amihud_avg_{7/30}M':'liq_amihud_avg_7D'})
#
#def liq_amihud_avg_15D(df):
#    return liq_amihud_stats_nM(df, 15/30, 'avg').rename(columns={f'liq_amihud_avg_{15/30}M':'liq_amihud_avg_15D'})
#
#
#def liq_amihud_std_3D(df):
#    return liq_amihud_stats_nM(df, 3/30, 'std').rename(columns={f'liq_amihud_std_{3/30}M':'liq_amihud_std_3D'})
#
#def liq_amihud_std_7D(df):
#    return liq_amihud_stats_nM(df, 7/30, 'std').rename(columns={f'liq_amihud_std_{7/30}M':'liq_amihud_std_7D'})
#
#def liq_amihud_std_15D(df):
#    return liq_amihud_stats_nM(df, 15/30, 'std').rename(columns={f'liq_amihud_std_{15/30}M':'liq_amihud_std_15D'})




#def liq_shortcut_stats_nM(df, n, stats_mode):
#    """
#    stats_mode = 'avg', 'std'
#    liq_shortcut_avg_nM = 过去n个月（日k线最短路径/成交额）的平均值
#    liq_shortcut_std_nM = 过去n个月（日k线最短路径/成交额）的标准差
#    shortcut: 日k线最短路径 = 2 *（最高价-最低价）- abs(开盘价-收盘价)
#    """
#    df = df.copy()
#    df['shortcut'] = 2 * (df['high'] - df['low']) - (df['open'] - df['close']).abs()
#    if stats_mode == 'avg':
#        df[f'liq_shortcut_{stats_mode}_{n}M'] = df.groupby('ticker')['shortcut'].rolling(int(30*n)).mean().reset_index().set_index('level_1').rename_axis('', axis=0)['shortcut']
#    elif stats_mode == 'std':
#        df[f'liq_shortcut_{stats_mode}_{n}M'] = df.groupby('ticker')['shortcut'].rolling(int(30*n)).std().reset_index().set_index('level_1').rename_axis('', axis=0)['shortcut']
#    else:
#        print('stats_mode input error')
#
#    df.sort_values('open_time', inplace=True)
#    df['datetime'] = df['open_time']
#    idx = pd.MultiIndex.from_arrays([df['datetime'], df['ticker']], names=['datetime', 'ticker'])
#    df = df.set_index(idx)
#
#    return df[[f'liq_shortcut_{stats_mode}_{n}M']]


    def alpha_liq_shortcut_avg_nP(self, n=10):
        """
        stats_mode = 'avg', 'std'
        liq_shortcut_avg_nM = 过去n个时间单位（日k线最短路径/成交额）的平均值
        liq_shortcut_std_nM = 过去n个时间单位（日k线最短路径/成交额）的标准差
        shortcut: 日k线最短路径 = 2 *（最高价-最低价）- abs(开盘价-收盘价)
        """
        
        return (self.shortcut.rolling(n).mean()).stack()




    def alpha_liq_shortcut_std_nP(self, n=10):
        """
        stats_mode = 'avg', 'std'
        liq_shortcut_avg_nM = 过去n个时间单位（日k线最短路径/成交额）的平均值
        liq_shortcut_std_nM = 过去n个时间单位（日k线最短路径/成交额）的标准差
        shortcut: 日k线最短路径 = 2 *（最高价-最低价）- abs(开盘价-收盘价)
        """
        
        return (self.shortcut.rolling(n).std()).stack()


#def liq_shortcut_avg_1M(df):
#    return liq_shortcut_stats_nM(df, 1, 'avg')
#
#def liq_shortcut_avg_3M(df):
#    return liq_shortcut_stats_nM(df, 3, 'avg')
#
#def liq_shortcut_avg_6M(df):
#    return liq_shortcut_stats_nM(df, 6, 'avg')
#
#def liq_shortcut_std_1M(df):
#    return liq_shortcut_stats_nM(df, 1, 'std')
#
#def liq_shortcut_std_3M(df):
#    return liq_shortcut_stats_nM(df, 3, 'std')
#
#def liq_shortcut_std_6M(df):
#    return liq_shortcut_stats_nM(df, 6, 'std')
#
#def liq_shortcut_avg_3D(df):
#    return liq_shortcut_stats_nM(df, 3/30, 'avg').rename(columns={f'liq_shortcut_avg_{3/30}M':'liq_shortcut_avg_3D'})
#
#def liq_shortcut_avg_7D(df):
#    return liq_shortcut_stats_nM(df, 7/30, 'avg').rename(columns={f'liq_shortcut_avg_{7/30}M':'liq_shortcut_avg_7D'})
#
#def liq_shortcut_avg_15D(df):
#    return liq_shortcut_stats_nM(df, 15/30, 'avg').rename(columns={f'liq_shortcut_avg_{15/30}M':'liq_shortcut_avg_15D'})
#
#def liq_shortcut_std_3D(df):
#    return liq_shortcut_stats_nM(df, 3/30, 'std').rename(columns={f'liq_shortcut_std_{3/30}M':'liq_shortcut_std_3D'})
#
#def liq_shortcut_std_7D(df):
#    return liq_shortcut_stats_nM(df, 7/30, 'std').rename(columns={f'liq_shortcut_std_{7/30}M':'liq_shortcut_std_7D'})
#
#def liq_shortcut_std_15D(df):
#    return liq_shortcut_stats_nM(df, 15/30, 'std').rename(columns={f'liq_shortcut_std_{15/30}M':'liq_shortcut_std_15D'})


    def alpha_liq_LIX_avg_np(self,n=10):
        """
        liq_LIX = 过去n个时间单位的LIX均值
        n: int, 过去n个时间单位
        LIX = log(成交额 / (最高价 - 最低价))
        """
        

        return (self.LIX.rolling(n).mean()).stack()



    def alpha_liq_LIX_std_np(self,n=10):
        """
        liq_LIX = 过去n个时间单位的LIX标准差
        n: int, 过去n个时间单位
        LIX = log(成交额 / (最高价 - 最低价))
        """
        

        return (self.LIX.rolling(n).std()).stack()


    """
量价相关性因子
    """
    def alpha_corr_shift_price_turn_nP(self, n=10, m=2):
        """
        换手率与价格相关性因子（量领先m期）：过去n个交易周期，t日换手率与t+m日价格的相关系数
        n: 过去n期交易
        m: price shift 参数
        """
        
        price_shift = self.close.shift(m)

        return (self.turnover.rolling(n).corr(price_shift)).stack()

#def corr_price_turn_3D(df):
#    a = corr_shift_price_turn_nP(df, 3, 0)
#    a.columns = ['corr_price_turn_3D']
#    return a
#
#def corr_price_turn_7D(df):
#    a = corr_shift_price_turn_nP(df, 7, 0)
#    a.columns = ['corr_price_turn_7D']
#    return a
#
#def corr_price_turn_14D(df):
#    a = corr_shift_price_turn_nP(df, 14, 0)
#    a.columns = ['corr_price_turn_14D']
#    return a
#
#def corr_price_turn_1M(df):
#    a = corr_shift_price_turn_nP(df, 30, 0)
#    a.columns = ['corr_price_turn_1M']
#    return a
#
#def corr_price_turn_post_3D(df):
#    a = corr_shift_price_turn_nP(df, 3, -1)
#    a.columns = ['corr_price_turn_post_3D']
#    return a
#
#def corr_price_turn_post_7D(df):
#    a = corr_shift_price_turn_nP(df, 7, -1)
#    a.columns = ['corr_price_turn_post_7D']
#    return a
#
#def corr_price_turn_post_14D(df):
#    a = corr_shift_price_turn_nP(df, 14, -1)
#    a.columns = ['corr_price_turn_post_14D']
#    return a
#
#def corr_price_turn_post_1M(df):
#    a = corr_shift_price_turn_nP(df, 30, -1)
#    a.columns = ['corr_price_turn_post_1M']
#    return a
#
#def corr_price_turn_prior_3D(df):
#    a = corr_shift_price_turn_nP(df, 3, 1)
#    a.columns = ['corr_price_turn_prior_3D']
#    return a
#
#def corr_price_turn_prior_7D(df):
#    a = corr_shift_price_turn_nP(df, 7, 1)
#    a.columns = ['corr_price_turn_prior_7D']
#    return a

#def corr_price_turn_prior_14D(df):
#    a = corr_shift_price_turn_nP(df, 14, 1)
#    a.columns = ['corr_price_turn_prior_14D']
#    return a
#
#def corr_price_turn_prior_1M(df):
#    a = corr_shift_price_turn_nP(df, 30, 1)
#    a.columns = ['corr_price_turn_prior_1M']
#    return a

    def alpha_corr_shift_ret_turn_nP(self, n=10, m=2):
        """
        换手率与收益率相关性因子（量领先m期）：过去n个交易周期，t日换手率与t+m日收益率的相关系数
        n: 过去n期交易
        m: ret shift 参数
        """
        ret_shift = self.returns.shift(m)

        return (self.turnover.rolling(n).corr(ret_shift)).stack()

    def alpha_corr_shift_ret_turn_inv_nP(self, n=10, m=2):
        """
        换手率与收益率相关性因子（量领先m期）：过去n个交易周期，t日换手率与t+m日收益率的相关系数
        n: 过去n期交易
        m: ret shift 参数
        """
        turnover_shift = self.turnover.shift(m)

        return (turnover_shift.rolling(n).corr(self.returns)).stack()



#def corr_ret_turn_3D(df):
#    a = corr_shift_ret_turn_nP(df, 3, 0)
#    a.columns = ['corr_ret_turn_3D']
#    return a
#
#def corr_ret_turn_7D(df):
#    a = corr_shift_ret_turn_nP(df, 7, 0)
#    a.columns = ['corr_ret_turn_7D']
#    return a
#
#def corr_ret_turn_14D(df):
#    a = corr_shift_ret_turn_nP(df, 14, 0)
#    a.columns = ['corr_ret_turn_14D']
#    return a
#
#def corr_ret_turn_1M(df):
#    a = corr_shift_ret_turn_nP(df, 30, 0)
#    a.columns = ['corr_ret_turn_1M']
#    return a
#
#def corr_ret_turn_post_3D(df):
#    a = corr_shift_ret_turn_nP(df, 3, -1)
#    a.columns = ['corr_ret_turn_post_3D']
#    return a
#
#def corr_ret_turn_post_7D(df):
#    a = corr_shift_ret_turn_nP(df, 7, -1)
#    a.columns = ['corr_ret_turn_post_7D']
#    return a
#
#def corr_ret_turn_post_14D(df):
#    a = corr_shift_ret_turn_nP(df, 14, -1)
#    a.columns = ['corr_ret_turn_post_14D']
#    return a
#
#def corr_ret_turn_post_1M(df):
#    a = corr_shift_ret_turn_nP(df, 30, -1)
#    a.columns = ['corr_ret_turn_post_1M']
#    return a

#def corr_ret_turn_prior_3D(df):
#    a = corr_shift_ret_turn_nP(df, 3, 1)
#    a.columns = ['corr_ret_turn_prior_3D']
#    return a
#
#def corr_ret_turn_prior_7D(df):
#    a = corr_shift_ret_turn_nP(df, 7, 1)
#    a.columns = ['corr_ret_turn_prior_7D']
#    return a
#
#def corr_ret_turn_prior_14D(df):
#    a = corr_shift_ret_turn_nP(df, 14, 1)
#    a.columns = ['corr_ret_turn_prior_14D']
#    return a
#
#def corr_ret_turn_prior_1M(df):
#    a = corr_shift_ret_turn_nP(df, 30, 1)
#    a.columns = ['corr_ret_turn_prior_1M']
#    return a


    def alpha_corr_shift_ret_turnd_nP(self, n=10, m=2):
        """
        换手率变动与收益率相关性因子（量领先m期）：过去n个交易周期，t日换手率与t+m日价格的相关系数
        n: 过去n期交易
        m: ret shift 参数
        """

        returns_d = self.returns.pct_change().shift(m)
        turnover_d = self.turnover.pct_change()

        return (turnover_d.rolling(n).corr(returns_d)).stack()


    def alpha_corr_shift_ret_turnd_inv_nP(self, n=10, m=2):
        """
        换手率变动与收益率相关性因子（量落后m期）：过去n个交易周期，t日换手率与t+m日价格的相关系数
        n: 过去n期交易
        m: ret shift 参数
        """

        returns_d = self.returns.pct_change()
        turnover_d = self.turnover.pct_change().shift(m)

        return (turnover_d.rolling(n).corr(returns_d)).stack()


#def corr_ret_turnd_3D(df):
#    a = corr_shift_ret_turnd_nP(df, 3, 0)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#def corr_ret_turnd_7D(df):
#    a = corr_shift_ret_turnd_nP(df, 7, 0)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#def corr_ret_turnd_14D(df):
#    a = corr_shift_ret_turnd_nP(df, 14, 0)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#def corr_ret_turnd_1M(df):
#    a = corr_shift_ret_turnd_nP(df, 30, 0)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#
#def corr_ret_turnd_post_3D(df):
#    a = corr_shift_ret_turnd_nP(df, 3, -1)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#def corr_ret_turnd_post_7D(df):
#    a = corr_shift_ret_turnd_nP(df, 7, -1)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#def corr_ret_turnd_post_14D(df):
#    a = corr_shift_ret_turnd_nP(df, 14, -1)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#def corr_ret_turnd_post_1M(df):
#    a = corr_shift_ret_turnd_nP(df, 30, -1)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#def corr_ret_turnd_prior_3D(df):
#    a = corr_shift_ret_turnd_nP(df, 3, 1)
#    a.columns = ['corr_ret_turnd_3D']
#    return a

#def corr_ret_turnd_prior_7D(df):
#    a = corr_shift_ret_turnd_nP(df, 7, 1)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#def corr_ret_turnd_prior_14D(df):
#    a = corr_shift_ret_turnd_nP(df, 14, 1)
#    a.columns = ['corr_ret_turnd_3D']
#    return a
#
#def corr_ret_turnd_prior_1M(df):
#    a = corr_shift_ret_turnd_nP(df, 30, 1)
#    a.columns = ['corr_ret_turnd_3D']
#    return a




# 非量价因子：包含规模因子、供应因子、合约因子、情绪因子

    '''
规模因子：市值相关因子，共8个
    '''


    def alpha_market_cap_nP(self, n=10):
        """
        市值因子过去n个时间单位的平均值
        market_cap = close * circulating_supply
        """
        
        return (self.marketcap.rolling(n).mean()).stack()


    def alpha_log_market_cap_nP(self, n=10):
        """
        市值对数因子过去n个时间单位的平均值
        log_market_cap = log(market_cap)
        """
        
        
        return np.log(self.marketcap.rolling(n).mean()).stack()

    def alpha_size_factor_nP(self, n=10):
        """
        市值因子过去n个时间单位的平均值
        size_factor = market_cap / median(market_cap)
        """


        
        
        return (self.marketcap / self.marketcap).rolling(n).median().stack()


    def alpha_market_cap_pct_nP(self, n=10):
        """
        计算每只股票在过去n个期间的市值增长率
        market_cap_pct = (current_market_cap - previous_market_cap) / previous_market_cap
        """
        
        return (self.marketcap.pct_change(n)).stack()

# def market_cap_pct_1D(df):
#     return market_cap_pct_nD(df, 1).rename(columns={'market_cap_pct_1D': 'market_cap_pct_1D'})

# def market_cap_pct_3D(df):
#     return market_cap_pct_nD(df, 3).rename(columns={'market_cap_pct_3D': 'market_cap_pct_3D'})

# def market_cap_pct_7D(df):
#     return market_cap_pct_nD(df, 7).rename(columns={'market_cap_pct_7D': 'market_cap_pct_7D'})

# def market_cap_pct_14D(df):
#     return market_cap_pct_nD(df, 14).rename(columns={'market_cap_pct_14D': 'market_cap_pct_14D'})

# def market_cap_pct_1M(df):
#     return market_cap_pct_nD(df, 30).rename(columns={'market_cap_pct_30D': 'market_cap_pct_1M'})

# def market_cap_pct_3M(df):
#     return market_cap_pct_nD(df, 90).rename(columns={'market_cap_pct_90D': 'market_cap_pct_3M'})

# def market_cap_pct_6M(df):
#     return market_cap_pct_nD(df, 180).rename(columns={'market_cap_pct_180D': 'market_cap_pct_6M'})


    def alpha_market_cap_ratio_nP(self, n=10):
        """
        计算市值占比
        market_cap_ratio = market_cap / total_market_cap
        """
        
        
        return (self.marketcap / self.marketcap.sum(axis=1)).rolling(n).mean().stack()


    def alpha_fdv_nP(self, n=10):
        """
        计算完全稀释价值过去n个时间单位的平均值
        FDV = total_supply * close
        """
        
        return (self.FDV.rolling(n).mean()).stack()


    def alpha_log_fdv_nP(self, n=10):
        """
        计算完全稀释价值的对数因子过去n个时间单位的平均值
        log_fdv = log(total_supply * close)
        """

        return np.log(self.FDV.rolling(n).mean()).stack()


    def alpha_fdv_pct_nD(self, n=10):
        """
        计算过去n期的完全稀释价值增长率
        FDV_pct = (current_FDV - previous_FDV) / previous_FDV
        """

        return (self.FDV.pct_change(n)).stack()

# def fdv_pct_1D(df):
#     return fdv_pct_nD(df, 1).rename(columns={'fdv_pct_1D': 'fdv_pct_1D'})

# def fdv_pct_3D(df):
#     return fdv_pct_nD(df, 3).rename(columns={'fdv_pct_3D': 'fdv_pct_3D'})

# def fdv_pct_7D(df):
#     return fdv_pct_nD(df, 7).rename(columns={'fdv_pct_7D': 'fdv_pct_7D'})

# def fdv_pct_14D(df):
#     return fdv_pct_nD(df, 14).rename(columns={'fdv_pct_14D': 'fdv_pct_14D'})

# def fdv_pct_1M(df):
#     return fdv_pct_nD(df, 30).rename(columns={'fdv_pct_30D': 'fdv_pct_1M'})

# def fdv_pct_3M(df):
#     return fdv_pct_nD(df, 90).rename(columns={'fdv_pct_90D': 'fdv_pct_3M'})

# def fdv_pct_6M(df):
#     return fdv_pct_nD(df, 180).rename(columns={'fdv_pct_180D': 'fdv_pct_6M'})



    '''
供应因子：与代币供应量相关的因子，共4个
    '''

    def alpha_circ_nP(self, n=10):
        """
        计算流通供应量因子过去n个时间单位的平均值   
        circ = circulating_supply
        """
    
        return (self.circulating_supply.rolling(n).mean()).stack()

    def alpha_circ_pct_nP(self, n=10):
        """
        计算过去n期的流通供应量百分比变化
        circ_pct_{n}D = (current_circ - previous_circ) / previous_circ
        """
        
        return (self.circulating_supply.pct_change(n)).stack()

# def circ_pct_1D(df):
#     return circ_pct_nP(df, 1).rename(columns={'circ_pct_1D': 'circ_pct_1D'})

# def circ_pct_3D(df):
#     return circ_pct_nP(df, 3).rename(columns={'circ_pct_3D': 'circ_pct_3D'})

# def circ_pct_7D(df):
#     return circ_pct_nP(df, 7).rename(columns={'circ_pct_7D': 'circ_pct_7D'})

# def circ_pct_14D(df):
#     return circ_pct_nP(df, 14).rename(columns={'circ_pct_14D': 'circ_pct_14D'})

# def circ_pct_1M(df):
#     return circ_pct_nP(df, 30).rename(columns={'circ_pct_30D': 'circ_pct_1M'})

# def circ_pct_3M(df):
#     return circ_pct_nP(df, 90).rename(columns={'circ_pct_90D': 'circ_pct_3M'})

# def circ_pct_6M(df):
#     return circ_pct_nD(df, 180).rename(columns={'circ_pct_180D': 'circ_pct_6M'})


    def alpha_circ_ratio_nP(self, n=10):
        """
        计算流通供应量与总供应量之比过去n个时间单位的平均值 
        circ_ratio = circulating_supply / total_supply
        """
        
        return (self.circulating_supply / self.total_supply).rolling(n).mean().stack()


    def alpha_circ_ratio_pct_nD(self, n=10):
        """
        计算过去n期的流通供应量与总供应量之比的百分比变化
        circ_ratio_pct_{n}D = (current_circ_ratio - previous_circ_ratio) / previous_circ_ratio
        """
       

        return (self.circulating_supply / self.total_supply).pct_change(n)

# def circ_ratio_pct_1D(df):
#     return circ_ratio_pct_nD(df, 1).rename(columns={'circ_ratio_pct_1D': 'circ_ratio_pct_1D'})

# def circ_ratio_pct_3D(df):
#     return circ_ratio_pct_nD(df, 3).rename(columns={'circ_ratio_pct_3D': 'circ_ratio_pct_3D'})

# def circ_ratio_pct_7D(df):
#     return circ_ratio_pct_nD(df, 7).rename(columns={'circ_ratio_pct_7D': 'circ_ratio_pct_7D'})

# def circ_ratio_pct_14D(df):
#     return circ_ratio_pct_nD(df, 14).rename(columns={'circ_ratio_pct_14D': 'circ_ratio_pct_14D'})

# def circ_ratio_pct_1M(df):
#     return circ_ratio_pct_nD(df, 30).rename(columns={'circ_ratio_pct_30D': 'circ_ratio_pct_1M'})

# def circ_ratio_pct_3M(df):
#     return circ_ratio_pct_nD(df, 90).rename(columns={'circ_ratio_pct_90D': 'circ_ratio_pct_3M'})

# def circ_ratio_pct_6M(df):
#     return circ_ratio_pct_nD(df, 180).rename(columns={'circ_ratio_pct_180D': 'circ_ratio_pct_6M'})


    '''
合约因子：与永续合约指标相关的因子，如资金费率、未平仓量、多空比等，共16个
    '''

    def alpha_funding_rate_nP(self, n=10):
        """
        资金费率因子过去n个时间单位的平均值
        """

        return (self.funding_rate.rolling(n).mean()).stack()


    def alpha_funding_rate_pct_nD(self, n=10):
        """
        计算过去n期的资金费率百分比变化
        funding_rate_pct_{n}D = (current_funding_rate - previous_funding_rate) / previous_funding_rate
        """


        return (self.funding_rate.pct_change(n)).stack()
    


# def funding_rate_pct_1D(df):
#     return funding_rate_pct_nD(df, 1).rename(columns={'funding_rate_pct_1D': 'funding_rate_pct_1D'})

# def funding_rate_pct_3D(df):
#     return funding_rate_pct_nD(df, 3).rename(columns={'funding_rate_pct_3D': 'funding_rate_pct_3D'})

# def funding_rate_pct_7D(df):
#     return funding_rate_pct_nD(df, 7).rename(columns={'funding_rate_pct_7D': 'funding_rate_pct_7D'})

# def funding_rate_pct_14D(df):
#     return funding_rate_pct_nD(df, 14).rename(columns={'funding_rate_pct_14D': 'funding_rate_pct_14D'})

# def funding_rate_pct_1M(df):
#     return funding_rate_pct_nD(df, 30).rename(columns={'funding_rate_pct_30D': 'funding_rate_pct_1M'})

# def funding_rate_pct_3M(df):
#     return funding_rate_pct_nD(df, 90).rename(columns={'funding_rate_pct_90D': 'funding_rate_pct_3M'})

# def funding_rate_pct_6M(df):
#     return funding_rate_pct_nD(df, 180).rename(columns={'funding_rate_pct_180D': 'funding_rate_pct_6M'})


    def alpha_oi_nP(self, n=10):
        """
        币本位的未平仓合约数量过去n个时间单位的平均值
        oi = sum_open_interest
        """


        return (self.open_interest.rolling(n).mean()).stack()


    def alpha_log_oi_nP(self, n=10):
        """
        计算币本位未平仓合约数量的对数因子过去n个时间单位的平均值
        log_oi = log(sum_open_interest)
        """


        return (np.log(self.open_interest.rolling(n).mean())).stack()


    def alpha_oi_pct_nD(self, n=10):
        """
        计算过去n期的币本位未平仓合约数量百分比变化
        oi_pct_{n}D = (current_oi - previous_oi) / previous_oi
        """


        return (self.open_interest.pct_change(n)).stack()









# def oi_pct_1D(df):
#     return oi_pct_nD(df, 1).rename(columns={'oi_pct_1D': 'oi_pct_1D'})

# def oi_pct_3D(df):
#     return oi_pct_nD(df, 3).rename(columns={'oi_pct_3D': 'oi_pct_3D'})

# def oi_pct_7D(df):
#     return oi_pct_nD(df, 7).rename(columns={'oi_pct_7D': 'oi_pct_7D'})

# def oi_pct_14D(df):
#     return oi_pct_nD(df, 14).rename(columns={'oi_pct_14D': 'oi_pct_14D'})

# def oi_pct_1M(df):
#     return oi_pct_nD(df, 30).rename(columns={'oi_pct_30D': 'oi_pct_1M'})

# def oi_pct_3M(df):
#     return oi_pct_nD(df, 90).rename(columns={'oi_pct_90D': 'oi_pct_3M'})

# def oi_pct_6M(df):
#     return oi_pct_nD(df, 180).rename(columns={'oi_pct_180D': 'oi_pct_6M'})


    def alpha_oivalue_nP(self, n=10):
        """
        以U计算的未平仓合约总价值因子过去n个时间单位的平均值
        """
    
        return (self.open_interest_value.rolling(n).mean()).stack()


    def alpha_log_oivalue_nP(self, n=10):
        """
        计算U计价的未平仓合约总价值的对数因子过去n个时间单位的平均值
        log_oivalue = log(sum_open_interest_value)
        """
        

        return np.log(self.open_interest_value.rolling(n).mean()).stack()


    def alpha_oivalue_pct_nP(self, n=10):
        """
        计算过去n期的U计价未平仓合约总价值百分比变化
        oivalue_pct_{n}D = (current_oivalue - previous_oivalue) / previous_oivalue
        """
        

        return (self.open_interest_value.pct_change(n)).stack()

# def oivalue_pct_1D(df):
#     return oivalue_pct_nD(df, 1).rename(columns={'oivalue_pct_1D': 'oivalue_pct_1D'})

# def oivalue_pct_3D(df):
#     return oivalue_pct_nD(df, 3).rename(columns={'oivalue_pct_3D': 'oivalue_pct_3D'})

# def oivalue_pct_7D(df):
#     return oivalue_pct_nD(df, 7).rename(columns={'oivalue_pct_7D': 'oivalue_pct_7D'})

# def oivalue_pct_14D(df):
#     return oivalue_pct_nD(df, 14).rename(columns={'oivalue_pct_14D': 'oivalue_pct_14D'})

# def oivalue_pct_1M(df):
#     return oivalue_pct_nD(df, 30).rename(columns={'oivalue_pct_30D': 'oivalue_pct_1M'})

# def oivalue_pct_3M(df):
#     return oivalue_pct_nD(df, 90).rename(columns={'oivalue_pct_90D': 'oivalue_pct_3M'})

# def oivalue_pct_6M(df):
#     return oivalue_pct_nD(df, 180).rename(columns={'oivalue_pct_180D': 'oivalue_pct_6M'})


    def alpha_lsr_account_top_nP(self, n=10):
        """
        以账户数计算的顶级交易者多空比率因子过去n个时间单位的平均值
    """
    

        return (self.count_toptrader_long_short_ratio.rolling(n).mean()).stack()


    def alpha_lsr_account_top_pct_nD(self, n=10):
        """
        计算过去n期以账户数计算的顶级交易者多空比率百分比变化
        lsr_account_top_pct_{n}D = (current_lsr_account_top - previous_lsr_account_top) / previous_lsr_account_top
        """
        

        return (self.count_toptrader_long_short_ratio.pct_change(n)).stack()

# def lsr_account_top_pct_1D(df):
#     return lsr_account_top_pct_nD(df, 1).rename(columns={'lsr_account_top_pct_1D': 'lsr_account_top_pct_1D'})

# def lsr_account_top_pct_3D(df):
#     return lsr_account_top_pct_nD(df, 3).rename(columns={'lsr_account_top_pct_3D': 'lsr_account_top_pct_3D'})

# def lsr_account_top_pct_7D(df):
#     return lsr_account_top_pct_nD(df, 7).rename(columns={'lsr_account_top_pct_7D': 'lsr_account_top_pct_7D'})

# def lsr_account_top_pct_14D(df):
#     return lsr_account_top_pct_nD(df, 14).rename(columns={'lsr_account_top_pct_14D': 'lsr_account_top_pct_14D'})

# def lsr_account_top_pct_1M(df):
#     return lsr_account_top_pct_nD(df, 30).rename(columns={'lsr_account_top_pct_30D': 'lsr_account_top_pct_1M'})

# def lsr_account_top_pct_3M(df):
#     return lsr_account_top_pct_nD(df, 90).rename(columns={'lsr_account_top_pct_90D': 'lsr_account_top_pct_3M'})

# def lsr_account_top_pct_6M(df):
#     return lsr_account_top_pct_nD(df, 180).rename(columns={'lsr_account_top_pct_180D': 'lsr_account_top_pct_6M'})


    def alpha_lsr_position_top_nP(self, n=10):
        """
        以头寸计算的顶级交易者多空比率因子过去n个时间单位的平均值
    """


        return (self.sum_toptrader_long_short_ratio.rolling(n).mean()).stack()


    def alpha_lsr_position_top_pct_nP(self, n=10):
        """
        计算过去n期以头寸计算的顶级交易者多空比率百分比变化
        lsr_position_top_pct_{n}D = (current_lsr_position_top - previous_lsr_position_top) / previous_lsr_position_top
        """
        

        return (self.sum_toptrader_long_short_ratio.pct_change(n)).stack()
    
# def lsr_position_top_pct_1D(df):
#     return lsr_position_top_pct_nD(df, 1).rename(columns={'lsr_position_top_pct_1D': 'lsr_position_top_pct_1D'})

# def lsr_position_top_pct_3D(df):
#     return lsr_position_top_pct_nD(df, 3).rename(columns={'lsr_position_top_pct_3D': 'lsr_position_top_pct_3D'})

# def lsr_position_top_pct_7D(df):
#     return lsr_position_top_pct_nD(df, 7).rename(columns={'lsr_position_top_pct_7D': 'lsr_position_top_pct_7D'})

# def lsr_position_top_pct_14D(df):
#     return lsr_position_top_pct_nD(df, 14).rename(columns={'lsr_position_top_pct_14D': 'lsr_position_top_pct_14D'})

# def lsr_position_top_pct_1M(df):
#     return lsr_position_top_pct_nD(df, 30).rename(columns={'lsr_position_top_pct_30D': 'lsr_position_top_pct_1M'})

# def lsr_position_top_pct_3M(df):
#     return lsr_position_top_pct_nD(df, 90).rename(columns={'lsr_position_top_pct_90D': 'lsr_position_top_pct_3M'})

# def lsr_position_top_pct_6M(df):
#     return lsr_position_top_pct_nD(df, 180).rename(columns={'lsr_position_top_pct_180D': 'lsr_position_top_pct_6M'})


    def alpha_lsr_account_nP(self, n=10):
        """
        市场总体以账户数计算的多空比率因子过去n个时间单位的平均值
        lsr_account = count_long_short_ratio
        """

        return (self.count_long_short_ratio.rolling(n).mean()).stack()


    def alpha_lsr_account_pct_nP(self, n=10):
        """
        计算过去n期市场总体以账户数计算的多空比率百分比变化
        lsr_account_pct_{n}D = (current_lsr_account - previous_lsr_account) / previous_lsr_account
        """
        

        return (self.count_long_short_ratio.pct_change(n)).stack()

# def lsr_account_pct_1D(df):
#     return lsr_account_pct_nD(df, 1).rename(columns={'lsr_account_pct_1D': 'lsr_account_pct_1D'})

# def lsr_account_pct_3D(df):
#     return lsr_account_pct_nD(df, 3).rename(columns={'lsr_account_pct_3D': 'lsr_account_pct_3D'})

# def lsr_account_pct_7D(df):
#     return lsr_account_pct_nD(df, 7).rename(columns={'lsr_account_pct_7D': 'lsr_account_pct_7D'})

# def lsr_account_pct_14D(df):
#     return lsr_account_pct_nD(df, 14).rename(columns={'lsr_account_pct_14D': 'lsr_account_pct_14D'})

# def lsr_account_pct_1M(df):
#     return lsr_account_pct_nD(df, 30).rename(columns={'lsr_account_pct_30D': 'lsr_account_pct_1M'})

# def lsr_account_pct_3M(df):
#     return lsr_account_pct_nD(df, 90).rename(columns={'lsr_account_pct_90D': 'lsr_account_pct_3M'})

# def lsr_account_pct_6M(df):
#     return lsr_account_pct_nD(df, 180).rename(columns={'lsr_account_pct_180D': 'lsr_account_pct_6M'})


    def alpha_lsr_takerVol_nP(self, n=10):
        """
        市价单买入卖出量的多空比率因子过去n个时间单位的平均值
        lsr_takerVol = sum_taker_long_short_vol_ratio
        """

        return (self.taker_long_short_vol_ratio.rolling(n).mean()).stack()


    def alpha_lsr_takerVol_pct_nD(self, n=10):
        """
        计算过去n期的市价单买入卖出量的多空比率百分比变化
        lsr_takerVol_pct_{n}D = (current_lsr_takerVol - previous_lsr_takerVol) / previous_lsr_takerVol
        """
    

        return (self.taker_long_short_vol_ratio.pct_change(n)).stack()

# def lsr_takerVol_pct_1D(df):
#     return lsr_takerVol_pct_nD(df, 1).rename(columns={'lsr_takerVol_pct_1D': 'lsr_takerVol_pct_1D'})

# def lsr_takerVol_pct_3D(df):
#     return lsr_takerVol_pct_nD(df, 3).rename(columns={'lsr_takerVol_pct_3D': 'lsr_takerVol_pct_3D'})

# def lsr_takerVol_pct_7D(df):
#     return lsr_takerVol_pct_nD(df, 7).rename(columns={'lsr_takerVol_pct_7D': 'lsr_takerVol_pct_7D'})

# def lsr_takerVol_pct_14D(df):
#     return lsr_takerVol_pct_nD(df, 14).rename(columns={'lsr_takerVol_pct_14D': 'lsr_takerVol_pct_14D'})

# def lsr_takerVol_pct_1M(df):
#     return lsr_takerVol_pct_nD(df, 30).rename(columns={'lsr_takerVol_pct_30D': 'lsr_takerVol_pct_1M'})

# def lsr_takerVol_pct_3M(df):
#     return lsr_takerVol_pct_nD(df, 90).rename(columns={'lsr_takerVol_pct_90D': 'lsr_takerVol_pct_3M'})

# def lsr_takerVol_pct_6M(df):
#     return lsr_takerVol_pct_nD(df, 180).rename(columns={'lsr_takerVol_pct_180D': 'lsr_takerVol_pct_6M'})



    '''
    情绪因子：与未平仓合约量相关比率来衡量市场情绪，共6个
    '''

    def alpha_oi_vol_nP(self,n=10):
        """
        计算未平仓合约与交易量之比因子过去n个时间单位的平均值
        oi_vol = sum_open_interest / volume
        """

        return (self.open_interest / self.volume).rolling(n).mean().stack()


    def alpha_oi_vol_pct_nP(self, n=10):
        """
        计算过去n期的未平仓合约与交易量之比的百分比变化
        oi_vol_pct_{n}D = (current_oi_vol - previous_oi_vol) / previous_oi_vol
        """

        return (self.open_interest / self.volume).pct_change(n).stack()

# def oi_vol_pct_1D(df):
#     return oi_vol_pct_nD(df, 1).rename(columns={'oi_vol_pct_1D': 'oi_vol_pct_1D'})

# def oi_vol_pct_3D(df):
#     return oi_vol_pct_nD(df, 3).rename(columns={'oi_vol_pct_3D': 'oi_vol_pct_3D'})

# def oi_vol_pct_7D(df):
#     return oi_vol_pct_nD(df, 7).rename(columns={'oi_vol_pct_7D': 'oi_vol_pct_7D'})

# def oi_vol_pct_14D(df):
#     return oi_vol_pct_nD(df, 14).rename(columns={'oi_vol_pct_14D': 'oi_vol_pct_14D'})

# def oi_vol_pct_1M(df):
#     return oi_vol_pct_nD(df, 30).rename(columns={'oi_vol_pct_30D': 'oi_vol_pct_1M'})

# def oi_vol_pct_3M(df):
#     return oi_vol_pct_nD(df, 90).rename(columns={'oi_vol_pct_90D': 'oi_vol_pct_3M'})

# def oi_vol_pct_6M(df):
#     return oi_vol_pct_nD(df, 180).rename(columns={'oi_vol_pct_180D': 'oi_vol_pct_6M'})


    def alpha_oi_mc_nP(self, n=10):
        """
        计算未平仓合约与流通供应量之比因子过去n个时间单位的平均值
        oi_mc = sum_open_interest / circulating_supply
        """


        return (self.open_interest / self.circulating_supply).rolling(n).mean().stack()


    def alpha_oi_mc_pct_nD(self, n=10):
        """
        计算过去n期的未平仓合约量与流通供应量之比的百分比变化
        oi_mc_pct_{n}D = (current_oi_mc - previous_oi_mc) / previous_oi_mc
        """
     

        return (self.open_interest / self.circulating_supply).pct_change(n).stack()

# def oi_mc_pct_1D(df):
#     return oi_mc_pct_nD(df, 1).rename(columns={'oi_mc_pct_1D': 'oi_mc_pct_1D'})

# def oi_mc_pct_3D(df):
#     return oi_mc_pct_nD(df, 3).rename(columns={'oi_mc_pct_3D': 'oi_mc_pct_3D'})

# def oi_mc_pct_7D(df):
#     return oi_mc_pct_nD(df, 7).rename(columns={'oi_mc_pct_7D': 'oi_mc_pct_7D'})

# def oi_mc_pct_14D(df):
#     return oi_mc_pct_nD(df, 14).rename(columns={'oi_mc_pct_14D': 'oi_mc_pct_14D'})

# def oi_mc_pct_1M(df):
#     return oi_mc_pct_nD(df, 30).rename(columns={'oi_mc_pct_30D': 'oi_mc_pct_1M'})

# def oi_mc_pct_3M(df):
#     return oi_mc_pct_nD(df, 90).rename(columns={'oi_mc_pct_90D': 'oi_mc_pct_3M'})

# def oi_mc_pct_6M(df):
#     return oi_mc_pct_nD(df, 180).rename(columns={'oi_mc_pct_180D': 'oi_mc_pct_6M'})


    def alpha_oi_fdv_nP(self, n=10):
        """
        计算未平仓合约量与总供应量之比因子过去n个时间单位的平均值
        oi_fdv = sum_open_interest / total_supply
        """
        

        return (self.open_interest / self.total_supply).rolling(n).mean().stack()




    def alpha_oi_fdv_pct_nP(self, n=10):
        """
        计算过去n期的未平仓合约量与总供应量之比的百分比变化
        oi_fdv_pct_{n}D = (current_oi_fdv - previous_oi_fdv) / previous_oi_fdv
        """
        

        return (self.open_interest / self.total_supply).pct_change(n).stack()

# def oi_fdv_pct_1D(df):
#     return oi_fdv_pct_nD(df, 1).rename(columns={'oi_fdv_pct_1D': 'oi_fdv_pct_1D'})

# def oi_fdv_pct_3D(df):
#     return oi_fdv_pct_nD(df, 3).rename(columns={'oi_fdv_pct_3D': 'oi_fdv_pct_3D'})

# def oi_fdv_pct_7D(df):
#     return oi_fdv_pct_nD(df, 7).rename(columns={'oi_fdv_pct_7D': 'oi_fdv_pct_7D'})

# def oi_fdv_pct_14D(df):
#     return oi_fdv_pct_nD(df, 14).rename(columns={'oi_fdv_pct_14D': 'oi_fdv_pct_14D'})

# def oi_fdv_pct_1M(df):
#     return oi_fdv_pct_nD(df, 30).rename(columns={'oi_fdv_pct_30D': 'oi_fdv_pct_1M'})

# def oi_fdv_pct_3M(df):
#     return oi_fdv_pct_nD(df, 90).rename(columns={'oi_fdv_pct_90D': 'oi_fdv_pct_3M'})

# def oi_fdv_pct_6M(df):
#     return oi_fdv_pct_nD(df, 180).rename(columns={'oi_fdv_pct_180D': 'oi_fdv_pct_6M'})



    '''
    from Dabu
    '''

### 动量因子（4个）

    # MtmMean_v12 指标
    def alpha_mtm_mean_nP(self, n=14):
        """
        计算因子：加权的动量均值（MtmMean）
        因子含义：通过计算收盘价的动量，并结合买入成交量的指数加权平均，来衡量市场中的趋势力量。

        Parameters:
        n (int): 用于计算动量和指数加权平均的周期长度。默认值为 14。
        """

        mtm = self.close / self.close.shift(n) - 1

        ewm_volume = self.taker_buy_quote_volume.ewm(span=n, adjust=False).mean()

        mtm_mean_v12 = mtm * self.taker_buy_quote_volume / ewm_volume

        mtm_mean_v12 = mtm_mean_v12.ewm(span=n, adjust=False).mean()

        return mtm_mean_v12


# def mtm_mean_v12_1D(df):
#     return mtm_mean_nP(df, 1).rename(columns={f'mtm_mean_v12_1D': 'mtm_mean_v12_1D'})

# def mtm_mean_v12_3D(df):
#     return mtm_mean_nP(df, 3).rename(columns={f'mtm_mean_v12_3D': 'mtm_mean_v12_3D'})

# def mtm_mean_v12_7D(df):
#     return mtm_mean_nP(df, 7).rename(columns={f'mtm_mean_v12_7D': 'mtm_mean_v12_7D'})

# def mtm_mean_v12_14D(df):
#     return mtm_mean_nP(df, 14).rename(columns={f'mtm_mean_v12_14D': 'mtm_mean_v12_14D'})

# def mtm_mean_v12_1M(df):
#     return mtm_mean_nP(df, 30).rename(columns={f'mtm_mean_v12_30D': 'mtm_mean_v12_1M'})

# def mtm_mean_v12_3M(df):
#     return mtm_mean_nP(df, 90).rename(columns={f'mtm_mean_v12_90D': 'mtm_mean_v12_3M'})

# def mtm_mean_v12_6M(df):
#     return mtm_mean_nP(df, 180).rename(columns={f'mtm_mean_v12_180D': 'mtm_mean_v12_6M'})





# Bias

    def alpha_bias_ema_nP(self, n=12):
        """
        计算过去n期的基于均线偏离的加权动量均值变化
        bias_ema_{n}D = EMA((close / MA(close, n) - 1) * (quote_volume / MA(quote_volume, n)), n)
        
        因子含义：该因子通过计算收盘价相对于均线的偏离程度，并结合交易量的加权移动平均，来衡量市场的偏离程度和趋势强度。
        
        Parameters:
        n (int): 用于计算移动平均和指数加权平均的周期长度。
        """
        

        ma = self.close.rolling(n, min_periods=1).mean()
        mtm = (self.close / ma - 1) * self.quote_volume / self.quote_volume.rolling(n, min_periods=1).mean()

        return mtm.ewm(span=n, adjust=False).mean().stack()


# def bias_ema_1D(df):
#     return bias_ema_nP(df, 1).rename(columns={f'bias_ema_1D': 'bias_ema_1D'})

# def bias_ema_3D(df):
#     return bias_ema_nP(df, 3).rename(columns={f'bias_ema_3D': 'bias_ema_3D'})

# def bias_ema_7D(df):
#     return bias_ema_nP(df, 7).rename(columns={f'bias_ema_7D': 'bias_ema_7D'})

# def bias_ema_14D(df):
#     return bias_ema_nP(df, 14).rename(columns={f'bias_ema_14D': 'bias_ema_14D'})

# def bias_ema_1M(df):
#     return bias_ema_nP(df, 30).rename(columns={f'bias_ema_30D': 'bias_ema_1M'})

# def bias_ema_3M(df):
#     return bias_ema_nP(df, 90).rename(columns={f'bias_ema_90D': 'bias_ema_3M'})

# def bias_ema_6M(df):
#     return bias_ema_nP(df, 180).rename(columns={f'bias_ema_180D': 'bias_ema_6M'})




    # MaK
    def alpha_ma_k_nP(self, n=12, m=1):
        """
        计算因子：基于移动平均的变化率（MaK）
        因子含义：通过计算移动平均值的变化率，衡量市场趋势的波动强度。

        Parameters:
        n (int): 用于计算移动平均和变化率的周期长度。
        m (int): 用于计算变化率的周期长度。
        """
        
        ma = self.close.rolling(n, min_periods=1).mean()
        mak = (ma / ma.shift(m) - 1) * 1000

        return mak.rolling(n, min_periods=1).mean()




# def ma_k_1D(df):
#     return ma_k_nP(df, 1).rename(columns={f'ma_k_1D': 'ma_k_1D'})

# def ma_k_3D(df):
#     return ma_k_nP(df, 3).rename(columns={f'ma_k_3D': 'ma_k_3D'})

# def ma_k_7D(df):
#     return ma_k_nP(df, 7).rename(columns={f'ma_k_7D': 'ma_k_7D'})

# def ma_k_14D(df):
#     return ma_k_nP(df, 14).rename(columns={f'ma_k_14D': 'ma_k_14D'})

# def ma_k_1M(df):
#     return ma_k_nP(df, 30).rename(columns={f'ma_k_30D': 'ma_k_1M'})

# def ma_k_3M(df):
#     return ma_k_nP(df, 90).rename(columns={f'ma_k_90D': 'ma_k_90D'})

# def ma_k_6M(df):
#     return ma_k_nP(df, 180).rename(columns={f'ma_k_180D': 'ma_k_6M'})




# 动量、主动成交占比和波动率综合因子
    def alpha_mtm_volatility_ratio_nP(self, n=12):
        """
        计算因子：动量 * 主动成交占比 * 波动率（MtmVolatilityRatio）
        因子含义：通过结合动量、主动成交占比以及波动率，来衡量市场中的趋势强度和成交积极性。

        Parameters:
        n (int): 用于计算各种移动平均和动量的周期长度。
        """

        # 计算动量
        mtm = self.close / self.close.shift(n) - 1

        # 主动成交占比
        volume = self.quote_volume.rolling(n, min_periods=1).sum()
        buy_volume = self.taker_buy_quote_volume.rolling(n, min_periods=1).sum()
        taker_buy_ratio = buy_volume / volume

        # 波动率因子
        c1 = self.high - self.low
        c2 = abs(self.high - self.close.shift(1))
        c3 = abs(self.low - self.close.shift(1))
        tr = np.maximum(c1,c2,c3)
        atr = tr.rolling(n, min_periods=1).mean()
        avg_price = self.close.rolling(n, min_periods=1).mean()
        wd_atr = atr / avg_price

        # 动量 * 主动成交占比 * 波动率
        mtm = mtm * taker_buy_ratio * wd_atr

        return mtm.rolling(n, min_periods=1).mean()

# def mtm_volatility_ratio_1D(df):
#     return mtm_volatility_ratio_nP(df, 1).rename(columns={f'mtm_volatility_ratio_1D': 'mtm_volatility_ratio_1D'})

# def mtm_volatility_ratio_3D(df):
#     return mtm_volatility_ratio_nP(df, 3).rename(columns={f'mtm_volatility_ratio_3D': 'mtm_volatility_ratio_3D'})

# def mtm_volatility_ratio_7D(df):
#     return mtm_volatility_ratio_nP(df, 7).rename(columns={f'mtm_volatility_ratio_7D': 'mtm_volatility_ratio_7D'})

# def mtm_volatility_ratio_14D(df):
#     return mtm_volatility_ratio_nP(df, 14).rename(columns={f'mtm_volatility_ratio_14D': 'mtm_volatility_ratio_14D'})

# def mtm_volatility_ratio_1M(df):
#     return mtm_volatility_ratio_nP(df, 30).rename(columns={f'mtm_volatility_ratio_30D': 'mtm_volatility_ratio_1M'})

# def mtm_volatility_ratio_3M(df):
#     return mtm_volatility_ratio_nP(df, 90).rename(columns={f'mtm_volatility_ratio_90D': 'mtm_volatility_ratio_3M'})

# def mtm_volatility_ratio_6M(df):
#     return mtm_volatility_ratio_nP(df, 180).rename(columns={f'mtm_volatility_ratio_180D': 'mtm_volatility_ratio_6M'})




### 流动性因子（1个）

# 流动性溢价因子

    def alpha_shortest_path_liquidity_nP(self, n=12):
        """
        计算过去n期的最短路径流动性溢价因子
        shortest_path_liquidity_{n}D = 成交额 / (最短路径 / 开盘价) 的移动平均
        
        因子含义：该因子通过计算市场的最短路径并结合成交额，来衡量市场的流动性溢价情况。
        
        Parameters:
        n (int): 用于计算移动平均的周期长度。
        """

        # 计算盘中最短路径
        open_low_high_close = self.open - self.low + self.high - self.low + self.high - self.close
        open_high_low_close = self.high - self.open + self.high - self.low + self.close - self.low
        intraday_shortest_path = np.minimum(open_low_high_close, open_high_low_close)

        # 计算最短路径和标准化的最短路径
        shortest_path = intraday_shortest_path + (self.open - self.close.shift(1)).abs()
        standardized_shortest_path = shortest_path / self.open

        # 计算流动性溢价因子
        liquidity_premium = self.volume / self.circulating_supply / standardized_shortest_path

        # 计算流动性溢价因子的移动平均
        return liquidity_premium.rolling(n, min_periods=1).mean()
        
    

# def shortest_path_liquidity_3D(df):
#     return shortest_path_liquidity_nP(df, 3).rename(columns={f'shortest_path_liquidity_3D': 'shortest_path_liquidity_3D'})

# def shortest_path_liquidity_5D(df):
#     return shortest_path_liquidity_nP(df, 5).rename(columns={f'shortest_path_liquidity_5D': 'shortest_path_liquidity_5D'})

# def shortest_path_liquidity_13D(df):
#     return shortest_path_liquidity_nP(df, 13).rename(columns={f'shortest_path_liquidity_13D': 'shortest_path_liquidity_13D'})

# def shortest_path_liquidity_22D(df):
#     return shortest_path_liquidity_nP(df, 22).rename(columns={f'shortest_path_liquidity_22D': 'shortest_path_liquidity_22D'})

# def shortest_path_liquidity_1M(df):
#     return shortest_path_liquidity_nP(df, 30).rename(columns={f'shortest_path_liquidity_30D': 'shortest_path_liquidity_1M'})

# def shortest_path_liquidity_3M(df):
#     return shortest_path_liquidity_nP(df, 90).rename(columns={f'shortest_path_liquidity_90D': 'shortest_path_liquidity_3M'})

# def shortest_path_liquidity_6M(df):
#     return shortest_path_liquidity_nP(df, 180).rename(columns={f'shortest_path_liquidity_180D': 'shortest_path_liquidity_6M'})



    def alpha_liquidity_premium_rolling_std_nP(self, n=10):
            """
            计算因子：流动溢价的滚动标准差
            因子含义：流动性溢价的波动情况

            Parameters:
            n (int): 用于计算各种移动标准差的周期长度。
            """
            
            route_1 = 2 * (self.high - self.low) + (self.open - self.close)
            route_2 = 2 * (self.high - self.low) + (self.close - self.open)
            cond = route_1 > route_2
            temp = np.where(cond, route_2, route_1)
            temp = temp / self.open
            liquidity_premium = self.quote_volume / temp
            
            return liquidity_premium.rolling(n, min_periods=2).std()
            


    def alpha_close_pct_change(self, n=10):
        """
        计算因子：
        因子含义：

        Parameters:
        n (int): 用于计算各种移动标准差的周期长度。
        """
        return self.close.pct_change(n)
    
    """
    Transform from CTA factor
    """

    def alpha_cta_adapt_bolling(self, n=7, cta_signal=0):
        """
        this factor transform from cta adapt bolling signal
        Editer: zqli
        factor = (close / median - 1) + cta_position + cta_signal
        """
        close = self.close
        median = close.rolling(n, min_periods=1).mean()
        std = close.rolling(n, min_periods=1).std(ddof=0)  # ddof代表标准差自由度
        z_score = abs(close - median) / std
        m = z_score.rolling(n, min_periods=1).mean().shift()
        upper = median + m * std
        lower = median - m * std
        median.fillna(method='backfill', inplace=True)
        std.fillna(method='backfill', inplace=True)
        z_score.fillna(method='backfill', inplace=True)
        m.fillna(method='backfill', inplace=True)
        upper.fillna(method='backfill', inplace=True)
        lower.fillna(method='backfill', inplace=True)
        bias = close / median - 1
        bias_pct = abs(bias).rolling(window=n, min_periods=1).max().shift()

        # 描述上穿上轨
        condition1 = close > upper  # 当前K线的收盘价 > 上轨
        condition2 = close.shift(1) <= upper.shift(1)  # 之前K线的收盘价 <= 上轨

        signal_long = pd.DataFrame(index=condition1.index, columns=condition1.columns)
        signal_long[condition1 & condition2] = 1

        # 描述下穿中轨
        condition1 = close < median  # 当前K线的收盘价 < 中轨
        condition2 = close.shift(1) >= median.shift(1)  # 之前K线的收盘价 >= 中轨

        # signal_long.where(~(condition1 & condition2), 0, inplace=True)
        signal_long[condition1 & condition2] = 0

        # 描述下穿下轨
        condition1 = close < lower  # 当前K线的收盘价 < 下轨
        condition2 = close.shift(1) >= lower.shift(1)  # 之前K线的收盘价 >= 下轨
        signal_short = pd.DataFrame(index=condition1.index, columns=condition1.columns)
        # signal_short.where(~(condition1 & condition2), -1, inplace=True)
        signal_short[condition1 & condition2] = -1
        # df.loc[condition1 & condition2,
        #        'signal_short'] = -1  # 将产生做空信号的那根K线的signal设置为-1，-1代表做空

        # 上穿中轨
        condition1 = close > median  # 当前K线的收盘价 > 中轨
        condition2 = close.shift(1) <= median.shift(1)  # 之前K线的收盘价 <= 中轨
        # df.loc[condition1 & condition2,
        #        'signal_short'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓
        # signal_short.where(~(condition1 & condition2), 0, inplace=True)
        signal_short[condition1 & condition2] = 0


        signal_short.fillna(method='ffill', inplace=True)
        signal_long.fillna(method='ffill', inplace=True)


        # signal = df[['signal_long', 'signal_short']].sum(axis=1)
        signal = signal_long + signal_short
        signal.fillna(value=0, inplace=True)
        # signal 已经是 position 

        raw_signal = signal
        # temp 是考虑了某种止盈止损方法
        temp = deepcopy(signal)
        condition1 = (signal == 1)
        condition2 = (bias > bias_pct)
        # df.loc[condition1 & condition2, 'temp'] = None
        temp[condition1 & condition2] = None
        condition1 = (signal == -1)
        condition2 = (bias < -bias_pct)
        # df.loc[condition1 & condition2, 'temp'] = None
        temp[condition1 & condition2] = None
        condition1 = (signal != signal.shift(1))
        condition2 = (temp.isnull())
        # df.loc[condition1 & condition2, 'temp'] = 0
        temp[condition1 & condition2] = 0
        temp.fillna(method='ffill', inplace=True)
        position = deepcopy(temp)
    
        # temp = df[['signal']]
        # temp = temp[temp['signal'] != temp['signal'].shift(1)]
        temp = temp[temp != temp.shift(1)]
        signal = temp # signal 是开平多空仓的信号，1表示开多仓，0表示平仓，-1表示开空仓，NaN表示无操作
        # signal = temp
        # df[factor_name] = signal
        # # 从前往后填充开仓信号，1,0，或者-1. ffile不会带来未来函数
        # df[factor_name] = df[factor_name].fillna(method='ffill', )
        # return df
        if cta_signal:
            return signal
        ########
        # 改写格式后进一步改写为 多因子中性策略因子，不执行一下代码，直接返回 signal 则就是CTA的开平多空仓信号结果
        ########
        factor_value = (close - median) / (upper - median) + position + signal.fillna(0)
        return factor_value

    def alpha_cta_adaptboll_with_cci(self, n=7, n2=7*35):

        '''
        this factor transform from cta adaptboll_with_cci signal
        Editer: zqli
        factor = (indicator / median - 1) + cta_position + cta_signal
        indicator = cci_atr + cci_atr_mean
        '''

        close = self.close
        median = close.rolling(window=n2).mean()
        std = close.rolling(n2, min_periods=1).std(ddof=0)  # ddof代表标准差自由度
        z_score = abs(close - median) / std
        m = z_score.rolling(window=n2).mean()
        upper = median + std * m
        lower = median - std * m

        condition_long = close > upper
        condition_short = close < lower

        # indicator = 'cci_atr'  #短线

        cci_atr_mean, cci_atr = self.helper_signal_cci(n)
        indicator = cci_atr * cci_atr_mean

        median = indicator.rolling(window=n).mean()
        std = indicator.rolling(n, min_periods=1).apply(np.std)  # 配合精度修改std
        z_score = abs(indicator - median) / std
        m = z_score.rolling(window=n).min().shift(1)
        up = median + std * m
        dn = median - std * m

        # indicator 上穿 up
        condition1 = indicator > up
        condition2 = indicator.shift(1) <= up.shift(1)
        condition = condition1 & condition2
        # df.loc[condition, 'signal_long'] = 1 (change this line by below)
        signal_long = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_long[condition] = 1

        # indicator 下穿 dn
        condition1 = indicator < dn
        condition2 = indicator.shift(1) >= dn.shift(1)
        condition = condition1 & condition2
        # df.loc[condition, 'signal_short'] = -1 （change this line by below）
        signal_short = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_short[condition] = -1

        condition1 = indicator < median
        condition2 = indicator.shift(1) >= median.shift(1)
        condition = condition1 & condition2
        # df.loc[condition, 'signal_long'] = 0 (change this line by below)
        signal_long[condition] = 0

        condition1 = indicator > median
        condition2 = indicator.shift(1) <= median.shift(1)
        condition = condition1 & condition2
        # df.loc[condition, 'signal_short'] = 0 (change this line by below)
        signal_short[condition] = 0

        # df.loc[condition_long, 'signal_short'] = 0 (change)
        signal_short[condition_long] = 0
        # df.loc[condition_short, 'signal_long'] = 0
        signal_long[condition_short] = 0

        signal_short.fillna(method='ffill', inplace=True)
        signal_long.fillna(method='ffill', inplace=True)
        # df['signal'] = df[['signal_long', 'signal_short']].sum(axis=1) (change)
        position = signal_long + signal_short
        position.fillna(value=0, inplace=True)

        # temp = df[df['signal'].notnull()][['signal']]
        # temp = temp[temp['signal'] != temp['signal'].shift(1)]
        signal = position[position != position.shift(1)]
        
        # df['signal'] = temp['signal']

        # df[factor_name] = df['signal']
        # # 从前往后填充开仓信号，1,0，或者-1. ffile不会带来未来函数
        # df[factor_name] = df[factor_name].fillna(method='ffill', )

        # return df
        ######
        # 接下来把 cta 信号改成 中性多因子值
        ######
        factor_value = (indicator - median) / (upper - median) + position + signal.fillna(0)
        return factor_value

    def helper_signal_cci(self, n=7):
        """
        used for function: alpha_cta_adaptboll_with_cci
        """
        high = self.high
        low = self.low
        close = self.close
        tp = (high + low + close) / 3
        ma = tp.rolling(window=n, min_periods=1).mean()
        md = abs(close - ma).rolling(window=n,
                                                       min_periods=1).mean()
        cci_c = (tp - ma) / md / 0.015

        cci_mean = cci_c.rolling(window=n, min_periods=1).mean()

        ma_h = tp.rolling(window=n, min_periods=1).max()
        md_h = abs(close - ma_h).rolling(window=n,
                                                           min_periods=1).max()
        cci_h = (tp - ma_h) / md_h / 0.015

        ma_l = tp.rolling(window=n, min_periods=1).min()
        md_l = abs(close - ma_l).rolling(window=n,
                                                           min_periods=1).min()
        cci_l = (tp - ma_l) / md_l / 0.015

        cci_c1 = cci_h - cci_l
        cci_c2 = abs(cci_h - cci_c.shift(1))
        cci_c3 = abs(cci_l - cci_c.shift(1))
        # cci_tr = df[['cci_c1', 'cci_c2', 'cci_c3']].max(axis=1)
        combined = pd.concat([cci_c1, cci_c2, cci_c3])
        # 计算每列的最大值
        cci_tr = combined.groupby(combined.index).max()
        cci_atr = cci_tr.rolling(window=n, min_periods=1).mean()

        cci_l_mean = cci_l.rolling(window=n, min_periods=1).mean()
        cci_h_mean = cci_h.rolling(window=n, min_periods=1).mean()
        cci_c_mean = cci_c.rolling(window=n, min_periods=1).mean()
        cci_c1 = cci_h_mean - cci_l_mean
        cci_c2 = abs(cci_h_mean - cci_c_mean.shift(1))
        cci_c3 = abs(cci_l_mean - cci_c_mean.shift(1))
        # cci_tr = df[['cci_c1', 'cci_c2', 'cci_c3']].max(axis=1)
        combined = pd.concat([cci_c1, cci_c2, cci_c3])
        cci_tr = combined.groupby(combined.index).max()
        cci_atr_mean = cci_tr.rolling(window=n, min_periods=1).mean()
        # 删除无关变量
        # df.drop([
        #     'tp', 'ma', 'md', 'cci_c', 'cci_mean', 'ma_h', 'md_h', 'cci_h', 'ma_l',
        #     'md_l', 'cci_l', 'cci_c1', 'cci_c2', 'cci_c3', 'cci_tr', 'cci_l_mean',
        #     'cci_h_mean', 'cci_c_mean'
        # ],
        #         axis=1,
        #         inplace=True)
        return cci_atr_mean, cci_atr


    def alpha_cta_adaptboll_with_mtm_cci_zdf(self, n=7, n2=7*35):
        '''
        indicator: cci_atr + cci_atr_mean + mtm_atr + zdf_atr_mean
        this factor transform from cta adaptboll_with_mtm_cci_zdf signal
        Editer: zqli
        factor = (indicator / median - 1) + cta_position + cta_signal
        '''
        close = self.close
        median = close.rolling(window=n2).mean()
        std = close.rolling(n2, min_periods=1).std(ddof=0)  # ddof代表标准差自由度
        z_score = abs(close - median) / std
        m = z_score.rolling(window=n2).mean()
        upper = median + std * m
        lower = median - std * m

        condition_long = close > upper
        condition_short = close < lower

        high = self.high
        low = self.low
        c1 = high - low
        c2 = abs(high - close.shift(1))
        c3 = abs(low - close.shift(1))
        # tr = df[['c1', 'c2', 'c3']].max(axis=1) (change)
        combined = pd.concat([c1, c2, c3])
        tr = combined.groupby(combined.index).max()
        atr = tr.rolling(window=n, min_periods=1).mean()
        avg_price = close.rolling(window=n, min_periods=1).mean()
        wd_atr = atr / avg_price

        # indicator = 'cci_atr'  #短线
        zdf_atr_mean = self.helper_signal_zhangdiefu_std(n)
        cci_atr_mean, _ = self.helper_signal_cci(n)
        mtm_atr = self.helper_signal_mtm(n)

        # df[indicator] = df[indicator] * df['cci_atr_mean']
        # df[indicator] = df[indicator] * df['mtm_atr']
        # df[indicator] = df[indicator] * df['zdf_atr_mean']
        indicator = zdf_atr_mean * cci_atr_mean * mtm_atr

        median = indicator.rolling(window=n).mean()
        std = indicator.rolling(n, min_periods=1).apply(np.std)  # 配合精度修改std
        z_score = abs(indicator - median) / std
        m = z_score.rolling(window=n).min().shift(1)
        up = median + std * m
        dn = median - std * m

        condition1 = indicator > up
        condition2 = indicator.shift(1) <= up.shift(1)
        condition = condition1 & condition2
        signal_long = pd.DataFrame(index=condition.index, columns=condition.columns)
        # df.loc[condition, 'signal_long'] = 1
        signal_long[condition] = 1

        condition1 = indicator < dn
        condition2 = indicator.shift(1) >= dn.shift(1)
        condition = condition1 & condition2
        # df.loc[condition, 'signal_short'] = -1
        signal_short = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_short[condition] = -1

        condition1 = indicator < median
        condition2 = indicator.shift(1) >= median.shift(1)
        condition = condition1 & condition2
        # df.loc[condition, 'signal_long'] = 0
        signal_long[condition] = 0

        condition1 = indicator > median
        condition2 = indicator.shift(1) <= median.shift(1)
        condition = condition1 & condition2
        # df.loc[condition, 'signal_short'] = 0
        signal_short[condition] = 0

        # df.loc[condition_long, 'signal_short'] = 0
        # df.loc[condition_short, 'signal_long'] = 0
        signal_short[condition_long] = 0
        signal_long[condition_short] = 0

        signal_short.fillna(method='ffill', inplace=True)
        signal_long.fillna(method='ffill', inplace=True)
        # df['signal'] = df[['signal_long', 'signal_short']].sum(axis=1)
        position = signal_long + signal_short
        position.fillna(value=0, inplace=True)

        # temp = df[signal.notnull()][['signal']]
        # temp = temp[temp['signal'] != temp['signal'].shift(1)]
        # signal = temp['signal']
        signal = position[position != position.shift(1)]

        # df[factor_name] = signal
        # 从前往后填充开仓信号，1,0，或者-1. ffile不会带来未来函数
        # df[factor_name] = df[factor_name].fillna(method='ffill', )
        ######
        # 下面转化成中性因子值
        ######
        factor_value = (indicator - median) / (upper - median) + position + signal.fillna(0)
        return factor_value
    


    def helper_signal_zhangdiefu_std(self, n=7):

        # 涨跌幅std，振幅的另外一种形式
        close = self.close
        change = close.pct_change()
        zhf_c = change.rolling(n).std()

        high = self.high
        change = high.pct_change()
        zhf_h = change.rolling(n).std()

        low = self.low
        change = low.pct_change()
        zhf_l = change.rolling(n).std()

        zhf_c1 = zhf_h - zhf_l
        zhf_c2 = abs(zhf_h - zhf_c.shift(1))
        zhf_c3 = abs(zhf_l - zhf_c.shift(1))
        # df['zhf_tr'] = df[['zhf_c1', 'zhf_c2', 'zhf_c3']].max(axis=1) (change to)
        combined = pd.concat([zhf_c1, zhf_c2, zhf_c3])
        zhf_tr = combined.groupby(combined.index).max()
        zdf_atr = zhf_tr.rolling(window=n, min_periods=1).mean()

        # 参考ATR，对MTM mean指标，计算波动率因子
        zdf_l_mean = zhf_l.rolling(window=n, min_periods=1).mean()
        zdf_h_mean = zhf_h.rolling(window=n, min_periods=1).mean()
        zdf_c_mean = zhf_c.rolling(window=n, min_periods=1).mean()
        zdf_c1 = zdf_h_mean - zdf_l_mean
        zdf_c2 = abs(zdf_h_mean - zdf_c_mean.shift(1))
        zdf_c3 = abs(zdf_l_mean - zdf_c_mean.shift(1))
        # df['zdf_tr'] = df[['zdf_c1', 'zdf_c2', 'zdf_c3']].max(axis=1)
        combined = pd.concat([zdf_c1, zdf_c2, zdf_c3])
        zdf_tr = combined.groupby(combined.index).max()
        zdf_atr_mean = zdf_tr.rolling(window=n, min_periods=1).mean()

        # 删除无关变量
        # df.drop([
        #     'zhf_c', 'zhf_h', 'zhf_l', 'zhf_c1', 'zhf_c2', 'zhf_c3', 'zhf_tr',
        #     'zdf_l_mean', 'zdf_h_mean', 'zdf_c_mean'
        # ],
        #         axis=1,
        #         inplace=True)
        return zdf_atr_mean


    def helper_signal_mtm(self, n=7):
        close = self.close

        mtm = close / close.shift(n) - 1
        mtm_mean = mtm.rolling(window=n, min_periods=1).mean()

        low = self.low
        high = self.high
        # 参考ATR，对MTM指标，计算波动率因子
        mtm_l = low / low.shift(n) - 1
        mtm_h = high / high.shift(n) - 1
        mtm_c = close / close.shift(n) - 1
        mtm_c1 = mtm_h - mtm_l
        mtm_c2 = abs(mtm_h - mtm_c.shift(1))
        mtm_c3 = abs(mtm_l - mtm_c.shift(1))
        # mtm_tr = df[['mtm_c1', 'mtm_c2', 'mtm_c3']].max(axis=1)
        combined = pd.concat([mtm_c1, mtm_c2, mtm_c3])
        mtm_tr = combined.groupby(combined.index).max()
        mtm_atr = mtm_tr.rolling(window=n, min_periods=1).mean()

        # 参考ATR，对MTM mean指标，计算波动率因子
        mtm_l_mean = mtm_l.rolling(window=n, min_periods=1).mean()
        mtm_h_mean = mtm_h.rolling(window=n, min_periods=1).mean()
        mtm_c_mean = mtm_c.rolling(window=n, min_periods=1).mean()
        mtm_c1 = mtm_h_mean - mtm_l_mean
        mtm_c2 = abs(mtm_h_mean - mtm_c_mean.shift(1))
        mtm_c3 = abs(mtm_l_mean - mtm_c_mean.shift(1))
        # mtm_tr = df[['mtm_c1', 'mtm_c2', 'mtm_c3']].max(axis=1)
        combined = pd.concat([mtm_c1, mtm_c2, mtm_c3])
        mtm_tr = combined.groupby(combined.index).max()
        mtm_atr_mean = mtm_tr.rolling(window=n, min_periods=1).mean()
        # 删除无关变量
        # df.drop([
        #     'mtm', 'mtm_mean', 'mtm_l', 'mtm_h', 'mtm_c', 'mtm_c1', 'mtm_c2',
        #     'mtm_c3', 'mtm_tr', 'mtm_l_mean', 'mtm_h_mean', 'mtm_c1', 'mtm_c2',
        #     'mtm_c3', 'mtm_tr'
        # ],
        #         axis=1,
        #         inplace=True)
        # return df
        return mtm_atr

    def alpha_cta_adx_dc_tunnel(self, n=7, cta_signal=0):
        '''
        indicator: 
        this factor transform from cta adx_dc_tunnel signal
        Editer: zqli
        factor = (indicator / median - 1) + cta_position + cta_signal
        !!!!! 这里 df 是一个字典，其元素是 dataframe
        '''
        _adx = self.helper_calculate_adx(n)
        adx = (_adx - _adx.rolling(n).min()) / (_adx.rolling(n).max() - _adx.rolling(n).min())
        # indicator = "adx"
        signal, adx, max, mean, position = self.helper_dc_tunnel_formatter(adx, n)

        if cta_signal:
            return signal
        else:
            return (adx - mean) / (max - mean) + position + signal.fillna(0)



    def helper_calculate_adx(self, n=7):
        """
        这里 df 是一个字典
        """
        # Step 1: Calculate the True Range (TR)
        high = self.high
        low = self.low
        close = self.close
       
        temp= np.maximum(high - low,
                              np.maximum(abs(high - close.shift(1)),
                                         abs(low - close.shift(1))))
        temp = pd.DataFrame(index=close.index, columns=close.columns, data=temp)
        TR = temp
        # Step 2: Calculate the Directional Movement (DM)
        temp = np.where((high - high.shift(1)) > (low.shift(1) - low),
                             np.maximum(high - high.shift(1), 0), 0)
        temp = pd.DataFrame(index=close.index, columns=close.columns, data=temp)
        DM_plus = temp
        temp = np.where((low.shift(1) - low) > (high - high.shift(1)),
                             np.maximum(low.shift(1) - low, 0), 0)
        temp = pd.DataFrame(index=close.index, columns=close.columns, data=temp)
        DM_minus = temp
        # Step 3: Calculate the Directional Indicator (DI)
        DI_plus = 100 * (DM_plus.rolling(window=n).sum() / TR.rolling(window=n).sum())
        DI_minus = 100 * (DM_minus.rolling(window=n).sum() / TR.rolling(window=n).sum())

        # Step 4: Calculate the DX
        DX = (abs(DI_plus - DI_minus) / (DI_plus + DI_minus)) * 100

        # Step 5: Calculate the ADX
        _adx = DX.rolling(window=n).mean()

        # Clean up intermediate columns
        #df.drop(['TR', 'DM+', 'DM-', 'DI+', 'DI-', 'DX'], axis=1, inplace=True)

        return _adx


    def helper_dc_tunnel_formatter(self, adx, n=7):
        # 基础dc通道模板

        mean = adx.rolling(n).mean()
        max = adx.rolling(n).max().shift()
        min = adx.rolling(n).min().shift()

        # 做多信号
        condition1 = adx > max
        condition2 = adx.shift() <= max.shift()
        # dictt.loc[condition1 & condition2, 'signal_long'] = 1  # 1代表做多
        signal_long = pd.DataFrame(index=condition1.index, columns=condition1.columns)
        signal_long[condition1 & condition2] = 1
        # 平多信号
        condition1 = adx < mean
        condition2 = adx.shift() >= mean.shift()
        # dictt.loc[condition1 & condition2, 'signal_long'] = 0
        signal_long[condition1 & condition2] = 0
        # 做空信号
        condition1 = adx < min
        condition2 = adx.shift() >= min.shift()
        # dictt.loc[condition1 & condition2, 'signal_short'] = -1
        signal_short = pd.DataFrame(index=condition1.index, columns=condition1.columns)
        signal_short[condition1 & condition2] = -1
        # 平空信号
        condition1 = adx > mean
        condition2 = adx.shift() <= mean.shift()
        # dictt.loc[condition1 & condition2, 'signal_short'] = 0
        signal_short[condition1 & condition2] = 0

        # ===将long和short合并为signal
        signal_short.fillna(method='ffill', inplace=True)
        signal_long.fillna(method='ffill', inplace=True)
        # dictt['signal'] = dictt[['signal_long', 'signal_short']].sum(axis=1)
        position = signal_long + signal_short
        # dictt['signal'].fillna(value=0, inplace=True)
        position.fillna(value=0, inplace=True)

        # temp = dictt[['signal']]
        # temp = temp[temp['signal'] != temp['signal'].shift(1)]
        # dictt['signal'] = temp
        signal = position[position != position.shift(1)]

        ######
        # 接下来计算 中性因子值
        ######
        #signal = (adx - mean) / (max - mean) + position + signal.fillna(0)

        return signal, adx, max, mean, position
    


    def alpha_cta_adx_keltner_channel(self, n=14, cta_signal=0):
        '''
        计算KC
        TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-REF(LOW,1)))
        ATR=MA(TR,N)
        Middle=EMA(CLOSE,20)
        自适应转换
        UPPER=MIDDLE+2*ATR
        LOWER=MIDDLE-2*ATR
        '''
        _adx = self.helper_calculate_adx(n)
    
        adx = _adx / _adx.rolling(n, min_periods=1).mean().shift()

        # 基于指标计算KC通道
        kc_high = adx.rolling(n).max().shift()
        kc_low = adx.rolling(n).min().shift()

        TR = abs(kc_high - kc_low)
        ATR = TR.rolling(n, min_periods=1).mean()
        median = adx.ewm(span=20, min_periods=1, adjust=False).mean()
        z_score = abs(adx - median) / ATR
        m = z_score.rolling(window=n).max().shift(1)
        upper = median + ATR * m
        lower = median - ATR * m

        # 找出做多信号
        condition1 = (adx > upper)
        condition2 = (adx.shift() <= upper.shift(1))
        signal_long = pd.DataFrame(index=condition1.index, columns=condition1.columns)
        signal_long[condition1 & condition2] = 1

        # 找出做多平仓信号
        condition1 = (adx < lower)
        condition2 = (adx.shift() >= lower.shift())
        signal_long[condition1 & condition2] = 0

        # 找出做空信号
        condition1 = (adx < lower)
        condition2 = (adx.shift(1) >= lower.shift(1))
        signal_short = pd.DataFrame(index=condition1.index, columns=condition1.columns)
        signal_short[condition1 & condition2] = -1

        # 找出做空平仓信号
        condition1 = (adx > upper)
        condition2 = (adx.shift() <= upper.shift())
        signal_short[condition1 & condition2] = 0

        # ========================= 固定代码 =========================
        signal_short.fillna(method='ffill', inplace=True)
        signal_long.fillna(method='ffill', inplace=True)
        
        # df['signal'] = df[['signal_long', 'signal_short']].sum(axis=1,
        #                                                        min_count=1,
        #                                                        skipna=True)
        temp = signal_short + signal_long

        #temp = df[df['signal'].notnull()][['signal']]
        signal = temp[temp != temp.shift(1)]
        #df['signal'] = temp['signal']

        # ========================= 固定代码 =========================

        if cta_signal:
            return signal

        position = signal.fillna(method='ffill').fillna(0)

        factor = (adx - median) / (upper - median) + position + signal.fillna(0)
        
        # 删除无关变量
        # df.drop(['TR', 'ATR', 'kc_high', 'kc_low', 'm', 'z_score', 'signal_long', 'signal_short'],
        #         axis=1,
        #         inplace=True)

        return factor
    
    def alpha_cta_amv_bolling(self, n=14, cta_signal=0):
        
        """
        N1=13
        N2=34
        AMOV=VOLUME*(OPEN+CLOSE)/2
        AMV1=SUM(AMOV,N1)/SUM(VOLUME,N1)
        AMV2=SUM(AMOV,N2)/SUM(VOLUME,N2)
        AMV 指标用成交量作为权重对开盘价和收盘价的均值进行加权移动
        平均。成交量越大的价格对移动平均结果的影响越大，AMV 指标减
        小了成交量小的价格波动的影响。当短期 AMV 线上穿/下穿长期 AMV
        线时，产生买入/卖出信号。
        """
        AMOV = self.volume * (self.open + self.close) / 2
        AMV1 = AMOV.rolling(n).sum() / self.volume.rolling(n).sum()
        amv = (AMV1 - AMV1.rolling(n).min()) / (AMV1.rolling(n).max() - AMV1.rolling(n).min()) # 标准化
        
    
        signal, median, indicator, upper, lower = self.helper_bolling_formatter(amv, n)
        position = signal.fillna(method='ffill').fillna(0)

        if cta_signal:
            return signal
        else:
            width = upper - median
            return (indicator - median) / width + position + signal.fillna(0)

    
    def helper_bolling_formatter(self, indicator, n=14):
        # 布林通道模板

        # 使用自适应 m
        median = indicator.rolling(n, min_periods=1).mean()
        std = indicator.rolling(n, min_periods=1).std(ddof=0)  # ddof代表标准差自由度
        z_score = abs(indicator - median) / std
        m = z_score.rolling(n, min_periods=1).mean().shift()

        # ===计算指标
        # 计算均线
        # 计算上轨、下轨道
        upper = median + m * std
        lower = median - m * std

        median.fillna(method='backfill', inplace=True)
        std.fillna(method='backfill', inplace=True)
        z_score.fillna(method='backfill', inplace=True)
        m.fillna(method='backfill', inplace=True)
        upper.fillna(method='backfill', inplace=True)
        lower.fillna(method='backfill', inplace=True)

        # 计算bias
        bias = self.close / median - 1

        # bias_pct 自适应
        bias_pct = abs(bias).rolling(window=n, min_periods=1).max().shift()

        # ===计算原始布林策略信号
        # 找出做多信号
        condition1 = indicator > upper  # 当前K线的收盘价 > 上轨
        condition2 = indicator.shift(1) <= upper.shift(1)  # 之前K线的收盘价 <= 上轨
        condition = condition1 & condition2
        signal_long = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_long[condition] = 1 # 将产生做多信号的那根K线的signal设置为1，1代表做多

        # 找出做多平仓信号
        condition1 = indicator < median  # 当前K线的收盘价 < 中轨
        condition2 = indicator.shift(1) >= median.shift(1)  # 之前K线的收盘价 >= 中轨
        condition = condition1 & condition2
        signal_long[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓

        # 找出做空信号
        condition1 = indicator < lower  # 当前K线的收盘价 < 下轨
        condition2 = indicator.shift(1) >= lower.shift(1)  # 之前K线的收盘价 >= 下轨
        condition = condition1 & condition2
        signal_short = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_short[condition] = -1 # 将产生做空信号的那根K线的signal设置为-1，-1代表做空


        # 找出做空平仓信号
        condition1 = indicator > median  # 当前K线的收盘价 > 中轨
        condition2 = indicator.shift(1) <= median.shift(1)  # 之前K线的收盘价 <= 中轨
        condition = condition1 & condition2
        signal_short[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓
    
        # ===将long和short合并为signal
        position_short = signal_short.fillna(method='ffill').fillna(0)
        position_long = signal_long.fillna(method='ffill').fillna(0)
        position = position_long + position_short
        position.fillna(value=0, inplace=True)
        signal = position[position != position.shift()]
        #df['raw_signal'] = df['signal']
    
        # ===根据bias，修改开仓时间
        #df['temp'] = df['signal']

        # 生成 sltp 信号
        sltp_signal = pd.DataFrame(index=signal.index, columns=signal.columns, data=False)
       
        # 将原始信号做多时，当bias大于阀值，设置为空
        condition1 = (position == 1)
        condition2 = (bias > bias_pct)
        sltp_signal[condition1 & condition2] = True

        # 将原始信号做空时，当bias大于阀值，设置为空
        condition1 = (position == -1)
        condition2 = (bias < -bias_pct)
        sltp_signal[condition1 & condition2] = True

        # 生成 sltp 之后的 signal
        signal[sltp_signal] = 0
        position = signal.fillna(method='ffill').fillna(0)
        signal = signal[position != position.shift()]

        return signal, median, indicator, upper, lower
    
    def helper_bolling_formatter_basic(self, indicator, n=150, m=2):
        # 布林通道模板

        # 使用自适应 m
        median = indicator.rolling(n, min_periods=1).mean()
        std = indicator.rolling(n, min_periods=1).std(ddof=0)  # ddof代表标准差自由度

        # ===计算指标
        # 计算均线
        # 计算上轨、下轨道
        upper = median + m * std
        lower = median - m * std

        median.fillna(method='backfill', inplace=True)
        std.fillna(method='backfill', inplace=True)
        upper.fillna(method='backfill', inplace=True)
        lower.fillna(method='backfill', inplace=True)


        # ===计算原始布林策略信号
        # 找出做多信号
        condition1 = indicator > upper  # 当前K线的收盘价 > 上轨
        condition2 = indicator.shift(1) <= upper.shift(1)  # 之前K线的收盘价 <= 上轨
        condition = condition1 & condition2
        signal_long = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_long[condition] = 1 # 将产生做多信号的那根K线的signal设置为1，1代表做多

        # 找出做多平仓信号
        condition1 = indicator < median  # 当前K线的收盘价 < 中轨
        condition2 = indicator.shift(1) >= median.shift(1)  # 之前K线的收盘价 >= 中轨
        condition = condition1 & condition2
        signal_long[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓

        # 找出做空信号
        condition1 = indicator < lower  # 当前K线的收盘价 < 下轨
        condition2 = indicator.shift(1) >= lower.shift(1)  # 之前K线的收盘价 >= 下轨
        condition = condition1 & condition2
        signal_short = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_short[condition] = -1 # 将产生做空信号的那根K线的signal设置为-1，-1代表做空


        # 找出做空平仓信号
        condition1 = indicator > median  # 当前K线的收盘价 > 中轨
        condition2 = indicator.shift(1) <= median.shift(1)  # 之前K线的收盘价 <= 中轨
        condition = condition1 & condition2
        signal_short[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓
    
        # ===将long和short合并为signal
        position_short = signal_short.fillna(method='ffill').fillna(0)
        position_long = signal_long.fillna(method='ffill').fillna(0)
        position = position_long + position_short
        position.fillna(value=0, inplace=True)
        signal = position[position != position.shift()]
        #df['raw_signal'] = df['signal']
    
        # ===根据bias，修改开仓时间
        #df['temp'] = df['signal']

        position = signal.fillna(method='ffill').fillna(0)
        signal = signal[position != position.shift()]

        return signal, median, indicator, upper, lower
    
    def alpha_cta_basic_bolling(self, n=150, cta_signal=0):
        
        """
        基础版本的布林信号
        n 表示几期均线
        helper_bolling_formatter_basic 有两个参数 n, m
        - n 表示几期均线
        - m 表示几倍std作为布林带宽度
        """
        signal, median, indicator, upper, lower = self.helper_bolling_formatter_basic(self.close, n=n)
        position = signal.fillna(method='ffill').fillna(0)

        if cta_signal:
            return signal
        else:
            width = upper - median
            return (indicator - median) / width + position + signal.fillna(0)
    

    def helper_bolling_formatter_longOnly(self, indicator, n=14):
        # 布林通道模板

        # 使用自适应 m
        median = indicator.rolling(n, min_periods=1).mean()
        std = indicator.rolling(n, min_periods=1).std(ddof=0)  # ddof代表标准差自由度
        z_score = abs(indicator - median) / std
        m = z_score.rolling(n, min_periods=1).mean().shift()

        # ===计算指标
        # 计算均线
        # 计算上轨、下轨道
        upper = median + m * std
        lower = median - m * std

        median.fillna(method='backfill', inplace=True)
        std.fillna(method='backfill', inplace=True)
        z_score.fillna(method='backfill', inplace=True)
        m.fillna(method='backfill', inplace=True)
        upper.fillna(method='backfill', inplace=True)
        lower.fillna(method='backfill', inplace=True)

        # 计算bias
        bias = self.close / median - 1

        # bias_pct 自适应
        bias_pct = abs(bias).rolling(window=n, min_periods=1).max().shift()

        # ===计算原始布林策略信号
        # 找出做多信号
        condition1 = indicator > upper  # 当前K线的收盘价 > 上轨
        condition2 = indicator.shift(1) <= upper.shift(1)  # 之前K线的收盘价 <= 上轨
        condition = condition1 & condition2
        signal_long = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_long[condition] = 1 # 将产生做多信号的那根K线的signal设置为1，1代表做多

        # 找出做多平仓信号
        condition1 = indicator < median  # 当前K线的收盘价 < 中轨
        condition2 = indicator.shift(1) >= median.shift(1)  # 之前K线的收盘价 >= 中轨
        condition = condition1 & condition2
        signal_long[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓

        # 找出做空信号
        condition1 = indicator < lower  # 当前K线的收盘价 < 下轨
        condition2 = indicator.shift(1) >= lower.shift(1)  # 之前K线的收盘价 >= 下轨
        condition = condition1 & condition2
        signal_short = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_short[condition] = -1 # 将产生做空信号的那根K线的signal设置为-1，-1代表做空
        

        # 找出做空平仓信号
        condition1 = indicator > median  # 当前K线的收盘价 > 中轨
        condition2 = indicator.shift(1) <= median.shift(1)  # 之前K线的收盘价 <= 中轨
        # condition = condition1 & condition2
        signal_short[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓
    
        # ===将long和short合并为signal
        position_short = signal_short.fillna(method='ffill').fillna(0)
        position_long = signal_long.fillna(method='ffill').fillna(0)
        position = position_long + position_short
        position.fillna(value=0, inplace=True)
        signal = position[position != position.shift()]
        #df['raw_signal'] = df['signal']
    
        # ===根据bias，修改开仓时间
        #df['temp'] = df['signal']

        # 生成 sltp 信号
        sltp_signal = pd.DataFrame(index=signal.index, columns=signal.columns, data=False)
       
        # 将原始信号做多时，当bias大于阀值，设置为空
        condition1 = (position == 1)
        condition2 = (bias > bias_pct)
        sltp_signal[condition1 & condition2] = True

        # 将原始信号做空时，当bias大于阀值，设置为空
        condition1 = (position == -1)
        condition2 = (bias < -bias_pct)
        sltp_signal[condition1 & condition2] = True

        # 生成 sltp 之后的 signal
        signal[sltp_signal] = 0
        position = signal.fillna(method='ffill').fillna(0)
        signal = signal[position != position.shift()]

        return signal, median, indicator, upper, lower

    
    def alpha_cta_amv_dc_tunnel(self, n=14, cta_signal=False):
        
        """
        N1=13
        N2=34
        AMOV=VOLUME*(OPEN+CLOSE)/2
        AMV1=SUM(AMOV,N1)/SUM(VOLUME,N1)
        AMV2=SUM(AMOV,N2)/SUM(VOLUME,N2)
        AMV 指标用成交量作为权重对开盘价和收盘价的均值进行加权移动
        平均。成交量越大的价格对移动平均结果的影响越大，AMV 指标减
        小了成交量小的价格波动的影响。当短期 AMV 线上穿/下穿长期 AMV
        线时，产生买入/卖出信号。
        """

        AMOV = self.volume * (self.open + self.close) / 2
        AMV1 = AMOV.rolling(n).sum() / self.volume.rolling(n).sum()
        amv = (AMV1 - AMV1.rolling(n).min()) / (AMV1.rolling(n).max() - AMV1.rolling(n).min()) # 标准化
        
        signal, amx, max, mean, position = self.helper_dc_tunnel_formatter(amv, n)

        if cta_signal:
            return signal
        else:
            return (amx - mean) / (max - mean) + position + signal.fillna(0)

        
    def alpha_cta_amv_keltner_channel(self, n=14, cta_signal=0):      
        """
        N1=13
        N2=34
        AMOV=VOLUME*(OPEN+CLOSE)/2
        AMV1=SUM(AMOV,N1)/SUM(VOLUME,N1)
        AMV2=SUM(AMOV,N2)/SUM(VOLUME,N2)
        AMV 指标用成交量作为权重对开盘价和收盘价的均值进行加权移动
        平均。成交量越大的价格对移动平均结果的影响越大，AMV 指标减
        小了成交量小的价格波动的影响。当短期 AMV 线上穿/下穿长期 AMV
        线时，产生买入/卖出信号。
        """
        AMOV = self.volume * (self.open + self.close) / 2
        AMV1 = AMOV.rolling(n).sum() / self.volume.rolling(n).sum()
        amv = (AMV1 - AMV1.rolling(n).min()) / (AMV1.rolling(n).max() - AMV1.rolling(n).min()) # 标准化
        
        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter(amv, n)

        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        
    def alpha_cta_amv_keltner_channel_longOnly(self, n=14, cta_signal=0):      
        """
        N1=13
        N2=34
        AMOV=VOLUME*(OPEN+CLOSE)/2
        AMV1=SUM(AMOV,N1)/SUM(VOLUME,N1)
        AMV2=SUM(AMOV,N2)/SUM(VOLUME,N2)
        AMV 指标用成交量作为权重对开盘价和收盘价的均值进行加权移动
        平均。成交量越大的价格对移动平均结果的影响越大，AMV 指标减
        小了成交量小的价格波动的影响。当短期 AMV 线上穿/下穿长期 AMV
        线时，产生买入/卖出信号。
        """
        AMOV = self.volume * (self.open + self.close) / 2
        AMV1 = AMOV.rolling(n).sum() / self.volume.rolling(n).sum()
        amv = (AMV1 - AMV1.rolling(n).min()) / (AMV1.rolling(n).max() - AMV1.rolling(n).min()) # 标准化
        
        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter_longOnly(amv, n)

        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
    

    def helper_keltner_channel_formatter(self, indicator, n=14):
        '''
        计算KC
        TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-REF(LOW,1)))
        ATR=MA(TR,N)
        Middle=EMA(CLOSE,20)
        自适应转换
        UPPER=MIDDLE+2*ATR
        LOWER=MIDDLE-2*ATR

        close = self.close
        median = close.rolling(window=n2).mean()
        std = close.rolling(n2, min_periods=1).std(ddof=0)  # ddof代表标准差自由度
        z_score = abs(close - median) / std
        m = z_score.rolling(window=n2).mean()
        upper = median + std * m
        lower = median - std * m

        condition_long = close > upper
        condition_short = close < lower

        high = self.high
        low = self.low
        c1 = high - low
        c2 = abs(high - close.shift(1))
        c3 = abs(low - close.shift(1))
        # tr = df[['c1', 'c2', 'c3']].max(axis=1) (change)
        combined = pd.concat([c1, c2, c3])
        tr = combined.groupby(combined.index).max()
        '''
        # 基于指标计算KC通道
        kc_high = indicator.rolling(n).max().shift()
        kc_low = indicator.rolling(n).min().shift()

        TR= abs(kc_high - kc_low)
        ATR = TR.rolling(n, min_periods=1).mean()
        median = indicator.ewm(span=20, min_periods=1, adjust=False).mean()
        z_score = abs(indicator - median) / ATR
        m = z_score.rolling(window=n).max().shift(1)
        upper = median + ATR * m
        lower = median - ATR * m

        # ===计算原始布林策略信号
        # 找出做多信号
        condition1 = indicator > upper  # 当前K线的收盘价 > 上轨
        condition2 = indicator.shift(1) <= upper.shift(1)  # 之前K线的收盘价 <= 上轨
        condition = condition1 & condition2
        signal_long = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_long[condition] = 1 # 将产生做多信号的那根K线的signal设置为1，1代表做多

        # 找出做多平仓信号
        condition1 = indicator < median  # 当前K线的收盘价 < 中轨
        condition2 = indicator.shift(1) >= median.shift(1)  # 之前K线的收盘价 >= 中轨
        condition = condition1 & condition2
        signal_long[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓

        # 找出做空信号
        condition1 = indicator < lower  # 当前K线的收盘价 < 下轨
        condition2 = indicator.shift(1) >= lower.shift(1)  # 之前K线的收盘价 >= 下轨
        condition = condition1 & condition2
        signal_short = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_short[condition] = -1 # 将产生做空信号的那根K线的signal设置为-1，-1代表做空

        # 找出做空平仓信号
        condition1 = indicator > median  # 当前K线的收盘价 > 中轨
        condition2 = indicator.shift(1) <= median.shift(1)  # 之前K线的收盘价 <= 中轨
        condition = condition1 & condition2
        signal_short[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓
    
        # ===将long和short合并为signal
        position_short = signal_short.fillna(method='ffill').fillna(0)
        position_long = signal_long.fillna(method='ffill').fillna(0)
        position = position_long + position_short
        position.fillna(value=0, inplace=True)
        signal = position[position != position.shift()]

        return signal, indicator, median, upper, position
    

    def helper_keltner_channel_formatter_longOnly(self, indicator, n=14):
        '''
        计算KC
        TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-REF(LOW,1)))
        ATR=MA(TR,N)
        Middle=EMA(CLOSE,20)
        自适应转换
        UPPER=MIDDLE+2*ATR
        LOWER=MIDDLE-2*ATR

        close = self.close
        median = close.rolling(window=n2).mean()
        std = close.rolling(n2, min_periods=1).std(ddof=0)  # ddof代表标准差自由度
        z_score = abs(close - median) / std
        m = z_score.rolling(window=n2).mean()
        upper = median + std * m
        lower = median - std * m

        condition_long = close > upper
        condition_short = close < lower

        high = self.high
        low = self.low
        c1 = high - low
        c2 = abs(high - close.shift(1))
        c3 = abs(low - close.shift(1))
        # tr = df[['c1', 'c2', 'c3']].max(axis=1) (change)
        combined = pd.concat([c1, c2, c3])
        tr = combined.groupby(combined.index).max()
        '''
        # 基于指标计算KC通道
        kc_high = indicator.rolling(n).max().shift()
        kc_low = indicator.rolling(n).min().shift()

        TR= abs(kc_high - kc_low)
        ATR = TR.rolling(n, min_periods=1).mean()
        median = indicator.ewm(span=20, min_periods=1, adjust=False).mean()
        z_score = abs(indicator - median) / ATR
        m = z_score.rolling(window=n).max().shift(1)
        upper = median + ATR * m
        lower = median - ATR * m

        # ===计算原始布林策略信号
        # 找出做多信号
        condition1 = indicator > upper  # 当前K线的收盘价 > 上轨
        condition2 = indicator.shift(1) <= upper.shift(1)  # 之前K线的收盘价 <= 上轨
        condition = condition1 & condition2
        signal_long = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_long[condition] = 1 # 将产生做多信号的那根K线的signal设置为1，1代表做多

        # 找出做多平仓信号
        condition1 = indicator < median  # 当前K线的收盘价 < 中轨
        condition2 = indicator.shift(1) >= median.shift(1)  # 之前K线的收盘价 >= 中轨
        condition = condition1 & condition2
        signal_long[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓

        # 找出做空信号
        condition1 = indicator < lower  # 当前K线的收盘价 < 下轨
        condition2 = indicator.shift(1) >= lower.shift(1)  # 之前K线的收盘价 >= 下轨
        condition = condition1 & condition2
        signal_short = pd.DataFrame(index=condition.index, columns=condition.columns)
        signal_short[condition] = -1 # 将产生做空信号的那根K线的signal设置为-1，-1代表做空

        # 找出做空平仓信号
        condition1 = indicator > median  # 当前K线的收盘价 > 中轨
        condition2 = indicator.shift(1) <= median.shift(1)  # 之前K线的收盘价 <= 中轨
        # condition = condition1 & condition2
        signal_short[condition] = 0 # 将产生平仓信号当天的signal设置为0，0代表平仓
    
        # ===将long和short合并为signal
        position_short = signal_short.fillna(method='ffill').fillna(0)
        position_long = signal_long.fillna(method='ffill').fillna(0)
        position = position_long + position_short
        position.fillna(value=0, inplace=True)
        signal = position[position != position.shift()]

        return signal, indicator, median, upper, position


    ###########################
    # 需要进一步改进 angle 的计算
    ###########################
    # def alpha_cta_angle_bolling(self, n=14, cta_signal=0):

    #     df['_angle'] = self.helper_calculate_linearreg_angle(df['close'], n)
    #     df['angle'] = (df['_angle'] - df['_angle'].rolling(n).min()) / (df['_angle'].rolling(n).max() - df['_angle'].rolling(n).min())
    #     indicator = "angle"
    #     df = bolling_formatter(df, n, indicator)

    #     df[factor_name] = df['signal']
    #     # 从前往后填充开仓信号，1,0，或者-1. ffile不会带来未来函数
    #     df[factor_name] = df[factor_name].fillna(method='ffill', )

    #     return df

    # def helper_calculate_linearreg_angle(self, series, timeperiod=14):
    #     def linear_regression_angle(values):
    #         x = np.arange(len(values))
    #         if len(x) == 0:
    #             return np.nan
    #         slope, intercept = np.polyfit(x, values, 1)
    #         angle = np.arctan(slope) * (180 / np.pi)  # Convert slope to angle in degrees
    #         return angle
        
    #     def linear_regression_angle_df(values_sheet):
    #         res = []
    #         for col in values_sheet.columns:
    #             res.append(linear_regression_angle(values_sheet[col]))
    #         return res

    #     return series.rolling(window=timeperiod).apply(linear_regression_angle, raw=True)


    def alpha_cta_ar_bolling(self, n=14, cta_signal=0):

        
        v1 = (self.high - self.open).rolling(n, min_periods=1).sum()
        v2 = (self.open - self.low).rolling(n, min_periods=1).sum()
        ar = 100 * v1 / v2

  
        signal, median, indicator, upper, lower = self.helper_bolling_formatter(ar, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)

    def alpha_cta_ar_bolling_longOnly(self, n=14, cta_signal=0):

        
        v1 = (self.high - self.open).rolling(n, min_periods=1).sum()
        v2 = (self.open - self.low).rolling(n, min_periods=1).sum()
        ar = 100 * v1 / v2

  
        signal, median, indicator, upper, lower = self.helper_bolling_formatter_longOnly(ar, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        
    
    def alpha_cta_ar_dc_tunnel(self, n=14, cta_signal=0):

        
        v1 = (self.high - self.open).rolling(n, min_periods=1).sum()
        v2 = (self.open - self.low).rolling(n, min_periods=1).sum()
        ar = 100 * v1 / v2
  
        signal, indicator, upper, median, position = self.helper_dc_tunnel_formatter(ar, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        

    def alpha_cta_ar_keltner_channel(self, n=14, cta_signal=0):

        
        v1 = (self.high - self.open).rolling(n, min_periods=1).sum()
        v2 = (self.open - self.low).rolling(n, min_periods=1).sum()
        ar = 100 * v1 / v2
  
        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter(ar, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)


    def alpha_cta_atr_bolling(self, n=14, cta_signal=0):

        eps = np.finfo(float).eps
        """
        N=20
        TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        ATR=MA(TR,N)
        MIDDLE=MA(CLOSE,N)
        """

        c1 = self.high - self.low  # HIGH-LOW
        c2 = abs(self.high - self.close.shift(1))  # ABS(HIGH-REF(CLOSE,1)
        c3 = abs(self.low - self.close.shift(1))  # ABS(LOW-REF(CLOSE,1))
        TR = np.maximum(np.maximum(c1, c2), c3)
        TR = pd.DataFrame(index=c1.index, columns=c1.columns, data=TR)
        
        # df['TR'] = df[['c1', 'c2', 'c3']].max(
        #     axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        _ATR = TR.rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
        middle = self.close.rolling(n, min_periods=1).mean()  # MIDDLE=MA(CLOSE,N)
        atr = _ATR / (middle + eps)
        
        signal, median, indicator, upper, lower = self.helper_bolling_formatter(atr, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        

    def alpha_cta_atr_bolling_longOnly(self, n=14, cta_signal=0):

        eps = np.finfo(float).eps
        """
        N=20
        TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        ATR=MA(TR,N)
        MIDDLE=MA(CLOSE,N)
        """

        c1 = self.high - self.low  # HIGH-LOW
        c2 = abs(self.high - self.close.shift(1))  # ABS(HIGH-REF(CLOSE,1)
        c3 = abs(self.low - self.close.shift(1))  # ABS(LOW-REF(CLOSE,1))
        TR = np.maximum(np.maximum(c1, c2), c3)
        TR = pd.DataFrame(index=c1.index, columns=c1.columns, data=TR)
        
        # df['TR'] = df[['c1', 'c2', 'c3']].max(
        #     axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        _ATR = TR.rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
        middle = self.close.rolling(n, min_periods=1).mean()  # MIDDLE=MA(CLOSE,N)
        atr = _ATR / (middle + eps)
        
        signal, median, indicator, upper, lower = self.helper_bolling_formatter_longOnly(atr, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        

    def alpha_cta_atr_dc_tunnel(self, n=14, cta_signal=0):

        eps = np.finfo(float).eps

        """
        N=20
        TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        ATR=MA(TR,N)
        MIDDLE=MA(CLOSE,N)
        """
        c1 = self.high - self.low  # HIGH-LOW
        c2 = abs(self.high - self.close.shift(1))  # ABS(HIGH-REF(CLOSE,1)
        c3 = abs(self.low - self.close.shift(1))  # ABS(LOW-REF(CLOSE,1))
        TR = np.maximum(np.maximum(c1, c2), c3)
        TR = pd.DataFrame(index=c1.index, columns=c1.columns, data=TR)
        
        # df['TR'] = df[['c1', 'c2', 'c3']].max(
        #     axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        _ATR = TR.rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
        middle = self.close.rolling(n, min_periods=1).mean()  # MIDDLE=MA(CLOSE,N)
        atr = _ATR / (middle + eps)

        signal, indicator, upper, median, position = self.helper_dc_tunnel_formatter(atr, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        

    def alpha_cta_atr_keltner_channel(self, n=14, cta_signal=0):

        eps = np.finfo(float).eps

        """
        N=20
        TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        ATR=MA(TR,N)
        MIDDLE=MA(CLOSE,N)
        """
        c1 = self.high - self.low  # HIGH-LOW
        c2 = abs(self.high - self.close.shift(1))  # ABS(HIGH-REF(CLOSE,1)
        c3 = abs(self.low - self.close.shift(1))  # ABS(LOW-REF(CLOSE,1))
        TR = np.maximum(np.maximum(c1, c2), c3)
        TR = pd.DataFrame(index=c1.index, columns=c1.columns, data=TR)
        
        # df['TR'] = df[['c1', 'c2', 'c3']].max(
        #     axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        _ATR = TR.rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
        middle = self.close.rolling(n, min_periods=1).mean()  # MIDDLE=MA(CLOSE,N)
        atr = _ATR / (middle + eps)

        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter(atr, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
    
    def data_sheet_winsorizing(self, sheet_df, rolling_window=3*30*24, upper_lower_multiplier=3):
        high_rolling_mean = sheet_df.rolling(rolling_window).mean()
        high_rolling_std = sheet_df.rolling(rolling_window).std()
        high_upper_win = high_rolling_mean + upper_lower_multiplier * high_rolling_std
        high_lower_win = high_rolling_mean - upper_lower_multiplier * high_rolling_std
        high_winsorized = np.where(sheet_df > high_upper_win, high_upper_win, sheet_df)
        high_winsorized = np.where(high_winsorized < high_lower_win, high_lower_win, high_winsorized)
        high_winsorized = pd.DataFrame(index = sheet_df.index, columns = sheet_df.columns, data=high_winsorized)
        return high_winsorized
        
    def alpha_cta_atr_keltner_channel_winsorized(self, n=14, cta_signal=0):

        eps = np.finfo(float).eps

        """
        N=20
        TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        ATR=MA(TR,N)
        MIDDLE=MA(CLOSE,N)
        """
        high_winsorized = self.data_sheet_winsorizing(self.high)
        low_winsorized = self.data_sheet_winsorizing(self.low)
        close_winsorized = self.data_sheet_winsorizing(self.close)

        
        c1 = high_winsorized - low_winsorized  # HIGH-LOW
        c2 = abs(high_winsorized - close_winsorized.shift(1))  # ABS(HIGH-REF(CLOSE,1)
        c3 = abs(low_winsorized - close_winsorized.shift(1))  # ABS(LOW-REF(CLOSE,1))
        TR = np.maximum(np.maximum(c1, c2), c3)
        TR = pd.DataFrame(index=c1.index, columns=c1.columns, data=TR)
        
        # df['TR'] = df[['c1', 'c2', 'c3']].max(
        #     axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        _ATR = TR.rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
        middle = close_winsorized.rolling(n, min_periods=1).mean()  # MIDDLE=MA(CLOSE,N)
        atr = _ATR / (middle + eps) # 对 atr 做归一化

        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter(atr, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)

    def alpha_cta_atr_keltner_channel_longOnly(self, n=14, cta_signal=0):

        eps = np.finfo(float).eps

        """
        N=20
        TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        ATR=MA(TR,N)
        MIDDLE=MA(CLOSE,N)
        """
        c1 = self.high - self.low  # HIGH-LOW
        c2 = abs(self.high - self.close.shift(1))  # ABS(HIGH-REF(CLOSE,1)
        c3 = abs(self.low - self.close.shift(1))  # ABS(LOW-REF(CLOSE,1))
        TR = np.maximum(np.maximum(c1, c2), c3)
        TR = pd.DataFrame(index=c1.index, columns=c1.columns, data=TR)
        
        # df['TR'] = df[['c1', 'c2', 'c3']].max(
        #     axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
        _ATR = TR.rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
        middle = self.close.rolling(n, min_periods=1).mean()  # MIDDLE=MA(CLOSE,N)
        atr = _ATR / (middle + eps)

        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter_longOnly(atr, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        

    def alpha_cta_bbw_dc_tunnel(self, n=14, cta_signal=0):

        median = self.close.rolling(n, min_periods=1).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        upper = median + std * 2
        lower = median - std * 2
        bbw = (upper - lower) / median
      
        signal, indicator, upper, median, position = self.helper_dc_tunnel_formatter(bbw, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
    
    def alpha_cta_bbw_dc_tunnel_winsorized(self, n=14, cta_signal=0):

        close_winsorized = self.data_sheet_winsorizing(self.close)
        median = close_winsorized.rolling(n, min_periods=1).mean()
        std = close_winsorized.rolling(n, min_periods=1).std(ddof=0)
        upper = median + std * 2
        lower = median - std * 2
        bbw = (upper - lower) / median
      
        signal, indicator, upper, median, position = self.helper_dc_tunnel_formatter(bbw, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        
    def alpha_cta_bbw_keltner_channel(self, n=14, cta_signal=0):

        median = self.close.rolling(n, min_periods=1).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        upper = median + std * 2
        lower = median - std * 2
        bbw = (upper - lower) / median

        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter(bbw, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        
    def alpha_cta_bbw_keltner_channel_longOnly(self, n=14, cta_signal=0):

        median = self.close.rolling(n, min_periods=1).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        upper = median + std * 2
        lower = median - std * 2
        bbw = (upper - lower) / median

        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter_longOnly(bbw, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)


    def alpha_cta_bias_bolling(self, n=14, cta_signal=0):

        bias = (self.close / self.close.rolling(n, min_periods=1).mean() - 1)
        
        signal, median, indicator, upper, lower = self.helper_bolling_formatter(bias, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)

    def alpha_cta_bias_bolling_longOnly(self, n=14, cta_signal=0):

        bias = (self.close / self.close.rolling(n, min_periods=1).mean() - 1)
        
        signal, median, indicator, upper, lower = self.helper_bolling_formatter_longOnly(bias, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)

    def alpha_cta_bias_dc_tunnel(self, n=14, cta_signal=0):

        bias = (self.close / self.close.rolling(n, min_periods=1).mean() - 1)
        
        signal, indicator, upper, median, position = self.helper_dc_tunnel_formatter(bias, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        
    def alpha_cta_bias_keltner_channel(self, n=14, cta_signal=0):

        bias = (self.close / self.close.rolling(n, min_periods=1).mean() - 1)
        
        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter(bias, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)


    def alpha_cta_bolling_width_bolling(self, n=150, cta_signal=0):
       
        eps = np.finfo(float).eps

        median = self.close.rolling(window=n).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        z_score = abs(self.close - median) / std
        m = z_score.rolling(window=n).mean()
        # upper = median + std * m
        # lower = median - std * m
        bolling_width = std * m * 2 / (median + eps)

        indicator = bolling_width
        signal, median, indicator, upper, lower = self.helper_bolling_formatter(indicator, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
                
    def alpha_cta_bolling_width_bolling_longOnly(self, n=14, cta_signal=0):
       
        eps = np.finfo(float).eps

        median = self.close.rolling(window=n).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        z_score = abs(self.close - median) / std
        m = z_score.rolling(window=n).mean()
        # upper = median + std * m
        # lower = median - std * m
        bolling_width = std * m * 2 / (median + eps)

        indicator = bolling_width
        signal, median, indicator, upper, lower = self.helper_bolling_formatter_longOnly(indicator, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        

    def alpha_cta_bolling_width_dc_tunnel(self, n=14, cta_signal=0):
       
        eps = np.finfo(float).eps

        median = self.close.rolling(window=n).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        z_score = abs(self.close - median) / std
        m = z_score.rolling(window=n).mean()
        # upper = median + std * m
        # lower = median - std * m
        bolling_width = std * m * 2 / (median + eps)

        indicator = bolling_width
        signal, indicator, upper, median, position = self.helper_dc_tunnel_formatter(indicator, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        
    def alpha_cta_bolling_width_keltner_channel(self, n=14, cta_signal=0):
       
        eps = np.finfo(float).eps

        median = self.close.rolling(window=n).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        z_score = abs(self.close - median) / std
        m = z_score.rolling(window=n).mean()
        # upper = median + std * m
        # lower = median - std * m
        bolling_width = std * m * 2 / (median + eps)

        indicator = bolling_width
        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter(indicator, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        

    def alpha_cta_mtm_bolling(self, n=14, cta_signal=0):

        mtm = (self.close / self.close.shift(n) - 1) * 100

        indicator = mtm
        signal, median, indicator, upper, lower = self.helper_bolling_formatter(indicator, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        
    def alpha_cta_mtm_bolling_longOnly(self, n=14, cta_signal=0):

        mtm = (self.close / self.close.shift(n) - 1) * 100

        indicator = mtm
        signal, median, indicator, upper, lower = self.helper_bolling_formatter_longOnly(indicator, n)
        
        if cta_signal:
            return signal
        else:
            position = signal.fillna(method='ffill').fillna(0)
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        

    def alpha_cta_mtm_dc_tunnel(self, n=14, cta_signal=0):

        mtm = (self.close / self.close.shift(n) - 1) * 100

        indicator = mtm
        signal, indicator, upper, median, position = self.helper_dc_tunnel_formatter(indicator, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        

    def alpha_cta_mtm_keltner_channel(self, n=14, cta_signal=0):

        mtm = (self.close / self.close.shift(n) - 1) * 100

        indicator = mtm
        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter(indicator, n)
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
        
    def alpha_cta_mtm_keltner_channel_longOnly(self, n=14, cta_signal=0):

        mtm = (self.close / self.close.shift(n) - 1) * 100

        indicator = mtm
        signal, indicator, median, upper, position = self.helper_keltner_channel_formatter_longOnly(indicator, n)
        
        
        if cta_signal:
            return signal
        else:
            return (indicator - median) / (upper - median) + position + signal.fillna(0)
    

    # def alpha_cta_ema(self, n=14, cta_signal=0):

    #     ema = self.close.ewm(n, adjust=False).mean()

    #     condition1 = self.close > ema
    #     condition2 = self.close.shift(1) <= ema.shift(1)
    #     signal = pd.DataFrame(index=condition1.index, columns=condition1.columns)
    #     # df.loc[condition1 & condition2, 'signal'] = 1
    #     signal[condition1 & condition2] = 1

    #     condition1 = self.close < ema
    #     condition2 = self.close.shift(1) >= ema.shift(1)
    #     df.loc[condition1 & condition2, 'signal'] = -1

    #     df[factor_name] = df['signal']
    #     # 从前往后填充开仓信号，1,0，或者-1. ffile不会带来未来函数
    #     df[factor_name] = df[factor_name].fillna(method='ffill', )

    #     return df

    # def alpha_cta_mike_stop_with_bias(self, n=14, n2=28, cta_signal=0):

    #     typ = (self.close + self.high + self.low) / 3
    #     dhh = self.high.rolling(n, min_periods=1).max()
    #     ll = self.low.rolling(n, min_periods=1).min()

    #     ma = self.close.rolling(window=n2, min_periods=1).mean()

    #     sr = dhh * 2 - ll
    #     mr = typ + dhh - ll
    #     wr = typ * 2 - ll

    #     ws = typ * 2 - dhh
    #     ms = typ - (dhh - ll)
    #     ss = ll * 2 - dhh

    #     median = self.helper_calculate_dema(self.close, n)

    #     cond1 = (self.close < ws.shift(1)) & (self.close >
    #                                                 ms.shift(1))
    #     cond2 = self.close > sr.shift(1)

    #     signal_long = pd.DataFrame(index=cond1.index, columns=cond1.columns)

    #     # df.loc[cond1 | cond2, 'signal_long'] = 1
    #     signal_long[cond1 | cond2] = 1

    #     condition_sell = (self.close < median) & (self.close.shift() >= median.shift())  # k线下穿中轨
    #     # df.loc[condition_sell, 'signal_long'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓
    #     signal_long[condition_sell] = 0


    #     cond3 = (self.close > wr.shift(1)) & (self.close < mr.shift(1))
    #     cond4 = self.close < ss.shift(1)

    #     # df.loc[cond3 | cond4, 'signal_short'] = -1
    #     signal_short = pd.DataFrame(index=cond3.index, columns=cond3.columns)
    #     signal_short[cond3 | cond4] = -1

    #     condition_cover = (self.close > median) & (self.close.shift() <= median.shift())  # K线上穿中轨
    #     # df.loc[condition_cover, 'signal_short'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓
    #     signal_short[condition_cover] = 0

    #     position_long = signal_long.fillna(method='ffill').fillna(0)
    #     position_short = signal_short.fillna(method='ffill').fillna(0)

    #     position = position_long + position_short

    #     signal = position[position != position.shift(1)]


    #     if cta_signal:
    #         return signal
    #     else:
    #         return 0

    def helper_calculate_dema(self, close, n=14):
        # 计算一次指数移动平均线（EMA）
        ema1 = close.ewm(span=n, adjust=False).mean()
        # 计算二次指数移动平均线（EMA）
        ema2 = ema1.ewm(span=n, adjust=False).mean()
        # 计算DEMA
        dema = 2 * ema1 - ema2
        return dema


    def alpha_cta_signal_simple_turtle(self, n=20, cta_signal=0):
        """
        今天收盘价突破过去20天中的收盘价和开盘价中的最高价，做多。今天收盘价突破过去10天中的收盘价的最低价，平仓。
        今天收盘价突破过去20天中的收盘价和开盘价中的最低价，做空。今天收盘价突破过去10天中的收盘价的最高价，平仓。
        将过去n期的signal值相加，出现平仓信号的因子值变为NaN。

        Parameter:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。

        """

        # 计算信号
        open_close_high = self.open.combine(self.close, func=np.maximum)
        open_close_low = self.open.combine(self.close, func=np.minimum)

        n_high = open_close_high.rolling(n, min_periods=1).max()
        n_low = open_close_low.rolling(n, min_periods=1).min()

        median = self.close.ewm(span=n, adjust=False).mean()

        # 多空信号部分
        signal = np.full_like(self.close, np.nan)   # 初始化信号为 NaN
        signal = np.where(self.close > n_high.shift(1), 1, signal)
        signal = np.where(self.close < n_low.shift(1), -1, signal)
        # 平仓信号（K线下穿或上穿中轨）
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)
        
        
        # 将信号转换为DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            # 返回 CTA 信号
            return signal

        else:
            # 中性因子信号处理
            factor_value = signal.copy()

            factor_value[signal != 0] = signal.rolling(n).sum()  # 对于 signal 不为 0 的，计算 rolling(n).sum()
            factor_value[signal == 0] = np.nan  # 对于 signal 为 0 的，设为 NaN

            # 返回 factor_value
            return factor_value



# def record_alpha_times(df):
#     stock = Alphas(df)
#     alpha_methods = [method for method in dir(stock) if method.startswith('alpha')]
#     times = {}
#     error = []

#     for method in alpha_methods:
#         print(f'evaluating {method}')
#         try:
#             start_time = time.time()
#             getattr(stock, method)()
#             end_time = time.time()
#             times[method] = end_time - start_time
#         except:
#             print(f'error in {method}')
#             error.append(method)
#     return times, error

    def alpha_cta_bollinger_count(self, n=10):

        """
        根据布林带策略生成交易信号。

        Parameter:
        n (int): 用于计算布林带的周期。

        """
    
        mean = self.close.rolling(n).mean()
        std = self.close.rolling(n).std(ddof=0)
        upper = mean + 2 * std
        lower = mean - 2 * std
        count = 0
        count = np.where(self.close > upper, 1, count)
        count = np.where(self.close < lower, -1, count)
        count = pd.DataFrame(count, index=self.close.index, columns=self.close.columns)
        return count.rolling(n).sum()



    
    def alpha_cta_simple_turtle_filtered_value(self, n=20, m=10, cta_signal=0):
        """
        计算因子：根据修改的海龟交易策略生成因子值。
        因子含义：通过价格相对于过去n天的最高价和最低价的变化，判断动量强弱。
        如果满足平仓条件，因子值为NaN。

        Parameters:
        n (int): 用于计算过去n天的最高价和最低价（默认20天）。
        m (int): 用于计算中轨的均线周期（默认10天）。
        """
            
        # 计算开盘价和收盘价中的最高价和最低价
        open_close_high = pd.DataFrame(np.maximum(self.open, self.close))
        open_close_low = pd.DataFrame(np.minimum(self.open, self.close))
            
        # 计算过去n天内的最高价和最低价
        n_high = open_close_high.rolling(window=n, min_periods=1).max()
        n_low = open_close_low.rolling(window=n, min_periods=1).min()
            
        # 计算收盘价的指数平滑均线，用于平仓信号
        median = self.close.ewm(span=m, adjust=False).mean()
            
        # 初始化因子列为NaN，确保未触发信号时为NaN
        factor = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
            
        # 计算平仓信号：当收盘价上下穿中轨时，设置因子值为NaN
        condition_close_long = (self.close < median) & (self.close.shift(1) >= median.shift(1))
        condition_close_short = (self.close > median) & (self.close.shift(1) <= median.shift(1))

        if cta_signal == 1:
            # 平仓信号为0值
            factor[condition_close_long | condition_close_short] = 0
            condition_open_long = (self.close > n_high) & (self.close.shift(1) <= n_high.shift(1))
            condition_open_short = (self.close < n_low) & (self.close.shift(1) >= n_low.shift(1))
            factor[condition_open_long] = 1
            factor[condition_open_short] = -1

            return factor

        if cta_signal == 0:
            
            factor[condition_close_long | condition_close_short] = np.nan  # 平仓信号因子值为NaN
                
            # 对于非平仓的时刻，计算新的因子值
            condition_non_close = ~(condition_close_long | condition_close_short)
                
            factor[condition_non_close] = (
                (self.close - n_high.shift(1)) / n_high.shift(1) +
                (self.close - n_low.shift(1)) / n_low.shift(1)
                )
                
            return factor    



    def alpha_cta_simple_turtle_wma_filtered_value(self, n=20, m=10, cta_signal=0):
        """
        计算因子：基于海龟交易策略，生成做多、做空和平仓信号的因子值，使用加权移动平均（WMA）作为中轨。
        因子含义：通过价格相对于过去n天的最高价和最低价的变化来判断动量强弱。
        如果满足平仓条件，因子值为NaN。

        Parameters:
        n (int): 用于计算过去n天的最高价和最低价（默认20天）。
        m (int): 用于计算中轨的均线周期（默认10天）。
        
        Returns:
        pd.DataFrame: 返回包含因子值的DataFrame，其中平仓时为NaN，其余为计算后的因子值。
        """
        
        # 计算开盘价和收盘价中的最高价和最低价
        open_close_high = pd.DataFrame(np.maximum(self.open, self.close))
        open_close_low = pd.DataFrame(np.minimum(self.open, self.close))
        
        # 计算过去n天内的最高价和最低价
        n_high = open_close_high.rolling(window=n, min_periods=1).max()
        n_low = open_close_low.rolling(window=n, min_periods=1).min()

        # 计算加权移动平均线（中轨），使用n天的加权
        """用下式替代
        weights = np.arange(1, n + 1)
        median = self.close.rolling(window=n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        """
        median = self.my_wma(self.close, n)

        # 初始化因子值为NaN，确保未触发信号时为NaN
        factor = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        
        # 计算平仓信号：当收盘价上下穿中轨时，设置因子值为NaN
        condition_close_long = (self.close < median) & (self.close.shift(1) >= median.shift(1))
        condition_close_short = (self.close > median) & (self.close.shift(1) <= median.shift(1))
        
        if cta_signal == 0:
        # 设置平仓信号为NaN
            factor[condition_close_long | condition_close_short] = np.nan
            
            # 对于非平仓的时刻，计算新的因子值
            condition_non_close = ~(condition_close_long | condition_close_short)
            
            # 计算公式：因子值为价格相对过去n天最高价和最低价的偏离度
            factor[condition_non_close] = (
                (self.close - n_high.shift(1)) / n_high.shift(1) +
                (self.close - n_low.shift(1)) / n_low.shift(1)
            )
            
            return factor
        if cta_signal == 1:
            factor[condition_close_long | condition_close_short] = 0

            condition_open_long = (self.close > n_high) & (self.close.shift(1) <= n_high.shift(1))
            condition_open_short = (self.close < n_low) & (self.close.shift(1) >= n_low.shift(1))

            factor[condition_open_long] = 1
            factor[condition_open_short] = -1

            return factor
    
    
    
    def alpha_cta_simple_turtle_reverse_filtered_value(self, n=20, m=10, cta_signal=0):
        """
        计算因子：根据修改的反向海龟交易策略生成因子值。
        因子含义：通过价格相对于过去n天的最高价和最低价的变化，判断反转强弱。
        如果满足平仓条件，因子值为NaN。

        Parameters:
        n (int): 用于计算过去n天的最高价和最低价（默认20天）。
        m (int): 用于计算中轨的均线周期（默认10天）。
        """
            
        # 计算开盘价和收盘价中的最高价和最低价
        open_close_high = pd.DataFrame(np.maximum(self.open, self.close))
        open_close_low = pd.DataFrame(np.minimum(self.open, self.close))
            
        # 计算过去n天内的最高价和最低价
        n_high = open_close_high.rolling(window=n, min_periods=1).max()
        n_low = open_close_low.rolling(window=n, min_periods=1).min()
            
        # 计算收盘价的指数平滑均线，用于平仓信号
        median = self.close.ewm(span=m, adjust=False).mean()
            
        # 初始化因子列为NaN，确保未触发信号时为NaN
        factor = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        
        
        # 计算平仓信号：当收盘价上下穿中轨时，设置因子值为NaN
        condition_close_long = (self.close < median) & (self.close.shift(1) >= median.shift(1))
        condition_close_short = (self.close > median) & (self.close.shift(1) <= median.shift(1))

        if cta_signal == 0:  
            factor[condition_close_long | condition_close_short] = np.nan  # 平仓信号因子值为NaN
                
            # 对于非平仓的时刻，计算新的因子值
            condition_non_close = ~(condition_close_long | condition_close_short)
                
            factor[condition_non_close] = -(
                (self.close - n_high.shift(1)) / n_high.shift(1) +
                (self.close - n_low.shift(1)) / n_low.shift(1)
                )
                
            return factor    
        
        if cta_signal == 1:
            factor[condition_close_long | condition_close_short] = 0

            condition_open_long = (self.close < n_low) & (self.close.shift(1) >= n_low.shift(1))
            condition_open_short = (self.close > n_high) & (self.close.shift(1) <= n_high.shift(1))

            factor[condition_open_long] = 1
            factor[condition_open_short] = -1

            return factor


    def alpha_cta_simple_turtle_dema_filtered_value(self, n=20, m=10, cta_signal=0):
        """
        计算因子：基于海龟交易策略，生成做多、做空和平仓信号的因子值，中轨使用DEMA计算。
        因子含义：通过价格相对于过去n天的最高价和最低价的变化来判断动量强弱。
        如果满足平仓条件，因子值为NaN。

        Parameters:
        n (int): 用于计算过去n天的最高价和最低价（默认20天）。
        m (int): 用于计算中轨的DEMA周期（默认10天）。
        
        Returns:
        pd.DataFrame: 返回包含因子值的DataFrame，其中平仓时为NaN，其余为计算后的因子值。
        """
        
        # 计算开盘价和收盘价中的最高价和最低价
        open_close_high = pd.DataFrame(np.maximum(self.open, self.close), index=self.close.index, columns=self.close.columns)
        open_close_low = pd.DataFrame(np.minimum(self.open, self.close), index=self.close.index, columns=self.close.columns)
        
        # 计算过去n天内的最高价和最低价
        n_high = open_close_high.rolling(window=n, min_periods=1).max()
        n_low = open_close_low.rolling(window=n, min_periods=1).min()

        # 使用DEMA作为中轨的计算方式
        median = self.calculate_dema(self.close, m)

        # 初始化因子值为NaN，确保未触发信号时为NaN
        factor = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        
        # 计算平仓信号：当收盘价上下穿中轨时，设置因子值为NaN
        condition_close_long = (self.close < median) & (self.close.shift(1) >= median.shift(1))
        condition_close_short = (self.close > median) & (self.close.shift(1) <= median.shift(1))
        
        if cta_signal == 0:
            # 设置平仓信号为NaN
            factor[condition_close_long | condition_close_short] = np.nan
            
            # 对于非平仓的时刻，计算新的因子值
            condition_non_close = ~(condition_close_long | condition_close_short)
            
            # 计算公式：因子值为价格相对过去n天最高价和最低价的偏离度
            factor[condition_non_close] = (
                (self.close - n_high.shift(1)) / n_high.shift(1) +
                (self.close - n_low.shift(1)) / n_low.shift(1)
            )
        
            return factor
        if cta_signal == 1:
            factor[condition_close_long | condition_close_short] = np.nan
            condition_open_long = (self.close > n_high) & (self.close.shift(1) <= n_high.shift(1))
            condition_open_short = (self.close > n_low) & (self.close.shift(1) <= n_low.shift(1))
            factor[condition_open_long] = 1
            factor[condition_open_short] = -1

            return factor
    
    
    def calculate_dema(self, close, n):
        """
        计算双重指数移动平均线（DEMA）

        Parameters:
        close (pd.DataFrame): 收盘价数据
        n (int): 用于计算EMA的周期长度

        Returns:
        pd.DataFrame: 返回计算后的DEMA值
        """
        # 计算一次指数移动平均线（EMA）
        ema1 = close.ewm(span=n, adjust=False).mean()
        # 计算二次指数移动平均线（EMA）
        ema2 = ema1.ewm(span=n, adjust=False).mean()
        # 计算DEMA
        dema = 2 * ema1 - ema2
        return dema
    
    
    
    def alpha_cta_signal_simple_turtle_unfiltered(self, n=20):
        """
        今天收盘价突破过去20天中的收盘价和开盘价中的最高价，做多。今天收盘价突破过去10天中的收盘价的最低价，平仓。
        今天收盘价突破过去20天中的收盘价和开盘价中的的最低价，做空。今天收盘价突破过去10天中的收盘价的最高价，平仓。
        将过去n期的signal值相加。

        Parameter:
        n (int): 用于计算信号累加的周期。

        """
        
        # 计算信号
        open_close_high = self.open.combine(self.close, func=np.maximum)
        open_close_low = self.open.combine(self.close, func=np.minimum)

        n_high = open_close_high.rolling(n, min_periods=1).max()
        n_low = open_close_low.rolling(n, min_periods=1).min()

        # 信号计数
        count = np.zeros_like(self.close)
        count = np.where(self.close > n_high.shift(1), 1, count)
        count = np.where(self.close < n_low.shift(1), -1, count)

        # 将计数结果转换为DataFrame
        count = pd.DataFrame(count, index=self.close.index, columns=self.close.columns)

        # 返回过去n期的信号累加
        return count.rolling(n, min_periods=1).sum()
    
   
    
    def alpha_cta_signal_simple_turtle(self, n=20, cta_signal=0):
        """
        今天收盘价突破过去20天中的收盘价和开盘价中的最高价，做多。今天收盘价突破过去10天中的收盘价的最低价，平仓。
        今天收盘价突破过去20天中的收盘价和开盘价中的最低价，做空。今天收盘价突破过去10天中的收盘价的最高价，平仓。
        将过去n期的signal值相加，出现平仓信号的因子值变为NaN。

        Parameter:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 计算信号
        open_close_high = self.open.combine(self.close, func=np.maximum)
        open_close_low = self.open.combine(self.close, func=np.minimum)

        n_high = open_close_high.rolling(n, min_periods=1).max()
        n_low = open_close_low.rolling(n, min_periods=1).min()

        median = self.close.ewm(span=n, adjust=False).mean()

        # 多空信号部分
        signal = np.full_like(self.close, np.nan)   # 初始化信号为 NaN
        signal = np.where(self.close > n_high.shift(1), 1, signal)
        signal = np.where(self.close < n_low.shift(1), -1, signal)
        # 平仓信号（K线下穿或上穿中轨）
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)
        
        
        # 将信号转换为DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            # 返回 CTA 信号
            return signal

        else:
            # 中性因子信号处理
            factor_value = signal.copy()

            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()  # 对于 signal 不为 0 的，计算 rolling(n).sum()
            factor_value[signal == 0] = np.nan  # 对于 signal 为 0 的，设为 NaN

            # 返回 factor_value
            return factor_value
    
    
    
    def alpha_cta_signal_simple_turtle_wma(self, n=20, cta_signal=0):
        """
        今天收盘价突破过去20天中的收盘价和开盘价中的最高价，做多。今天收盘价突破过去10天中的收盘价的最低价，平仓。
        今天收盘价突破过去20天中的收盘价和开盘价中的的最低价，做空。今天收盘价突破过去10天中的收盘价的最高价，平仓。
        中线用 wma。
        将过去n期的signal值相加，出现平仓信号的因子值变为NaN。

        Parameter:
        n (int): 用于计算信号累加的周期。

        """
    
        # 计算信号
        open_close_high = self.open.combine(self.close, func=np.maximum)
        open_close_low = self.open.combine(self.close, func=np.minimum)

        n_high = open_close_high.rolling(n, min_periods=1).max()
        n_low = open_close_low.rolling(n, min_periods=1).min()

        # 计算加权移动平均线（中轨），使用n天的加权，遇到平仓信号时将信号值设为NaN
        weights = np.arange(1, n + 1)
        median = self.close.rolling(window=n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

        # 多空信号部分
        signal = np.full_like(self.close, np.nan)   # 初始化信号为 NaN
        signal = np.where(self.close > n_high.shift(1), 1, signal)
        signal = np.where(self.close < n_low.shift(1), -1, signal)
        # 平仓信号（K线下穿或上穿中轨）
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)
        
        
        # 将信号转换为DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            # 返回 CTA 信号
            return signal

        else:
            # 中性因子信号处理
            factor_value = signal.copy()

            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()  # 对于 signal 不为 0 的，计算 rolling(n).sum()
            factor_value[signal == 0] = np.nan  # 对于 signal 为 0 的，设为 NaN

            # 返回 factor_value
            return factor_value
    
    
    
    def cta_signal_simple_turtle_reverse_unfilterd(self, n=20):
        """
        反向海龟策略
        今天收盘价突破过去20天中的收盘价和开盘价中的最高价，做空。
        今天收盘价突破过去20天中的收盘价和开盘价中的的最低价，做多。
        将过去n期的signal值相加。

        Parameter:
        n (int): 用于计算信号累加的周期。

        """
        
        # 计算信号
        open_close_high = self.open.combine(self.close, func=np.maximum)
        open_close_low = self.open.combine(self.close, func=np.minimum)

        n_high = open_close_high.rolling(n, min_periods=1).max()
        n_low = open_close_low.rolling(n, min_periods=1).min()

        # 信号计数
        count = np.zeros_like(self.close)
        count = np.where(self.close > n_high.shift(1), -1, count)
        count = np.where(self.close < n_low.shift(1), 1, count)

        # 将计数结果转换为DataFrame
        count = pd.DataFrame(count, index=self.close.index, columns=self.close.columns)

        # 返回过去n期的信号累加
        return count.rolling(n, min_periods=1).sum()
    
   
    
    def alpha_cta_signal_simple_turtle_reverse(self, n=20, cta_signal=0):
        """
        反向
        今天收盘价突破过去20天中的收盘价和开盘价中的最低价，做多。今天收盘价突破过去10天中的收盘价的最高价，平仓。
        今天收盘价突破过去20天中的收盘价和开盘价中的的最高价，做空。今天收盘价突破过去10天中的收盘价的最低价，平仓
        将过去n期的signal值相加，出现平仓信号的因子值变为NaN。

        Parameter:
        n (int): 用于计算信号累加的周期。
        """
        
        # 计算信号
        open_close_high = self.open.combine(self.close, func=np.maximum)
        open_close_low = self.open.combine(self.close, func=np.minimum)

        n_high = open_close_high.rolling(n, min_periods=1).max()
        n_low = open_close_low.rolling(n, min_periods=1).min()

        median = self.close.ewm(span=n, adjust=False).mean()

        # 多空信号部分
        signal = np.full_like(self.close, np.nan)   # 初始化信号为 NaN
        signal = np.where(self.close > n_high.shift(1), -1, signal)
        signal = np.where(self.close < n_low.shift(1), 1, signal)
        # 平仓信号（K线下穿或上穿中轨）
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)
        
        
        # 将信号转换为DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            # 返回 CTA 信号
            return signal

        else:
            # 中性因子信号处理
            factor_value = signal.copy()

            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()  # 对于 signal 不为 0 的，计算 rolling(n).sum()
            factor_value[signal == 0] = np.nan  # 对于 signal 为 0 的，设为 NaN

            # 返回 factor_value
            return factor_value
    
    
    def alpha_cta_signal_simple_turtle_dema(self, n=20, cta_signal=0):
        """
        今天收盘价突破过去20天中的收盘价和开盘价中的最高价，做多。今天收盘价突破过去10天中的收盘价的最低价，平仓。
        今天收盘价突破过去20天中的收盘价和开盘价中的的最低价，做空。今天收盘价突破过去10天中的收盘价的最高价，平仓
        中轨用dema。
        将过去n期的signal值相加，出现平仓信号的因子值变为NaN。

        Parameter:
        n (int): 用于计算信号累加的周期。
        """
        
        # 计算信号
        open_close_high = self.open.combine(self.close, func=np.maximum)
        open_close_low = self.open.combine(self.close, func=np.minimum)

        n_high = open_close_high.rolling(n, min_periods=1).max()
        n_low = open_close_low.rolling(n, min_periods=1).min()

        # 使用DEMA作为中轨的计算方式
        median = self.calculate_dema(self.close, n)
        
        # 多空信号部分
        signal = np.full_like(self.close, np.nan)   # 初始化信号为 NaN
        signal = np.where(self.close > n_high.shift(1), 1, signal)
        signal = np.where(self.close < n_low.shift(1), -1, signal)
        # 平仓信号（K线下穿或上穿中轨）
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)
        
        
        # 将信号转换为DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            # 返回 CTA 信号
            return signal

        else:
            # 中性因子信号处理
            factor_value = signal.copy()

            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()  # 对于 signal 不为 0 的，计算 rolling(n).sum()
            factor_value[signal == 0] = np.nan  # 对于 signal 为 0 的，设为 NaN

            # 返回 factor_value
            return factor_value
        

    def alpha_cta_signal_mtmbbw_bolling(self, n=20, cta_signal=0):
        """
        基于布林带和动量趋势的因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算布林带指标
        diff_c = self.close / self.close.shift(n)
        median = self.close.rolling(n, min_periods=1).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        upper = median + std * (diff_c + diff_c ** (-1))
        lower = median - std * (diff_c + diff_c ** (-1))
        mouth = upper - lower
        mouth_m = mouth.rolling(n).mean()

        # 做多信号：当前K线的收盘价 > 上轨
        signal = np.where(self.close > upper, 1, signal)

        # 做空信号：当前K线的收盘价 < 下轨
        signal = np.where(self.close < lower, -1, signal)

        # 平仓信号条件：K线下穿或上穿中轨，结合布林带收缩情况
        close_above_median = self.close > median
        close_below_median = self.close < median
        mouth_narrowing = mouth < mouth_m

        # 下穿中轨：布林带收缩且收盘价跌破中轨
        signal = np.where(mouth_narrowing & close_below_median & (self.close.shift(1) >= median.shift(1)), 0, signal)

        # 上穿中轨：布林带收缩且收盘价突破中轨
        signal = np.where(mouth_narrowing & close_above_median & (self.close.shift(1) <= median.shift(1)), 0, signal)

        # 将信号转换为 DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            return signal

        else:
            # 生成中性因子信号
            factor_value = signal.copy()
            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()
            factor_value[signal == 0] = np.nan
            return factor_value

    
    def alpha_cta_signal_mtmbbw_bolling_unfiltered(self, n=20):
        
        signal = np.zeros_like(self.close)
        
        # 计算布林带指标
        diff_c = self.close / self.close.shift(n)
        median = self.close.rolling(n, min_periods=1).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        upper = median + std * (diff_c + diff_c ** (-1))
        lower = median - std * (diff_c + diff_c ** (-1))
        # mouth = upper - lower
        # mouth_m = mouth.rolling(n).mean()

        # 做多信号：当前K线的收盘价 > 上轨
        signal = np.where(self.close > upper, 1, signal)

        # 做空信号：当前K线的收盘价 < 下轨
        signal = np.where(self.close < lower, -1, signal)

        # # 平仓信号条件：K线下穿或上穿中轨，结合布林带收缩情况
        # close_above_median = self.close > median
        # close_below_median = self.close < median
        # mouth_narrowing = mouth < mouth_m

        # # 下穿中轨：布林带收缩且收盘价跌破中轨
        # signal = np.where(mouth_narrowing & close_below_median & (self.close.shift(1) >= median.shift(1)), 0, signal)

        # # 上穿中轨：布林带收缩且收盘价突破中轨
        # signal = np.where(mouth_narrowing & close_above_median & (self.close.shift(1) <= median.shift(1)), 0, signal)

        # 将信号转换为 DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        # 返回过去n期的信号累加
        factor_value = signal.rolling(n, min_periods=1).sum()
        return factor_value
    
    
  
    def alpha_cta_signal_mike(self, n=20, cta_signal=0):
        """
        基于MIKE指标的因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算MIKE指标
        typ = (self.close + self.high + self.low) / 3
        hh = self.high.rolling(n, min_periods=1).max()
        ll = self.low.rolling(n, min_periods=1).min()

        sr = hh * 2 - ll
        mr = typ + (hh - ll)
        wr = typ * 2 - ll

        ws = typ * 2 - hh
        ms = typ - (hh - ll)
        ss = ll * 2 - hh

        # 做多信号：收盘价在ws和ms之间，或者收盘价突破sr
        signal = np.where((self.close < ws.shift()) & (self.close > ms.shift()) | (self.close > sr.shift()), 1, signal)

        # 做空信号：收盘价在wr和mr之间，或者收盘价跌破ss
        signal = np.where((self.close > wr.shift()) & (self.close < mr.shift()) | (self.close < ss.shift()), -1, signal)

        # 平仓信号：收盘价下穿ms或上穿mr
        # 下穿中轨平仓：当前收盘价 < ms 且前一周期收盘价 >= ms.shift()
        signal = np.where((self.close < ms) & (self.close.shift() >= ms.shift()), 0, signal)
        # 上穿中轨平仓：当前收盘价 > mr 且前一周期收盘价 <= mr.shift()
        signal = np.where((self.close > mr) & (self.close.shift() <= mr.shift()), 0, signal)

        # 将信号转换为 DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            return signal

        else:
            # 生成中性因子信号
            factor_value = signal.copy()
            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()
            factor_value[signal == 0] = np.nan
            return factor_value  
    
    
    def alpha_cta_signal_highlow_bolling(self, n=20, cta_signal=0):
        """
        基于High-Low Bollinger Bands的因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算指标
        median = self.close.rolling(n, min_periods=1).mean()
        std = (self.high - self.low).rolling(n, min_periods=1).mean()
        z_score = abs(self.close - median) / std
        m = z_score.rolling(window=n, min_periods=1).mean()
        upper = median + std * m
        lower = median - std * m

        # 做多信号：收盘价突破上轨
        signal = np.where(self.close > upper, 1, signal)

        # 做空信号：收盘价跌破下轨
        signal = np.where(self.close < lower, -1, signal)

        # 平仓信号（K线下穿中轨或上穿中轨）
        # 下穿中轨平仓：当前收盘价 < 中轨 且前一周期收盘价 >= 中轨
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        # 上穿中轨平仓：当前收盘价 > 中轨 且前一周期收盘价 <= 中轨
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)

        # 将信号转换为 DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            return signal

        else:
            # 生成中性因子信号
            factor_value = signal.copy()
            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()
            factor_value[signal == 0] = np.nan
            return factor_value
    

    def alpha_cta_signal_highlow_bolling_wma(self, n=20, cta_signal=0):
        """
        基于High-Low Bollinger Bands的因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算指标
        weights = np.arange(1, n + 1)
        median = self.close.rolling(n, min_periods=1).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        std = (self.high - self.low).rolling(n, min_periods=1).mean()
        z_score = abs(self.close - median) / std
        m = z_score.rolling(window=n, min_periods=1).mean()
        upper = median + std * m
        lower = median - std * m

        # 做多信号：收盘价突破上轨
        signal = np.where(self.close > upper, 1, signal)

        # 做空信号：收盘价跌破下轨
        signal = np.where(self.close < lower, -1, signal)

        # 平仓信号（K线下穿中轨或上穿中轨）
        # 下穿中轨平仓：当前收盘价 < 中轨 且前一周期收盘价 >= 中轨
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        # 上穿中轨平仓：当前收盘价 > 中轨 且前一周期收盘价 <= 中轨
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)

        # 将信号转换为 DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            return signal

        else:
            # 生成中性因子信号
            factor_value = signal.copy()
            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()
            factor_value[signal == 0] = np.nan
            return factor_value



    def alpha_cta_signal_dual_thrust(self, n=20, cta_signal=0):
        """
        根据区间范围计算因子信号，使用最高价、最低价和开盘价与收盘价的对比。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算最高价、最低价和收盘价的区间
        hh = self.high.rolling(n, min_periods=1).max()
        lc = self.close.rolling(n, min_periods=1).min()
        hc = self.close.rolling(n, min_periods=1).max()
        ll = self.low.rolling(n, min_periods=1).min()

        # 根据条件选择 range
        range_value = np.where((hh - lc) > (hc - ll), hh - lc, hc - ll)
        range_value = pd.DataFrame(range_value, index=self.close.index, columns=self.close.columns)

        # 计算 upper_open 和 lower_open
        upper_open = 2 * abs(self.close - self.open.shift()) / range_value.rolling(n, min_periods=1).max()
        lower_open = 2 * abs(self.open.shift() - self.close) / range_value.rolling(n, min_periods=1).max()

        # 计算 upper 和 lower
        upper = self.open.shift() + upper_open * range_value
        lower = self.open.shift() - lower_open * range_value

        # 做多信号：收盘价高于 upper，且（upper.shift() - lower.shift()）的变化率大于 0.05
        signal = np.where((self.close > upper) & ((upper.shift() - lower.shift()) / upper.shift() > 0.05), 1, signal)

        # 做空信号：收盘价低于 lower，且（upper.shift() - lower.shift()）的变化率大于 0.05
        signal = np.where((self.close < lower) & ((upper.shift() - lower.shift()) / upper.shift() > 0.05), -1, signal)

        # 平仓信号：上下轨间距小于高低价差，或上下轨变化率小于 0.05
        close_condition = (upper.shift() - lower.shift()) < (self.high.shift() - self.low.shift())
        narrow_range_condition = ((upper.shift() - lower.shift()) / upper.shift()) < 0.05
        signal = np.where(close_condition | narrow_range_condition, 0, signal)

        # 将信号转换为 DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            return signal

        else:
            # 生成中性因子信号
            signal = signal.fillna(method='ffill').fillna(0)
            factor_value = signal.copy()
            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()
            # factor_value[signal == 0] = np.nan
            return factor_value



    def alpha_cta_signal_dc_tunnel(self, n=20, cta_signal=0):
        """
        基于唐奇安通道的因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算移动平均、最高价和最低价通道
        mean = self.close.rolling(n, min_periods=1).mean()
        max_val = self.close.rolling(n, min_periods=1).max().shift()
        min_val = self.close.rolling(n, min_periods=1).min().shift()

        # 做多信号：当前收盘价突破上轨（max）
        signal = np.where((self.close > max_val), 1, signal)

        # 平仓信号：当前收盘价跌破均线（mean）
        signal = np.where((self.close < mean) & (self.close.shift() >= mean.shift()), 0, signal)

        # 做空信号：当前收盘价跌破下轨（min）
        signal = np.where((self.close < min_val), -1, signal)

        # 平仓信号：当前收盘价突破均线（mean）
        signal = np.where((self.close > mean) & (self.close.shift() <= mean.shift()), 0, signal)

        # 将信号转换为 DataFrame
        signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            return signal

        else:
            # 生成中性因子信号
            factor_value = signal.copy()
            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()
            factor_value[signal == 0] = np.nan
            return factor_value



    def alpha_cta_signal_dc_flash_with_stop_lose(self, n=20, stop_loss_pct=10, cta_signal=0):
        """
        基于唐奇安通道、ATR 和止损机制的因子信号生成，含 flash stop win 逻辑。

        Parameters:
        n (int): 用于计算信号累加的周期。
        stop_loss_pct (float): 止损百分比参数。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算唐奇安通道、ATR 和动量指标
        median = self.close.rolling(n, min_periods=1).mean()
        upper = self.close.rolling(window=n).max().shift(1)
        lower = self.close.rolling(window=n).min().shift(1)
        mtm = self.close / self.close.shift(n) - 1

        tr = np.maximum(self.high - self.low, abs(self.high - self.close.shift(1)), abs(self.low - self.close.shift(1)))
        atr = tr.rolling(window=n, min_periods=1).mean()

        # 初始化flash stop win
        flash_stop_win = median.copy()

        # 做多信号：当收盘价上穿DC上轨，且动量大于0
        signal = np.where((self.close > upper) & (mtm > 0), 1, signal)

        # 做空信号：当收盘价下穿DC下轨，且动量小于0
        signal = np.where((self.close < lower) & (mtm < 0), -1, signal)

        # 平仓信号：当收盘价上穿或下穿均线
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)

        signal_all_ticker = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)
        
        signal_all_ticker_updated = pd.DataFrame()
        for ticker in signal_all_ticker.columns:
            signal = signal_all_ticker[ticker]
            # 止损和平仓逻辑处理
            info_dict = {
                'pre_signal': 0,
                'stop_loss_price': None,
                'holding_times': 0,
                'stop_win_price': 0,
                'stop_win_times': 0
            }

            ma_dict = {}
            holding_times_min = 10

            for i in range(len(self.close)):
                # 初始持仓信号：做多或做空
                if info_dict['pre_signal'] == 0:
                    if signal[i] == 1:  # 做多信号
                        info_dict['pre_signal'] = 1
                        info_dict['stop_loss_price'] = self.close.iloc[i][ticker] * (1 - stop_loss_pct / 100)
                    elif signal[i] == -1:  # 做空信号
                        info_dict['pre_signal'] = -1
                        info_dict['stop_loss_price'] = self.close.iloc[i][ticker] * (1 + stop_loss_pct / 100)

                # 处理持有做多的情况
                elif info_dict['pre_signal'] == 1:
                    holding_times = info_dict['holding_times']

                    # 更新止损价格和持有时间
                    if self.close.iloc[i][ticker] < info_dict['stop_loss_price']:
                        signal[i] = 0
                        info_dict = {'pre_signal': 0, 'stop_loss_price': None, 'holding_times': 0, 'stop_win_times': 0, 'stop_win_price': 0}

                    # 计算自适应MA（与 ATR 结合）
                    ma_temp = max(n - int(n / 50) * 10 * holding_times, holding_times_min)
                    if ma_temp not in ma_dict:
                        ma_dict[ma_temp] = self.close[ticker].rolling(ma_temp, min_periods=1).mean()
                    flash_stop_win.iloc[i][ticker] = ma_dict[ma_temp].iloc[i]

                    # 如果收盘价低于flash stop win，设置平仓信号
                    if self.close.iloc[i][ticker] < flash_stop_win.iloc[i][ticker]:
                        if self.close.iloc[i][ticker] > info_dict['stop_win_price'] or info_dict['stop_win_times'] == 0:
                            info_dict['stop_win_price'] = self.close.iloc[i][ticker]
                            info_dict['stop_win_times'] += 1
                            info_dict['holding_times'] = 0
                        else:
                            signal[i] = 0

                # 处理持有做空的情况
                elif info_dict['pre_signal'] == -1:
                    holding_times = info_dict['holding_times']

                    # 更新止损价格和持有时间
                    if self.close.iloc[i][ticker] > info_dict['stop_loss_price']:
                        signal[i] = 0
                        info_dict = {'pre_signal': 0, 'stop_loss_price': None, 'holding_times': 0, 'stop_win_times': 0, 'stop_win_price': 0}

                    # 计算自适应MA（与 ATR 结合）
                    ma_temp = max(n - int(n / 50) * 10 * holding_times, holding_times_min)
                    if ma_temp not in ma_dict:
                        ma_dict[ma_temp] = self.close[ticker].rolling(ma_temp, min_periods=1).mean()
                    flash_stop_win.iloc[i][ticker] = ma_dict[ma_temp].iloc[i]

                    # 如果收盘价高于flash stop win，设置平仓信号
                    if self.close.iloc[i][ticker] > flash_stop_win.iloc[i][ticker]:
                        if self.close.iloc[i][ticker] < info_dict['stop_win_price'] or info_dict['stop_win_times'] == 0:
                            info_dict['stop_win_price'] = self.close.iloc[i][ticker]
                            info_dict['stop_win_times'] += 1
                            info_dict['holding_times'] = 0
                        else:
                            signal[i] = 0
            signal_all_ticker_updated = pd.concat([signal_all_ticker_updated, signal])    

        # 将信号转换为 DataFrame
        # signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)
        signal = signal_all_ticker_updated

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            return signal

        else:
            # 生成中性因子信号
            factor_value = signal.copy()
            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()
            factor_value[signal == 0] = np.nan
            return factor_value




    def alpha_cta_signal_atrbolling_bias(self, n=20, cta_signal=0):
        """
        基于 ATR 和 Bollinger Bands 以及偏差的因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算 ATR、标准差、中轨和偏差
        tr = np.maximum(
            abs(self.high - self.low),
            abs(self.high - self.close.shift(1)),
            abs(self.low - self.close.shift(1))
        )
        atr = tr.rolling(n, min_periods=1).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        median = self.close.rolling(n, min_periods=1).mean()

        # 计算 ATR 和 Bollinger "J神指标"
        atr_Js = abs(self.close - median) / atr
        boll_Js = abs(self.close - median) / std

        m_atr = atr_Js.rolling(n, min_periods=1).max().shift(1)
        m_boll = boll_Js.rolling(n, min_periods=1).max().shift(1)

        upper_atr = median + m_atr * atr
        lower_atr = median - m_atr * atr
        upper_boll = median + m_boll * std
        lower_boll = median - m_boll * std

        upper = (upper_atr + upper_boll) / 2
        lower = (lower_atr + lower_boll) / 2

        # 计算偏差和最大偏差
        bias = self.close / median - 1
        bias_pct = abs(bias).rolling(n, min_periods=1).max().shift(1)

        # 生成做多、做空和平仓信号
        # 做多信号：收盘价突破上轨
        signal = np.where((self.close > upper) & (self.close.shift(1) <= upper.shift(1)), 1, signal)

        # 做空信号：收盘价跌破下轨
        signal = np.where((self.close < lower) & (self.close.shift(1) >= lower.shift(1)), -1, signal)

        # 平仓信号：收盘价上穿或下穿中轨
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)

        # 处理偏差条件的过滤逻辑
        temp_signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)
        temp_signal[(temp_signal == 1) & (bias > bias_pct)] = np.nan
        temp_signal[(temp_signal == -1) & (bias < -bias_pct)] = np.nan

        signal = temp_signal
        # # 去掉连续的重复信号
        # temp_signal.fillna(method='ffill', inplace=True)
        # signal = np.where(temp_signal != temp_signal.shift(1), temp_signal, signal)

        # # 将信号转换为 DataFrame
        # signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            return signal

        else:
            # 生成中性因子信号
            factor_value = signal.copy()
            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()
            factor_value[signal == 0] = np.nan
            return factor_value
        
    def my_wma(self, price, n):
        weights = np.arange(1, n + 1)
        weights = weights / weights.sum()
        def _price_dot_weight(price):
            return price.T.dot(weights)
        median = self.close.rolling(window=n).apply(lambda prices: _price_dot_weight(prices))
        return median


    def alpha_cta_signal_atrbolling_bias_wma(self, n=20, cta_signal=0):
        """
        基于 ATR、Bollinger Bands 以及 WMA (加权移动平均) 的因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算 ATR、标准差和 WMA 中轨
        tr = np.maximum(
            abs(self.high - self.low),
            abs(self.high - self.close.shift(1)),
            abs(self.low - self.close.shift(1))
        )
        atr = tr.rolling(n, min_periods=1).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)

        # 计算 WMA (加权移动平均)
        # weights = np.arange(1, n + 1)
        # median = self.close.rolling(window=n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        
        weights = np.arange(1, n + 1)
        weights = weights / weights.sum()
        def _price_dot_weight(price):
            return price.T.dot(weights)
        median = self.close.rolling(window=n).apply(lambda prices: _price_dot_weight(prices))

        # ATR 和 Bollinger Bands 的 "神指标" 计算
        atr_Js = abs(self.close - median) / atr
        boll_Js = abs(self.close - median) / std

        m_atr = atr_Js.rolling(window=n).max().shift(1)
        m_boll = boll_Js.rolling(window=n).max().shift(1)

        # 计算 ATR 和 Bollinger Bands 上轨和下轨
        upper_atr = median + m_atr * atr
        lower_atr = median - m_atr * atr
        upper_boll = median + m_boll * std
        lower_boll = median - m_boll * std

        # 平均上轨和下轨
        upper = (upper_atr + upper_boll) / 2
        lower = (lower_atr + lower_boll) / 2

        # 计算偏差和最大偏差
        bias = self.close / median - 1
        bias_pct = abs(bias).rolling(n, min_periods=1).max().shift(1)

        # 生成做多、做空和平仓信号
        signal = np.where((self.close > upper) & (self.close.shift(1) <= upper.shift(1)), 1, signal)
        signal = np.where((self.close < lower) & (self.close.shift(1) >= lower.shift(1)), -1, signal)
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)

        
        # 根据偏差条件过滤信号
        temp_signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)
    
        temp_signal[(temp_signal == 1) & (bias > bias_pct)] = np.nan
        temp_signal[(temp_signal == -1) & (bias < -bias_pct)] = np.nan

        # 去掉连续的重复信号
        # temp_signal.fillna(method='ffill', inplace=True)
        # signal = np.where(temp_signal != temp_signal.shift(1), temp_signal, signal)

        # 将信号转换为 DataFrame
        # signal = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)
        signal = temp_signal

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal = signal.fillna(method='ffill').fillna(0)
            signal = signal[signal != signal.shift()]
            return signal

        else:
            # 生成中性因子信号
            factor_value = signal.copy()
            factor_value[signal != 0] = signal.rolling(n, min_periods=1).sum()
            factor_value[signal == 0] = np.nan
            return factor_value



    def alpha_cta_signal_atrbolling_bias_reverse(self, n=20, cta_signal=0):
        """
        基于 ATR 和 Bollinger Bands 的逆向因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算ATR、标准差和中轨
        tr = np.maximum(
            abs(self.high - self.low),
            abs(self.high - self.close.shift(1)),
            abs(self.low - self.close.shift(1))
        )
        atr = tr.rolling(n, min_periods=1).mean()
        std = self.close.rolling(n, min_periods=1).std(ddof=0)
        median = self.close.rolling(n, min_periods=1).mean()

        # ATR 和 Bollinger Bands 神指标计算
        atr_J神 = abs(self.close - median) / atr
        boll_J神 = abs(self.close - median) / std

        m_atr = atr_J神.rolling(n, min_periods=1).max().shift(1)
        m_boll = boll_J神.rolling(n, min_periods=1).max().shift(1)

        # 计算ATR和Bollinger上轨和下轨
        upper_atr = median + m_atr * atr
        lower_atr = median - m_atr * atr
        upper_boll = median + m_boll * std
        lower_boll = median - m_boll * std

        # 平均上轨和下轨
        upper = (upper_atr + upper_boll) / 2
        lower = (lower_atr + lower_boll) / 2

        # 生成信号
        # 做空信号：当前收盘价上穿上轨
        signal = np.where((self.close > upper) & (self.close.shift(1) <= upper.shift(1)), -1, signal)
        # 做多信号：当前收盘价下穿下轨
        signal = np.where((self.close < lower) & (self.close.shift(1) >= lower.shift(1)), 1, signal)
        # 平仓信号：当前收盘价上穿中轨或下穿中轨
        signal = np.where((self.close < median) & (self.close.shift() >= median.shift()), 0, signal)
        signal = np.where((self.close > median) & (self.close.shift() <= median.shift()), 0, signal)

        # 信号去重与前向填充
        signal_df = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)
        signal_df.fillna(method='ffill', inplace=True)
        signal_df = signal_df[signal_df != signal_df.shift()]

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal_df = signal_df.fillna(method='ffill').fillna(0)
            return signal_df

        else:
            # 生成中性因子信号
            factor_value = signal_df.copy()
            factor_value[signal_df != 0] = signal_df.rolling(n, min_periods=1).sum()
            factor_value[signal_df == 0] = np.nan
            return factor_value


    def alpha_cta_signal_adapt_kc(self, n=20, cta_signal=0):
        """
        基于自适应 Keltner Channel (KC) 的因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        factor_name (str): 输出的因子名称。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算长周期和短周期的 ATR 和中轨
        n2 = 3 * n
        tr = np.maximum(
            abs(self.high - self.low),
            abs(self.high - self.close.shift(1)),
            abs(self.low - self.close.shift(1))
        )

        # 长周期 ATR 和自适应 Keltner Channel
        atr_long = tr.rolling(n2, min_periods=1).mean()
        median_long = self.close.ewm(span=180, min_periods=1, adjust=False).mean()
        z_score_long = abs(self.close - median_long) / atr_long
        m_long = z_score_long.rolling(window=n2).max().shift(1)
        upper_long = median_long + atr_long * m_long
        lower_long = median_long - atr_long * m_long

        # 短周期 ATR 和 Keltner Channel
        atr_short = tr.rolling(n, min_periods=1).mean()
        median_short = self.close.ewm(span=20, min_periods=1, adjust=False).mean()
        z_score_short = abs(self.close - median_short) / atr_short
        m_short = z_score_short.rolling(window=n).max().shift(1)
        upper_short = median_short + atr_short * m_short
        lower_short = median_short - atr_short * m_short

        # 条件定义：多头和空头
        condition_long = upper_short > upper_long
        condition_short = lower_short < lower_long

        # 做多信号：收盘价突破上轨，且上轨大于长周期上轨
        signal = np.where((self.close > upper_short) & (self.close.shift(1) <= upper_short.shift(1)) & condition_long, 1, signal)

        # 平仓信号：上轨下穿长周期上轨，或收盘价跌破下轨
        close_long_condition = (self.close < lower_short) & (self.close.shift() >= lower_short.shift())
        upper_long_condition = (upper_short < upper_long) & (upper_short.shift(1) >= upper_long.shift(1))
        signal = np.where(upper_long_condition | close_long_condition, 0, signal)

        # 做空信号：收盘价跌破下轨，且下轨小于长周期下轨
        signal = np.where((self.close < lower_short) & (self.close.shift(1) >= lower_short.shift(1)) & condition_short, -1, signal)

        # 平仓信号：下轨上穿长周期下轨，或收盘价突破上轨
        lower_long_condition = (lower_short > lower_long) & (lower_short.shift(1) <= lower_long.shift(1))
        upper_break_condition = (self.close > upper_short) & (self.close.shift() <= upper_short.shift())
        signal = np.where(lower_long_condition | upper_break_condition, 0, signal)

        # 将信号转换为 DataFrame
        signal_df = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal_df.fillna(method='ffill', inplace=True)
            signal_df = signal_df[signal_df != signal_df.shift()]
            return signal_df

        else:
            # 生成中性因子信号
            factor_value = signal_df.copy()
            factor_value[signal_df != 0] = signal_df.rolling(n, min_periods=1).sum()
            factor_value[signal_df == 0] = np.nan
            return factor_value


    def alpha_cta_signal_adapt_kc_with_rsi(self, n=20, cta_signal=0):
        """
        基于自适应 Keltner Channel 和 RSI 指标的因子信号生成。

        Parameters:
        n (int): 用于计算信号累加的周期。
        cta_signal (int): 0 表示生成中性因子信号；1 表示生成 CTA 信号。
        factor_name (str): 输出的因子名称。
        """

        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算 Keltner Channel 的相关指标
        tr = np.maximum(
            abs(self.high - self.low),
            abs(self.high - self.close.shift(1)),
            abs(self.low - self.close.shift(1))
        )
        atr = tr.rolling(n, min_periods=1).mean()
        median = self.close.ewm(span=20, min_periods=1, adjust=False).mean()
        z_score = abs(self.close - median) / atr
        m = z_score.rolling(window=n).max().shift(1)
        upper = median + atr * m
        lower = median - atr * m

        # 计算 RSI 指标
        close_diff = self.close.diff()
        closeup = close_diff.where(close_diff > 0, 0)
        closedown = close_diff.where(close_diff < 0, 0)
        closedown = closedown.abs()
        closeup_ma = closeup.rolling(n).mean()
        closedown_ma = closedown.rolling(n).mean()
        rsi = 100 * closeup_ma / (closeup_ma + closedown_ma)

        # 做多信号：收盘价突破上轨且 RSI > 70
        signal = np.where((self.close > upper) & (self.close.shift(1) <= upper.shift(1)) & (rsi > 70), 1, signal)

        # 平仓信号：RSI < 65 或者收盘价跌破下轨
        close_long_condition = (self.close < lower) & (self.close.shift() >= lower.shift())
        rsi_close_condition = (rsi < 65)
        signal = np.where(rsi_close_condition & close_long_condition, 0, signal)

        # 做空信号：收盘价跌破下轨且 RSI < 30
        signal = np.where((self.close < lower) & (self.close.shift(1) >= lower.shift(1)) & (rsi < 30), -1, signal)

        # 平仓信号：RSI > 35 或者收盘价突破上轨
        rsi_short_close = (rsi > 35)
        upper_break_condition = (self.close > upper) & (self.close.shift() <= upper.shift())
        signal = np.where(rsi_short_close & upper_break_condition, 0, signal)

        # 将信号转换为 DataFrame
        signal_df = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)

        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal_df = signal_df.fillna(method='ffill').fillna(0)
            signal_df = signal_df[signal_df != signal_df.shift()]
            return signal_df

        else:
            # 生成中性因子信号
            factor_value = signal_df.copy()
            factor_value[signal_df != 0] = signal_df.rolling(n, min_periods=1).sum()
            factor_value[signal_df == 0] = np.nan
            return factor_value
        
    def alpha_cta_signal_rsi(self, n=20, cta_signal=0):
        # 初始化信号为 NaN
        signal = np.full_like(self.close, np.nan)

        # 计算 RSI 指标
        close_diff = self.close.diff()
        closeup = close_diff.where(close_diff > 0, 0)
        closedown = close_diff.where(close_diff < 0, 0)
        closedown = closedown.abs()
        closeup_ma = closeup.rolling(n).mean()
        closedown_ma = closedown.rolling(n).mean()
        rsi = 100 * closeup_ma / (closeup_ma + closedown_ma)

        # 做多信号：RSI 跌破 30
        signal = np.where((rsi.shift(1) >= 30) & (rsi < 30), 1, signal)

        # 做空信号: RSI 突破 70
        signal = np.where((rsi.shift(1) <= 70) & (rsi > 70), -1, signal)

        # 平仓信号：40 <= RSI <= 60
        signal = np.where((rsi <= 60) & (rsi >= 40), 0, signal)

        # 将信号转换为 DataFrame
        signal_df = pd.DataFrame(signal, index=self.close.index, columns=self.close.columns)


        if cta_signal == 1:
            # 生成CTA信号，填充缺失值并去重
            signal_df = signal_df.fillna(method='ffill').fillna(0)
            signal_df = signal_df[signal_df != signal_df.shift()]
            return signal_df

        else:
            # 生成中性因子信号
            return rsi
        
    def alpha_cta_adapt_bolling_updated(self, n=7, cta_signal=0):
        """
        this factor transform from cta adapt bolling signal
        Editer: zqli
        factor = (close / median - 1) + cta_position + cta_signal
        """
        
        close = self.close
        median = close.rolling(n, min_periods=1).mean()
        std = close.rolling(n, min_periods=1).std(ddof=0)  # ddof代表标准差自由度
        z_score = abs(close - median) / std
        m = z_score.rolling(n, min_periods=1).mean().shift()
        upper = median + m * std
        lower = median - m * std
        median.fillna(method='ffill', inplace=True)
        std.fillna(method='ffill', inplace=True)
        z_score.fillna(method='ffill', inplace=True)
        m.fillna(method='ffill', inplace=True)
        upper.fillna(method='ffill', inplace=True)
        lower.fillna(method='ffill', inplace=True)
        bias = close / median - 1
        bias_pct = abs(bias).rolling(window=n, min_periods=1).max().shift()

        # ================= 根据波动率情况止盈 =======================
        std_rolling_rank = std.rolling(1500).rank(pct=True).fillna(method='ffill')
        close_long_position_threshold = median.copy()
        close_long_position_threshold = close_long_position_threshold.where(~(std_rolling_rank > 0.7), 0.9*(upper - median) + median)
        close_long_position_threshold = close_long_position_threshold.where(~((std_rolling_rank <= 0.7) & (std_rolling_rank >= 0.3)), 0.5*(upper - median) + median)

        close_short_position_threshold = median.copy()
        close_short_position_threshold = close_short_position_threshold.where(~(std_rolling_rank > 0.7), 0.9*(lower - median) + median)
        close_short_position_threshold = close_short_position_threshold.where(~((std_rolling_rank <= 0.7) & (std_rolling_rank >= 0.3)), 0.5*(lower - median) + median)



        # 描述上穿上轨
        condition1 = close > upper  # 当前K线的收盘价 > 上轨
        condition2 = close.shift(1) <= upper.shift(1)  # 之前K线的收盘价 <= 上轨

        signal_long = pd.DataFrame(index=condition1.index, columns=condition1.columns)
        signal_long[condition1 & condition2] = 1

        # 描述下穿中轨
        # condition1 = close < median  # 当前K线的收盘价 < 中轨
        # condition2 = close.shift(1) >= median.shift(1)  # 之前K线的收盘价 >= 中轨
        

        condition1 = close < close_long_position_threshold
        condition2 = close.shift(1) >= close_long_position_threshold.shift(1)

        # signal_long.where(~(condition1 & condition2), 0, inplace=True)
        signal_long[condition1 & condition2] = 0

        # 描述下穿下轨
        condition1 = close < lower  # 当前K线的收盘价 < 下轨
        condition2 = close.shift(1) >= lower.shift(1)  # 之前K线的收盘价 >= 下轨
        signal_short = pd.DataFrame(index=condition1.index, columns=condition1.columns)
        # signal_short.where(~(condition1 & condition2), -1, inplace=True)
        signal_short[condition1 & condition2] = -1
        # df.loc[condition1 & condition2,
        #        'signal_short'] = -1  # 将产生做空信号的那根K线的signal设置为-1，-1代表做空

        # 上穿中轨
        # condition1 = close > median  # 当前K线的收盘价 > 中轨
        # condition2 = close.shift(1) <= median.shift(1)  # 之前K线的收盘价 <= 中轨

        condition1 = close > close_short_position_threshold
        condition2 = close.shift(1) <= close_short_position_threshold.shift(1)
        # df.loc[condition1 & condition2,
        #        'signal_short'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓
        # signal_short.where(~(condition1 & condition2), 0, inplace=True)
        signal_short[condition1 & condition2] = 0


        position_short = signal_short.fillna(method='ffill').fillna(0)
        position_long = signal_long.fillna(method='ffill').fillna(0)


        # signal = df[['signal_long', 'signal_short']].sum(axis=1)
        position = position_long + position_short
        signal = position[position != position.shift(1)]
    
        # signal 已经是 position 

        raw_signal = signal
        # 止损方法
        # 通过 bias_pct 避免偶然的波动开仓
        temp = deepcopy(signal)
        condition1 = (signal == 1)
        condition2 = (bias > bias_pct)
        # df.loc[condition1 & condition2, 'temp'] = None
        temp[condition1 & condition2] = 0

        condition1 = (signal == -1)
        condition2 = (bias < -bias_pct)
        # df.loc[condition1 & condition2, 'temp'] = None
        temp[condition1 & condition2] = 0

        position = temp.fillna(method='ffill').fillna(0)
        signal = position[position != position.shift(1)]
        

        

        # 止盈
        # temp_signal = deepcopy(signal)
        # condition1 = (position == 1)
        # condition2 = (close - upper) > 3*(upper - median)
        # condition3 = (signal.isnull())
        # temp_signal[condition1 & condition2 & condition3] = 0

        # condition1 = (position == -1)
        # condition2 = (close - lower) < 3*(lower - median)
        # condition3 = (signal.isnull())
        # temp_signal[condition1 & condition2 & condition3] = 0

        # position = temp_signal.fillna(method='ffill').fillna(0)
        # signal = position[position != position.shift(1)]

        

        if cta_signal:
            return signal#, {'upper': upper, 'median': median, 'lower': lower, 'close_long': close_long_position_threshold, 'close_short': close_short_position_threshold}
        ########
        # 改写格式后进一步改写为 多因子中性策略因子，不执行一下代码，直接返回 signal 则就是CTA的开平多空仓信号结果
        ########
        factor_value = (close - median) / (upper - median) + position + signal.fillna(0)
        return factor_value



    def alpha_mmt_bvol_ratio_volatility(self, n=144, a=0.1):  
            # alpha_028
            f1 = 3
            f2 = 2
            temp1=self.close-self.low.rolling(n).min()
            temp2=self.high.rolling(n).max()-self.low.rolling(n).min()
            part1=f1*(temp1*100/temp2).ewm(alpha=a).mean()
            temp3=(temp1*100/temp2).ewm(alpha=a).mean()
            part2=f2*temp3.ewm(alpha=a).mean()
            alpha_028 = part1-part2

            # 主动成交占比
            volume = self.quote_volume.rolling(n, min_periods=1).sum()
            buy_volume = self.taker_buy_quote_volume.rolling(n, min_periods=1).sum()
            taker_buy_ratio = buy_volume / volume

            # 波动率因子
            c1 = self.high - self.low
            c2 = abs(self.high - self.close.shift(1))
            c3 = abs(self.low - self.close.shift(1))
            tr = np.maximum(c1,c2,c3)
            atr = tr.rolling(n, min_periods=1).mean()
            avg_price = self.close.rolling(n, min_periods=1).mean()
            wd_atr = atr / avg_price

            # 动量 * 主动成交占比 * 波动率
            factor = alpha_028 * taker_buy_ratio * wd_atr

            return factor.stack()
    
    def alpha_mmt_dabu1(self, n=4):
        max_high = self.high.rolling(n, min_periods=1).max()
        min_low = self.low.rolling(n, min_periods=1).min()
        factor = (self.close - min_low) / (max_high - min_low)
        return factor
    
    def alpha_mmt_dabu2(self, n=4):
        tp = (self.close + self.high + self.low) / 3
        tp2 = tp.ewm(span=n, adjust=False).mean()
        diff = tp - tp2
        min = diff.rolling(n, min_periods=1).min()
        max = diff.rolling(n, min_periods=1).max()

        factor = (diff - min) / (max - min)
        return factor
    
    def alpha_mmt_dabu3(self, n=4):
        ma = self.close.rolling(n, min_periods=1).mean()
        std = self.close.rolling(n, min_periods=1).std()
        zscore = (self.close - ma) / std
        zscore = zscore.abs().rolling(n, min_periods=1).mean().shift()
        down = ma - zscore * std

        diff = self.close - down
        min = diff.rolling(n, min_periods=1).min()
        max = diff.rolling(n, min_periods=1).max()

        factor = (diff - min) / (max - min)
        return factor