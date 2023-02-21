# purpose: plotting a graph showing volume of multiple levels in the orderbook over time with the corresponding market orders at those times
# packages
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
from statistics import median
import pandas as pd
import glob
import os
import nest_asyncio
from tardis_dev import datasets
from scipy.optimize import curve_fit
import warnings
import os.path; from os import path
import pickle as pkl
nest_asyncio.apply()
warnings.filterwarnings("ignore")


def getOB(name, startdate, enddate, ticks, loc, download_only = False):
    filename = os.path.join(loc + "\\data\\" + name + "USD\\", "coinbase_book_snapshot_25_" + startdate + "_" + name + "-USD.csv.gz")
    if not path.exists(filename):
        print("downloading data orderbook")
        datasets.download(
        exchange="coinbase",
        data_types=[
            "book_snapshot_25"],
        from_date = startdate,
        to_date = enddate,
        symbols = [name + "-USD"],
        download_dir = loc + "\\data\\" + name + "USD",
        api_key= "TD.CTo-GFKD9yJojKmk.0fsYIehJ-wDD1e-.FVe4814H5x7BkM7.uRoxlXUJQmcdRPz.HWyzjjLu4LiqBi4.sF6W",
        )
        
        orderbook_database =  glob.glob(os.path.join(loc + "\\data\\" + name + "USD\\", "coinbase_book_snapshot_25*" + startdate + "_" + name + "-USD.csv.gz"))
        df_orderbook = pd.concat((pd.read_csv(f, nrows=70000) for f in orderbook_database), ignore_index=True)
        df_orderbook['timestamp'] = pd.to_datetime(df_orderbook.timestamp * 1000)
        df_orderbook.sort_values(by='timestamp', inplace=True)
    else:
        print("data Orderbook already exists!")

    if not download_only:
        print("oke ff kijkenn")
        orderbook_database =  glob.glob(os.path.join(loc + "\\data\\" + name + "USD\\", "coinbase_book_snapshot_25*" + startdate + "_" + name + "-USD.csv.gz"))
        print("hoe lang heeft dit geduurt")
        df_orderbook = pd.concat((pd.read_csv(f, nrows=70000) for f in orderbook_database), ignore_index=True)
        print("en dit?")
        df_orderbook['timestamp'] = pd.to_datetime(df_orderbook.timestamp * 1000)
        df_orderbook.sort_values(by='timestamp', inplace=True)
        print(df_orderbook)
        return df_orderbook


def getTrades(name, startdate, enddate, trades, loc, download_only = False):
    print("getTrades")
    filename_trades = os.path.join(loc + "\\data\\" + name + "USD\\", "coinbase_trades_" + startdate + "_" + name + "-USD.csv.gz")
    filename_quotes = os.path.join(loc + "\\data\\" + name + "USD\\", "coinbase_quotes_" + startdate + "_" + name + "-USD.csv.gz")
    if not path.exists(filename_trades) or not path.exists(filename_quotes):
        print("downloading data..")
        datasets.download(
        exchange="coinbase",
        data_types=["quotes"],
        from_date = startdate,
        to_date = enddate,
        symbols = [name + "-USD"],
        download_dir = loc + "\\data\\" + name + "USD",
        api_key= "TD.CTo-GFKD9yJojKmk.0fsYIehJ-wDD1e-.FVe4814H5x7BkM7.uRoxlXUJQmcdRPz.HWyzjjLu4LiqBi4.sF6W",
        )
        print("downloading data trades", name)
        datasets.download(
        exchange="coinbase",
        data_types=[
            "trades"],
        from_date = startdate,
        to_date = enddate,
        symbols = [name + "-USD"],
        download_dir = loc + "\\data\\" + name + "USD",
        api_key= "TD.CTo-GFKD9yJojKmk.0fsYIehJ-wDD1e-.FVe4814H5x7BkM7.uRoxlXUJQmcdRPz.HWyzjjLu4LiqBi4.sF6W",
        )

        quote_database = glob.glob(os.path.join(loc + "\\data\\" + name + "USD\\", "coinbase_quotes*" + startdate + "_" + name + "-USD.csv.gz"))
        df_quotes = pd.concat((pd.read_csv(f) for f in quote_database), ignore_index=True)
        df_quotes['timestamp'] = pd.to_datetime(df_quotes.timestamp * 1000)
        df_quotes(by='timestamp', inplace=True)
        df_quotes['midprice'] = (df_quotes['ask_price'] + df_quotes['bid_price'])/2

        trade_database =  glob.glob(os.path.join(loc + "\\data\\" + name + "USD\\", "coinbase_trades*" + startdate + "_" + name + "-USD.csv.gz"))
        df_trades = pd.concat((pd.read_csv(f, nrows=trades) for f in trade_database), ignore_index=True)
        df_trades['timestamp'] = pd.to_datetime(df_trades.timestamp * 1000)
        df_trades.sort_values(by='timestamp', inplace=True)
        df_trades = df_trades[df_trades['timestamp'] >= df_quotes.loc[0, 'timestamp']]
        df_trades.index = range(len(df_trades))

        print(df_trades)
        return df_trades
    
    else:
        print("data Trades already exists")

    if not download_only:
        quote_database = glob.glob(os.path.join(loc + "\\data\\" + name + "USD\\", "coinbase_quotes*" + startdate + "_" + name + "-USD.csv.gz"))
        df_quotes = pd.concat((pd.read_csv(f) for f in quote_database), ignore_index=True)
        df_quotes['timestamp'] = pd.to_datetime(df_quotes.timestamp * 1000)
        df_quotes.sort_values(by='timestamp', inplace=True)
        df_quotes['midprice'] = (df_quotes['ask_price'] + df_quotes['bid_price'])/2

        trade_database =  glob.glob(os.path.join(loc + "\\data\\" + name + "USD\\", "coinbase_trades*" + startdate + "_" + name + "-USD.csv.gz"))
        df_trades = pd.concat((pd.read_csv(f, nrows=trades) for f in trade_database), ignore_index=True)
        df_trades['timestamp'] = pd.to_datetime(df_trades.timestamp * 1000)
        df_trades.sort_values(by='timestamp', inplace=True)
        df_trades = df_trades[df_trades['timestamp'] >= df_quotes.loc[0, 'timestamp']]
        df_trades.index = range(len(df_trades))

        print("hallo")
        print(df_trades)
        print("doei")
        return df_trades       


def getAggrOB(df):
    print("getAggrOB")
    bid_amount_cols = ["bids[" + str(x) + "].amount" for x in range(len(LEVELS) * 2)]; ask_amount_cols = ["asks[" + str(x) + "].amount" for x in range(len(LEVELS) * 2)]
    #bid_price_cols = ["bids[" + str(x) + "].price" for x in range(10)]; ask_price_cols = ["asks[" + str(x) + "].price" for x in range(10)]
    #for these plots we only look at the quantities instead of a product that calculates the $ value. Reason: We compare the token quantities in the orderbook to the realized market orders which are also in a token quantity
    df['l0 bids aggr'] = df[bid_amount_cols[0]] 
    df['l0 asks aggr'] = df[ask_amount_cols[0]]

    for i in range(1, len(LEVELS) * 2):
        df['l' + str(i) + ' bids aggr'] = df[bid_amount_cols[i]] + df['l' + str(i - 1) + ' bids aggr']
        df['l' + str(i) + ' asks aggr'] = df[ask_amount_cols[i]] + df['l' + str(i - 1) + ' asks aggr']

    df_lvls = df.iloc[:,-(LEVELS[-1]+1)*2:]
    df_lvls.index = df['timestamp']

    
    print(df_lvls)
    return df_lvls


def plot_it(dfOB, dfOBagg, dfTrades, TICKS, TRADES, LEVELS, COIN_NAME, save = False):
    if len(LEVELS) >= 9:
        print("WARNING: ADD MORE COLORS TO PLOT")

    print(dfOBagg)

    cols_select_bids = [x * 2 for x in LEVELS];
    cols_select_asks = [x * 2 + 1 for x in LEVELS]
    df_subset = dfOBagg.iloc[:, cols_select_bids + cols_select_asks]

    current_timestamp = dfOB['timestamp'][0]
    list=[]
    #list = [False for i in range(TICKS)]
    iter = 0
    marketOrdersTimestampList = []
    dfOBagg['sell_amount'] = 0
    dfOBagg['buy_amount'] = 0
    

    for i in range(len(dfTrades)):
        # print(i)
        # bestTimestamp = current_timestamp

        market_order = dfTrades['timestamp'][i]
        # current_tick = dfOB[(dfOB['timestamp'] < market_order)].iloc[-1]
        # dfOB[(dfOB['timestamp'] < market_order)].iloc[-1]
        current_index = dfOB.index[(dfOB['timestamp'] < market_order)][-1]
        list.append(current_index)
        # print(last_index)
        # bestTimestamp = last_tick['timestamp']
        if dfTrades['side'][i] == 'buy':
            # print(dfTrades['amount'][i])
            # current_tick['buy_amount'] = dfTrades['amount'][i]
           dfOBagg.iloc[current_index, dfOBagg.columns.get_loc('buy_amount')] += dfTrades['amount'][i]
        elif dfTrades['side'][i] == 'sell':
           dfOBagg.iloc[current_index, dfOBagg.columns.get_loc('sell_amount')] -= dfTrades['amount'][i]
            # current_tick['sell_amount'] = dfTrades['amount'][i]


    
        # marketOrdersTimestampList.append({
        #     'timestamp': last_tick['timestamp'],
        #     'amount': dfTrades['amount'][i],
        #     'side': dfTrades['side'][i]            
        # })

        # for j in range(iter, TICKS):
        #     if bestTimestamp < dfTrades['timestamp'][i]:
        #         bestTimestamp = dfOB['timestamp'][j]
        #         iter += 1
           # if dfOB['timestamp'][j] >= dfTrades['timestamp'][i]:
            #    iter += 1
             #   break
        # current_timestamp = bestTimestamp
        # list.append(iter)
    # plt.plot(df_aggrOB['timestamp'], df_aggrOB['l0 bids aggr'])
    bid_level_cols_subset = ['l' + str(x) + ' bids aggr' for x in range(len(LEVELS))]
    ask_level_cols_subset = ['l' + str(x) + ' asks aggr' for x in range(len(LEVELS))]
    y = bid_level_cols_subset + ask_level_cols_subset

    # print(y)
    
    # plt.subplot(df_aggrOB['timestamp'], -1 * df_aggrOB['l0 asks aggr'])


    # for iter in list:
    #    marketOrdersTimestampList.append(dfOB[iter]['timestamp'])
    #    marketOrdersTimestampList['amount'] = dfTrades[iter]['amount']
    
    # print(dfOB.describe())

    # df_plot = df_subset.iloc[list]
    all_bids = [x * 2 for x in range(len(y))]
    # print(all_bids)
    df_aggrOB.iloc[:, all_bids] = -df_aggrOB.iloc[:, all_bids]
    # print(df_plot)   

    #voor er geplot kan worden moet de trade gelinkt worden op ask/bid (onderscheid maken tussen market buys en sells). Sell order moet bid levels krijgen en buy order ask level. Ook is er een out of bounds error

    # bid_level_cols_subset = ['l' + str(x) + ' bids aggr' for x in LEVELS]; ask_level_cols_subset = ['l' + str(x) + ' asks aggr' for x in LEVELS]
    # y = bid_level_cols_subset + ask_level_cols_subset

    # max 16 lines possible with 8 standard colors
    colors = {}
    # colors = colors[0:len(LEVELS)]; colors_r = colors; colors = colors + colors_r
    # colors = ''

    for x in ask_level_cols_subset:
        colors[x] = '#d63344'
    
    for x in bid_level_cols_subset:
        colors[x] = '#3dd951'
    
    print(colors)
    # for x in range(len(LEVELS)):
    #     if x % 2 == 0:

    #         colors += 'r'
    #     else:
    #         colors += 'g'

    ax = df_aggrOB.plot(y = y, alpha=0.25, color=colors, linewidth=0, stacked=False, kind='area')
    dfOBagg.plot(y = ['buy_amount', 'sell_amount'], ax=ax, alpha=0.25)
    ax.get_legend().remove()

    # df_aggrOB.subplot
    plt.title(COIN_NAME + " LOB Levels")
    if save:
        plt.tight_layout()
        plt.savefig(r"C:\Users\Temp_Student_001\Desktop\research\0.projects\Lukas\Project4\images_with_market_orders\\" + COIN_NAME + "_LOB_Levels (autosave).png", dpi = 200)
    plt.show()






# THIS SECTION IS WHERE YOU RUN STUFF
if __name__ == "__main__":
    # inputs
    COIN_NAME = "LINK"
    LEVELS = [0, 3, 6] #start with 0/best bid-ask, then all the way to 10 instead of the 25 that we did in the earlier file
    TICKS = 70000 # amount of trades
    LOC = r"C:\Users\Temp_Student_001\Desktop\research\0.projects\Lukas\Project 4"
    START_DATE = '2022-11-09'
    END_DATE = '2022-11-10'
    TRADES = 1000

    ### IF YOU WANT TO RUN ONE COIN RU
    # grab 30 levels of orderbook tick data
    # can take a little time
    # dfOB = getOB(COIN_NAME, START_DATE, END_DATE, TICKS, loc = LOC)

    LOC = r"C:\Users\Temp_Student_001\Desktop\research\0.projects\Tijn"
    # dfTrades = getTrades(COIN_NAME, START_DATE, END_DATE, TRADES, loc = LOC)

    # build the aggregate OB
    # this can take a little time as well
    # df_aggrOB = getAggrOB(dfOB)

    # plot
    # plot_it(dfOB, df_aggrOB, dfTrades, TICKS, TRADES, LEVELS, COIN_NAME, save = True)

    ### MULTIPLE COINS
    # if you want to run multiple plots in one go
    # in this manner the plots will slowly roll out one by one 
    COIN_NAMES = ["ADA", "MATIC", "LINK", "DOGE", "SHIB", "XLM", "MANA", "SOL", "LTC", "USDT"]

    for i in COIN_NAMES:
        dfTrades = getTrades(i, START_DATE, END_DATE, TRADES, loc = LOC)
        dfOB = getOB(i, START_DATE, END_DATE, TICKS, loc = LOC)
        df_aggrOB = getAggrOB(dfOB)
        plot_it(dfOB, df_aggrOB, dfTrades, TICKS, TRADES, LEVELS, i, save = False)
     