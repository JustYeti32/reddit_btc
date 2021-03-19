import ccxt
import datetime as dt
import pandas as pd
import os


class crypto_scraper:
    def __init__(self, trading_pair, timeframe):
        self.exchange = ccxt.binance()
        self.trading_pair = trading_pair
        if timeframe == "5min":
            self.timeframe = "5m"
            self.timeframe_min = 5
        else:
            print("timeframe not implemented")

        self._limit = 1000
        self._second = 1000
        self._minute = 60 * 1000

    def scrape(self, start_date, end_date, verbose=True):
        module_path = os.path.dirname(__file__)
        data_path = module_path + "/../data"

        if verbose:
            print("Start scraping ", self.timeframe, " ohlcv data from binance")
            print(70 * "-")
            print("Start date: ", start_date, " | End date: ", end_date)
            print(70*"-", "\n")

        start_unix_ms = self.time_to_unix_ms(start_date)
        end_unix_ms = self.time_to_unix_ms(end_date)

        candles = pd.DataFrame()
        end_fetch_unix_ms = end_unix_ms
        while end_fetch_unix_ms >= start_unix_ms:
            fetched_candles, end_fetch_unix_ms = self.fetch(end_fetch_unix_ms, verbose)
            candles = pd.concat([fetched_candles, candles])

        candles = candles.loc[candles.date >= start_unix_ms]
        candles.date = candles.date.apply(self.unix_ms_to_time)
        candles.set_index("date", inplace=True)
        candles.to_csv(data_path + "/btc_ohlcv")

        if verbose:
            print("\n")
            print(70*"-")
            print("Finished scraping")
            print("Scraped ", len(candles), " rows of data")
            print(70 * "-")

        return candles

    def fetch(self, end_fetch_unix_ms, verbose):
        start_fetch_unix_ms = end_fetch_unix_ms - self._minute * self.timeframe_min * self._limit
        candles = self.exchange.fetch_ohlcv(
            self.trading_pair,
            timeframe=self.timeframe,
            since=start_fetch_unix_ms,
            limit=self._limit
        )

        if verbose:
            start_fetch_time = self.unix_ms_to_time(start_fetch_unix_ms)
            end_fetch_time = self.unix_ms_to_time(end_fetch_unix_ms)
            print_string = 4*" " + "Fetch data from {} to {}"
            print(print_string.format(end_fetch_time, start_fetch_time), end="\r")

        end_fetch_unix_ms = candles[0][0]
        candles = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume"])

        return candles, end_fetch_unix_ms

    def time_to_unix_ms(self, time):
        return int(dt.datetime.strptime(time, "%Y-%m-%d %H:%M:%S").timestamp() * self._second)

    def unix_ms_to_time(self, unix_time):
        return dt.datetime.fromtimestamp(unix_time / self._second)

########################################################################################################################