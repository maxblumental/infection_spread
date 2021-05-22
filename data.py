import pandas as pd

HOUR_IN_SEC = 60 * 60
DAY_IN_SEC = 24 * 60 * 60


def make_flights_df(path: str) -> pd.DataFrame:
    """
    - Select only 1 month
    - Drop faulty flights
    - Make a column with arrival timestamps
    :param path: csv with US Flights Data 2008 (https://www.kaggle.com/vikalpdongre/us-flights-data-2008)
    :return: [Origin, Dest, ArrTs] dataframe
    """
    df = pd.read_csv(path)
    df = df[df.Month == 6]
    df = df[(df.Cancelled == 0) & ~df.ArrTime.isna() & ~df.DepTime.isna()]

    dates = pd.to_datetime(df.rename(columns={'DayofMonth': 'Day'})[['Year', 'Month', 'Day']])
    date_ts = dates.astype(int) // 10 ** 9
    next_day = (df.ArrTime <= df.DepTime).astype(int)
    hours = df.ArrTime / 100
    arrival_ts = date_ts + hours * HOUR_IN_SEC + next_day * DAY_IN_SEC
    df['ArrTs'] = arrival_ts.astype(int)

    df = df[['Origin', 'Dest', 'ArrTs']].sort_values('ArrTs')

    return df
