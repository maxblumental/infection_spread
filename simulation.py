from typing import Dict, List

import numpy as np
import pandas as pd

# airport -> infection timestamp
Simulation = Dict[str, int]

DAY_IN_SEC = 24 * 60 * 60
HALF_DAY_IN_SEC = 12 * 60 * 60

# serves Allentown, Bethlehem, Easton, and surrounding areas
LEHIGH_VALLEY_AIRPORT = "ABE"


def simulate(flights: pd.DataFrame, infection_proba: float, start_airport: str = LEHIGH_VALLEY_AIRPORT) -> Simulation:
    """
    :param flights: [Origin - Dest - ArrTs] dataframe
    :param infection_proba:
    :param start_airport: infection source
    :return:  {airport: infection_ts}
    """
    infected = {start_airport: flights.ArrTs.min()}

    for _, row in flights.iterrows():
        if row.Origin in infected and row.Dest not in infected:
            if np.random.rand() < infection_proba:
                infected[row.Dest] = int(row.ArrTs)

    return infected


def make_simulation_df(simulation: Simulation) -> pd.DataFrame:
    """
    Split the simulation onto half-day periods
    and count infected airports per each period.
    :param simulation: {airport: infection_ts}
    :return: [time, total_infected] dataframe
    """
    data = [
        {'airport': airport, 'time': ts // HALF_DAY_IN_SEC}
        for airport, ts in simulation.items()
    ]
    df = pd.DataFrame(data).groupby('time').count().reset_index()
    df['total_infected'] = df.airport.cumsum()
    df.time = pd.to_datetime(df.time * HALF_DAY_IN_SEC, origin='unix', unit='s')
    df = df.set_index('time').sort_index()
    return df


def infection_stats(simulations: List[Simulation]) -> pd.DataFrame:
    """
    :param simulations: list of {airport: infection_ts}
    :return: [time, infected_mean, infected_std] dataframe
    """
    sim_dfs = [make_simulation_df(sim) for sim in simulations]
    common_index = pd.concat(sim_dfs).index.unique().sort_values()

    df = pd.concat([sim_df.reindex(common_index).fillna(method='ffill').fillna(0) for sim_df in sim_dfs])
    return pd.DataFrame({
        'infected_mean': df.groupby(df.index).mean().total_infected,
        'infected_std': df.groupby(df.index).std().total_infected,
    })


def make_time_to_infection_df(simulations: List[Simulation]) -> pd.DataFrame:
    """
    :param simulations: list of {airport: infection_ts}
    :return: [airport, mean time to infection] dataframe
    """
    data = []
    for sim in simulations:
        t0 = min(sim.values())
        data.extend([
            {'airport': airport, 'time2infection': (t - t0) / DAY_IN_SEC}
            for airport, t in sim.items()
        ])
    return pd.DataFrame(data).groupby('airport').mean()
