from typing import NamedTuple

import pandas as pd


class Metric(NamedTuple):
    dataset_name: str
    others: pd.DataFrame
    our: pd.DataFrame
