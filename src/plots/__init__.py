import numpy as np
import pandas as pd

from plots.radar import radar
from plots.read import load_metrics


def main() -> int:
    df = load_metrics("AEEEM", "f1")
    print(pd.concat([df.our, df.others], axis=1))
    radar(df)
    return 0
