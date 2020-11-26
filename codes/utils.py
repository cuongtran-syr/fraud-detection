import numpy as np
from scipy.stats import entropy
from math import log, e
import pandas as pd

def pandas_entropy(column, base=None):
  """
  Compute the entropy for a categorical column in a pandas table
  Example: pandas_entropy(pd00['merchantName']
  
  """
  vc = pd.Series(column).value_counts(normalize=True, sort=False)
  base = e if base is None else base
  return -(vc * np.log(vc)/np.log(base)).sum()


def is_