import csv
import pandas as pd

def save_to_csv(df, name, quoting=csv.QUOTE_ALL):
    df.to_csv(name, index=False, quoting=quoting)
