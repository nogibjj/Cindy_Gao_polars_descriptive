from main import (
    load_dataset_pl,
    pl_describe,
    get_mean,
    get_median,
    get_std,
    save_to_md,
    create_boxplot,
)

import polars as pl
import pandas as pd
from pyinstrument import Profiler

data = "https://raw.githubusercontent.com/anlane611/datasets/main/population.csv"
dataframe = load_dataset_pl(data)

print(dataframe)


# Define loading dataset and decribe functions using pandas:
def load_dataset_pd(dataset):
    df_pd = pd.read_csv(dataset)
    return df_pd


def describe_pd(df):
    df_description = df.describe()
    return df_description


# Use Profiler to compare pandas and polars
with Profiler(interval=0.1) as profiler:
    check_pl = data
    df = load_dataset_pl(check_pl)
    print(df.shape)
    print(pl_describe(df))
    print(df["Y"].mean)
    print(df["Y"].median)

profiler.print()

with Profiler(interval=0.1) as profiler:
    check_pd = data
    df = load_dataset_pd(check_pd)
    print(df.shape)
    print(describe_pd(df))
    print(df["Y"].mean)
    print(df["Y"].mean)

profiler.print()


# Print descriptive statistics
print(pl_describe(dataframe))
print(get_mean(dataframe, "Y"))
print(get_median(dataframe, "Y"))
print(get_std(dataframe, "Y"))


# Define test functions
def test_mean():
    """Test the get_mean function"""
    assert round(get_mean(dataframe, "Y"), 3) == 19.976


def test_median():
    """Test the get_median function"""
    assert round(get_median(dataframe, "Y"), 3) == 19.971


def test_std():
    """Test the get_std function"""
    assert round(get_std(dataframe, "Y"), 3) == 5.005


if __name__ == "__main__":
    test_mean()
    test_median()
    test_std()
    create_boxplot(dataframe["Y"], "boxplot.png")
    mean_y = get_mean(dataframe, "Y")
    median_y = get_median(dataframe, "Y")
    std_y = get_std(dataframe, "Y")
    save_to_md(mean_y, median_y, std_y)
