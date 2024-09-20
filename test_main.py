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


def load_dataset_pd(dataset):
    df = pd.read_csv(dataset)
    return df


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
    print(pl_describe(df))
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
    assert get_mean(dataframe, "Y") == 19.975792520390325


def test_median():
    """Test the get_median function"""
    assert get_median(dataframe, "Y") == 19.97102000166825


def test_std():
    """Test the get_std function"""
    assert get_std(dataframe, "Y") == 5.004964559422916


if __name__ == "__main__":
    test_mean()
    test_median()
    test_std()
    create_boxplot(dataframe["Y"], "boxplot.png")
    mean_y = get_mean(dataframe, "Y")
    median_y = get_median(dataframe, "Y")
    std_y = get_std(dataframe, "Y")
    save_to_md(mean_y, median_y, std_y)
