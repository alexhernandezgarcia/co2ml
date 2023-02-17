"""
This script plots a map color-coding the distribution of countries where the models
where trained.
"""
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.ticker import LogLocator
from scipy.stats import trim_mean

# -----------------------
# -----  Constants  -----
# -----------------------


def parsed_args():
    """
    Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-y",
        "--yaml_config",
        default=None,
        type=str,
        help="YAML configuration file",
    )
    parser.add_argument(
        "--input_csv",
        default="../data/all_merged_20220104.csv",
        type=str,
        help="CSV containing the data set",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--dpi",
        default=200,
        type=int,
        help="DPI for the output images",
    )

    return parser.parse_args()


def year2years(year):
    if year >= 2012 and year < 2017:
        return "2012-2016"
    elif year >= 2017 and year < 2018:
        return "2017"
    elif year >= 2018 and year < 2019:
        return "2018"
    elif year >= 2019 and year < 2020:
        return "2019"
    elif year >= 2020 and year < 2021:
        return "2020"
    elif year >= 2021 and year < 2022:
        return "2021"


def plot_map(df, args):
    # Set up plot
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update(
        {
            "font.serif": [
                "Computer Modern Roman",
                "Times New Roman",
                "Utopia",
                "New Century Schoolbook",
                "Century Schoolbook L",
                "ITC Bookman",
                "Bookman",
                "Times",
                "Palatino",
                "Charter",
                "serif" "Bitstream Vera Serif",
                "DejaVu Serif",
            ]
        }
    )

    fig, ax = plt.subplots(figsize=(20, 20), dpi=args.dpi)
    df.plot(
        ax=ax,
        column="count",
        legend=True,
        legend_kwds={"label": "count", "orientation": "horizontal"},
        missing_kwds={"color": "lightgrey"},
    )

    # Ticks and tick labels
    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )
    # Change spines
    sns.despine(ax=ax, left=True, bottom=True)

    return fig


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------
    args = parsed_args()
    if args.yaml_config is not None:
        yaml_path = Path(args.yaml_config)
        with yaml_path.open("r") as f:
            config = yaml.safe_load(f)
            for k, v in vars(args).items():
                if k not in config:
                    config.update({k: v})
        args = Namespace(**config)
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(args).items()]))

    # Determine output dir
    if args.output_dir is None:
        output_dir = Path(os.environ["SLURM_TMPDIR"])
    else:
        output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)

    # Store args
    output_yml = output_dir / "map_count_countries.yml"
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col=False)

    # Year(s)
    df["Year(s)"] = df["Year published"].map(year2years, na_action="ignore")

    # Get base map df
    df_country_count = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Count occurrences of each country
    df_country_count.rename(
        columns={
            "name": "Country",
        },
        inplace=True,
    )
    df_country_count["count"] = np.zeros(len(df_country_count), dtype=int)
    total = 0
    for country in df_country_count["Country"].unique():
        n = len(df.loc[df["Country"].str.contains(country)])
        if n > 0:
            print(f"{country}: {n}")
            df_country_count.loc[df_country_count["Country"] == country, "count"] = n
        else:
            df_country_count.loc[df_country_count["Country"] == country, "count"] = None
        total += n
    print(f"Total: {total}")

    fig = plot_map(df_country_count, args=args)

    # Save figure
    output_fig = output_dir / "map_count_countries.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
