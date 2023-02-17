"""
This script plots a map color-coding the median carbon intensity in the countries
where the models where trained.
"""
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import cm
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import trim_mean

# -----------------------
# -----  Constants  -----
# -----------------------
# Colors
palette_set1 = sns.color_palette("Set1")
palette = {
    "Coal": palette_set1[6],
    "Oil": palette_set1[0],
    "Gas": palette_set1[4],
    "Nuclear": palette_set1[3],
    "Hydro": palette_set1[1],
}
hue_order = ["Coal", "Oil", "Gas", "Nuclear", "Hydro"]


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
    parser.add_argument(
        "--crop_antarctica",
        default=False,
        action="store_true",
        help="Cut out the Antarctica region",
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.02)
    magma_r = cm.get_cmap("magma_r")
    df.plot(
        ax=ax,
        cax=cax,
        column="Median carbon intensity [g CO₂eq/kWh]",
        legend=True,
        legend_kwds={
            "label": "Median carbon intensity [g CO₂eq/kWh]",
            "orientation": "horizontal",
            "pad": 0.025,
        },
        cmap=magma_r,
        missing_kwds={"color": "lightgrey"},
    )

    # Legend
    handles = []
    df.sort_values(by="count", inplace=True, ascending=False)
    for idx, row in df.loc[df["count"].isnull() == False].iterrows():
        main_energy_source = row["Main energy source"]
        count = int(row["count"])
        country = row["Country"]
        handles.append(
            mpatches.Patch(
                color=palette[main_energy_source],
                label=f"{country}: {count}",
            )
        )
    handles.append(mlines.Line2D([], [], color="black", markersize=0, label=""))
    handles = handles + [
        mpatches.Patch(color=palette[el], label=el) for el in hue_order
    ]

    # Antarctica
    if args.crop_antarctica:
        ax.set_ylim([-60, ax.get_ylim()[1]])

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
    output_yml = output_dir / "map_carbonintensity.yml"
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col=False)

    # Year(s)
    df["Year(s)"] = df["Year published"].map(year2years, na_action="ignore")

    # Get base map df
    df_countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Count occurrences of each country and average carbon intensity
    df_countries.rename(
        columns={
            "name": "Country",
        },
        inplace=True,
    )
    df_countries["count"] = np.zeros(len(df_countries), dtype=int)
    df_countries["Median carbon intensity [g CO₂eq/kWh]"] = np.zeros(
        len(df_countries), dtype=int
    )
    df_countries["Main energy source"] = [None] * len(df_countries)
    total = 0
    for country in df_countries["Country"].unique():
        df_country = df.loc[df["Country"].str.contains(country)]
        n = len(df_country)
        if n > 0:
            print(f"{country}: {n}")
            df_countries.loc[df_countries["Country"] == country, "count"] = n
            df_countries.loc[
                df_countries["Country"] == country,
                "Median carbon intensity [g CO₂eq/kWh]",
            ] = df_country["Carbon intensity [g CO₂eq/kWh]"].median()
            df_countries.loc[
                df_countries["Country"] == country, "Main energy source"
            ] = df_country["Main energy source"].mode()[0]
        else:
            df_countries.loc[df_countries["Country"] == country, "count"] = None
            df_countries.loc[
                df_countries["Country"] == country,
                "Median carbon intensity [g CO₂eq/kWh]",
            ] = None
        total += n
    print(f"Total: {total}")

    fig = plot_map(df_countries, args=args)

    # Save figure
    output_fig = output_dir / "map_carbonintensity.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
