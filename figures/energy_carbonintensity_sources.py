"""
This script plots the carbon intensity emitted [kg] versus the energy consumed.
"""
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

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
        "--logxaxis",
        default=False,
        action="store_true",
        help="Logarithmic X-axis",
    )
    parser.add_argument(
        "--logyaxis",
        default=False,
        action="store_true",
        help="Logarithmic Y-axis",
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


def plot_data(df, args):

    # Set up plot
    sns.set(style="whitegrid")
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
    plt.rcParams.update({"ytick.left": True})
    plt.rcParams.update({"xtick.bottom": True})

    fig, ax = plt.subplots(figsize=(15, 6), dpi=args.dpi)
    sns.scatterplot(
        data=df.loc[(df["CO₂eq emitted [kg]"].isnull() == False)],
        x="Energy consumed [kWh]",
        y="Carbon intensity [g CO₂eq/kWh]",
        hue="Main energy source",
        hue_order=hue_order,
        palette=palette,
    )

    if args.logxaxis:
        ax.set_xscale("log")
    if args.logyaxis:
        ax.set_yscale("log")

    # Ticks
    ax.tick_params(axis="x", which="minor", length=2, color="gray")
    ax.tick_params(axis="x", which="major", length=0)
    ax.tick_params(axis="y", which="minor", length=2, color="gray")
    ax.tick_params(axis="y", which="major", length=0)
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
    output_yml = output_dir / "energy_carbonintensity_sources.yml"
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col=False)

    # Year(s)
    df["Year(s)"] = df["Year published"].map(year2years, na_action="ignore")

    fig = plot_data(df, args=args)

    # Save figure
    output_fig = output_dir / "energy_carbonintensity_sources.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
