"""
This script plots the energy used by each model in the data set, disaggregating
per task and year.
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
hue_order = [
    "Image Classification",
    "Object Detection",
    "Machine Translation",
    "Question Answering",
    "Named Entity Recognition",
]
# Markers
dict_markers = {
    "Image Classification": "o",
    "Object Detection": "s",
    "Machine Translation": "D",
    "Question Answering": "^",
    "Named Entity Recognition": "v",
}
# Colors
palette_colorblind = sns.color_palette("colorblind")
palette = {
    "Image Classification": palette_colorblind[0],
    "Object Detection": palette_colorblind[1],
    "Machine Translation": palette_colorblind[6],
    "Question Answering": palette_colorblind[8],
    "Named Entity Recognition": palette_colorblind[2],
}


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
        default="../data/all_merged_20211222.csv",
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
        "--estimator",
        default="mean",
        type=str,
        help="Estimator of central tendency: [mean, median, trim_mean99, trim_mean95]",
    )
    parser.add_argument(
        "--dpi",
        default=200,
        type=int,
        help="DPI for the output images",
    )
    parser.add_argument(
        "--n_bs",
        default=1e6,
        type=int,
        help="Number of bootrstrap samples",
    )
    parser.add_argument(
        "--bs_seed",
        default=2021,
        type=int,
        help="Bootstrap random seed",
    )
    parser.add_argument(
        "--ci",
        default=95,
        type=int,
        help="Confidence level for the bootstrapped confidence intervals",
    )
    parser.add_argument(
        "--dodge",
        default=0.5,
        type=float,
        help="Space between tasks",
    )
    parser.add_argument(
        "--conf",
        default=0.99,
        type=float,
        help="Confidence level",
    )
    parser.add_argument(
        "--errwidth",
        default=1.5,
        type=float,
        help="Thickness of error bar lines",
    )
    parser.add_argument(
        "--capsize",
        default=0.03,
        type=float,
        help="Width of the caps on error bars",
    )
    parser.add_argument(
        "--join",
        default=False,
        action="store_true",
        help="Whether to join each task column with a straight line",
    )
    parser.add_argument(
        "--no_stripplot",
        default=False,
        action="store_true",
        help="To skip strip plot (dots of each data point)",
    )
    parser.add_argument(
        "--stripsize",
        default=3.5,
        type=float,
        help="Size of the dots in the stripplot",
    )
    parser.add_argument(
        "--stripalpha",
        default=0.5,
        type=float,
        help="Transparency of the dots in the stripplot",
    )
    parser.add_argument(
        "--logaxis",
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


def plot_data(df, estimator, args):

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

    fig, ax = plt.subplots(figsize=(15, 6), dpi=args.dpi)
    sns.pointplot(
        ax=ax,
        data=df,
        x="Year(s)",
        y="Energy consumed [kWh]",
        hue="Task",
        hue_order=hue_order,
        markers=[dict_markers[k] for k in hue_order],
        dodge=args.dodge,
        order=sorted(df["Year(s)"].unique()),
        estimator=estimator,
        ci=args.ci,
        seed=args.bs_seed,
        n_boot=args.n_bs,
        join=args.join,
        errwidth=args.errwidth,
        capsize=args.capsize,
        palette=palette,
    )
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    if not args.no_stripplot:
        sns.stripplot(
            ax=ax,
            data=df,
            x="Year(s)",
            y="Energy consumed [kWh]",
            hue="Task",
            hue_order=hue_order,
            dodge=args.dodge,
            order=sorted(df["Year(s)"].unique()),
            size=args.stripsize,
            alpha=args.stripalpha,
            palette=palette,
        )
        ax.get_legend().remove()
        ax.legend(handles=leg_handles, labels=leg_labels)
    if args.logaxis:
        ax.set_yscale("log")

    # Y ticks
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

    # Determine estimator
    if args.estimator == "mean":
        estimator = np.mean
    elif args.estimator == "median":
        estimator = np.median
    elif args.estimator == "trim_mean95":
        estimator = partial(trim_mean, proportiontocut=0.05)
    elif args.estimator == "trim_mean99":
        estimator = partial(trim_mean, proportiontocut=0.01)
    else:
        raise NotImplementedError()

    # Store args
    output_yml = output_dir / "energy_task_year.yml"
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col=False)

    # Year(s)
    df["Year(s)"] = df["Year published"].map(year2years, na_action="ignore")

    fig = plot_data(df, estimator=estimator, args=args)

    # Save figure
    output_fig = output_dir / "energy_task_year.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
