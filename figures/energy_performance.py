"""
This script plots the CO₂eq emitted [kg] versus the energy consumed.
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

dict_datasetname = {
    "WMT2014 English-German": "English-German",
    "WMT2014 English-French": "English-French",
}
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
# Markers
dict_markers = {
    "English-German": "o",
    "English-French": "P",
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


def get_pareto_front(data):
    """
    Find the points in the Pareto front
    :param data: An (n_points, n_data) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    Ref:
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_pareto = np.ones(data.shape[0], dtype=bool)
    for idx, el in enumerate(data):
        is_pareto[idx] = np.all(np.any(data[:idx] > el, axis=1)) and np.all(
            np.any(data[idx + 1 :] > el, axis=1)
        )
    return is_pareto


def get_pareto_front_custom(data):
    """
    Find the points in the Pareto front
    :param data: An (n_points, n_data) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    Ref:
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_pareto = np.zeros(data.shape[0], dtype=bool)
    data_nonan = data[~np.any(np.isnan(data), axis=1)]
    for idx, el in enumerate(data):
        is_pareto[idx] = np.all(el[1] >= data_nonan[data_nonan[:, 0] <= el[0], 1])
    return is_pareto


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
    if args.logxaxis:
        plt.rcParams.update({"xtick.bottom": True})

    fig, ax = plt.subplots(
        figsize=(15, 10), dpi=args.dpi, nrows=2, ncols=2, gridspec_kw={"hspace": 0.3}
    )
    sns.scatterplot(
        ax=ax[0][0],
        data=df.loc[(df["Task"] == "Machine Translation")],
        x="Energy consumed [kWh]",
        y="BLEU score",
        hue="Main energy source",
        hue_order=hue_order,
        style="Data set",
        markers=dict_markers,
        palette=palette,
    )
    sns.lineplot(
        ax=ax[0][0],
        data=df.loc[(df["Task"] == "Machine Translation") & (df["is_pareto"])],
        x="Energy consumed [kWh]",
        y="BLEU score",
        hue="Data set",
        palette=[(0, 0, 0), (0, 0, 0)],
    )
    ax[0][0].set_title("Machine Translation")
    sns.scatterplot(
        ax=ax[0][1],
        data=df.loc[(df["Task"] == "Image Classification")],
        x="Energy consumed [kWh]",
        y="Top-1 accuracy",
        hue="Main energy source",
        hue_order=hue_order,
        palette=palette,
    )
    sns.lineplot(
        ax=ax[0][1],
        data=df.loc[(df["Task"] == "Image Classification") & (df["is_pareto"])],
        x="Energy consumed [kWh]",
        y="Top-1 accuracy",
        color=(0, 0, 0),
    )
    ax[0][1].set_title("Image Classification")
    sns.scatterplot(
        ax=ax[1][0],
        data=df.loc[(df["Task"] == "Question Answering")],
        x="Energy consumed [kWh]",
        y="F1 score",
        hue="Main energy source",
        hue_order=hue_order,
        palette=palette,
    )
    sns.lineplot(
        ax=ax[1][0],
        data=df.loc[(df["Task"] == "Question Answering") & (df["is_pareto"])],
        x="Energy consumed [kWh]",
        y="F1 score",
        color="black",
    )
    ax[1][0].set_title("Question Answering")
    sns.scatterplot(
        ax=ax[1][1],
        data=df.loc[(df["Task"] == "Named Entity Recognition")],
        x="Energy consumed [kWh]",
        y="F1 score",
        hue="Main energy source",
        hue_order=hue_order,
        palette=palette,
    )
    sns.lineplot(
        ax=ax[1][1],
        data=df.loc[(df["Task"] == "Named Entity Recognition") & (df["is_pareto"])],
        x="Energy consumed [kWh]",
        y="F1 score",
        color=(0, 0, 0),
    )
    ax[1][1].set_title("Named Entity Recognition")

    if args.logxaxis:
        for axis in ax.flatten():
            axis.set_xscale("log")

    # Change spines
    for axis in ax.flatten():
        sns.despine(ax=axis, left=True, bottom=True)

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
    output_yml = output_dir / "energy_performance.yml"
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col=False)

    # Rename dataset_name
    df.replace({"Data set": dict_datasetname}, regex=False, inplace=True)
    # Year(s)
    df["Year(s)"] = df["Year published"].map(year2years, na_action="ignore")

    # Get Pareto fronts
    df["is_pareto"] = [False for _ in range(len(df))]
    x_col = "Energy consumed [kWh]"
    for task in df["Task"].unique():
        if task == "Machine Translation":
            y_col = "BLEU score"
        elif task == "Image Classification":
            y_col = "Top-1 accuracy"
        elif task == "Question Answering" or task == "Named Entity Recognition":
            y_col = "F1 score"
        else:
            continue
        for dataset in df.loc[df["Task"] == task]["Data set"].unique():
            df_aux = df.loc[(df["Task"] == task) & (df["Data set"] == dataset)]
            df.loc[
                (df["Task"] == task) & (df["Data set"] == dataset), "is_pareto"
            ] = get_pareto_front_custom(df_aux.loc[:, [x_col, y_col]].values)
    fig = plot_data(df, args=args)

    # Save figure
    output_fig = output_dir / "energy_performance.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
