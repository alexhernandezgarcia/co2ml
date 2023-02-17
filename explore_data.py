"""
Explore data
"""
from argparse import ArgumentParser

import pandas as pd


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    args2config = {}
    parser.add_argument(
        "-y",
        "--yaml_config",
        default=None,
        required=False,
        type=str,
        help="Configuration file of the run",
    )
    args2config.update({"yaml_config": ["yaml_config"]})
    parser.add_argument(
        "--data_csv",
        default=None,
        type=str,
        help="Path to CSV file containing data",
    )
    args2config.update({"data_csv": ["data_csv"]})
    return parser, args2config


def main(args):
    df = pd.read_csv(args.data_csv, index_col=False)
    n_data = len(df)

    print("Completeness of data")
    keys = [
        "CO₂eq emitted [kg]",
        "Energy consumed [kWh]",
        "Carbon intensity [g CO₂eq/kWh]",
        "Number of GPUs",
        "Number of TPUs",
        "Training time [s]",
    ]
    for k in keys:
        print(f"\t{k} available in {n_data - df[k].isnull().sum()}/{n_data}")

    print("Energy sources")
    for source in df["Main energy source"].unique():
        df_source = df.loc[df["Main energy source"] == source]
        n_papers_source = len(df_source)
        intensity = df_source["Carbon intensity [g CO₂eq/kWh]"]
        print(
            "{}: {:d} models | {:d} papers | Mean: {:.2f} | Median: {:.2f} | "
            "Min: {:.2f} | Max: {:.2f}".format(
                source,
                len(df_source),
                n_papers_source,
                intensity.mean(),
                intensity.median(),
                intensity.min(),
                intensity.max(),
            )
        )
    print("Year published")
    for year in df["Year published"].unique():
        df_year = df.loc[df["Year published"] == year]
        n_papers_year = len(df_year.index)
        intensity = df_year["Carbon intensity [g CO₂eq/kWh]"]
        print(
            "{}: {:d} models | {:d} papers | Mean: {:.2f} | Median: {:.2f}".format(
                year,
                len(df_year),
                n_papers_year,
                intensity.mean(),
                intensity.median(),
            )
        )
    print("Training time (all GPUs/TPUs) [s]")
    training_time = df["Training time [s]"]
    gpu_time = df["Number of GPUs"] * training_time
    tpu_time = df["Number of TPUs"] * training_time
    hardware_time = gpu_time
    hardware_time[hardware_time.isna()] = tpu_time[hardware_time.isna()]
    print(
        "Mean: {:.2f} | Median: {:.2f} | Std.: {:.2f} | Min: {:.2f} | "
        "Max: {:.2f}".format(
            hardware_time.mean(),
            hardware_time.median(),
            hardware_time.std(),
            hardware_time.min(),
            hardware_time.max(),
        )
    )
    print("Total emissions")
    co2_emitted = df["CO₂eq emitted [kg]"]
    print("Total CO₂ emitted: {:.2f} kg".format(co2_emitted.sum()))


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)
    args = parser.parse_args()
    main(args)
