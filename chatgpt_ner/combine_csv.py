import argparse
import sys

import pandas as pd

from chatgpt_ner.logger import logger


def combine_csv_files(input_files):
    logger.info("start combine csvs")
    df = pd.concat((pd.read_csv(f) for f in input_files), ignore_index=True)
    df.to_csv(sys.stdout, index=False)
    logger.info("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple CSV files into one.")
    parser.add_argument("input_files", nargs="+", help="List of input CSV files")
    args = parser.parse_args()

    combine_csv_files(args.input_files)
