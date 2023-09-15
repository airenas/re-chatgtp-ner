import argparse
import json
import sys

import pandas as pd

from chatgpt_ner.logger import logger


def main(argv):
    logger.info("Starting")
    parser = argparse.ArgumentParser(description="Change JSON to CSV.")
    parser.add_argument("--input", nargs='?', required=True, help="Input file to parse")
    args = parser.parse_args(args=argv)

    try:
        # Read the text from the specified file
        logger.info("read file {}".format(args.input))
        with open(args.input, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        data = []
        for inst in json_data:
            data.append([inst.get("string", ""), inst.get("institutionName"), inst.get("addressLine", ""),
                         inst.get("city", ""), inst.get("postCode", ""), inst.get("country", ""), inst.get("state", "")])
        df = pd.DataFrame(data, columns=["name", "institutionName", "addressLine", "city", "postCode", "country","state"])

        df.to_csv(sys.stdout, index=False)
        logger.info("done")
    except Exception as e:
        logger.error(f"Error processing file {args.input}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
