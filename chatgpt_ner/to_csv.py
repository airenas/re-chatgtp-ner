import argparse
import json
import sys

import pandas as pd

from chatgpt_ner.logger import logger


def main(argv):
    logger.info("Starting")
    parser = argparse.ArgumentParser(description="Perform Named Entity Recognition (NER) on text from a file.")
    parser.add_argument("--input", nargs='?', required=True, help="Input file to parse")
    parser.add_argument("--name", nargs='?', required=True, help="short name of test")
    args = parser.parse_args(args=argv)

    try:
        # Read the text from the specified file
        logger.info("read file {}".format(args.input))
        with open(args.input, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        data = []
        for author in json_data.get("authors", []):
            insts = author.get("institutions", [])
            if len(insts) == 0:
                data.append([args.name, author.get("name", ""), "", author.get("email")])
            for inst in insts:
                arr = [inst.get("institution", "")]
                if inst.get("address", ""):
                    arr.append(inst.get("address", ""))
                data.append([args.name, author.get("name", ""), ", ".join(arr), author.get("email")])
        df = pd.DataFrame(data, columns=["file", "author", "institution", "email"])

        df.to_csv(sys.stdout, index=False)
        logger.info("done")
    except Exception as e:
        logger.error(f"Error processing file {args.input}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
