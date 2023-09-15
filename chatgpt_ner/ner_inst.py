import argparse
import json
import os
import sys

import openai
import pandas as pd

from chatgpt_ner.logger import logger

openai.api_key = os.getenv('OPENAI_API_KEY')


def main(argv):
    logger.info("Starting")
    parser = argparse.ArgumentParser(description="Perform Named Entity Recognition (NER) on institution.")
    parser.add_argument("--input", nargs='?', required=True, help="Input file to parse")
    parser.add_argument("--cache", nargs='?', required=True, help="Cache file")
    args = parser.parse_args(args=argv)
    logger.info("File       : {}".format(args.input))
    logger.info("Cache      : {}".format(args.cache))
    cache, json_data = {}, []
    if os.path.exists(args.cache):
        with open(args.cache, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
    for j in json_data:
        cache[j["string"]] = j
    gt = pd.read_csv(args.input, sep=',')
    institutions = gt["institution"].dropna().unique()

    for inst in institutions:
        logger.info("extract '{}'".format(inst))
        if inst in cache:
            logger.info("skip, found in cache")
            continue
        res, ok = extract(inst)
        if ok:
            res["string"] = inst
            json_data.append(res)

    json_string = json.dumps(json_data, indent=4)
    print(json_string)


def extract(inst):
    logger.info("Starting")
    try:
        input = f"Line: {inst}"
        out_format = """
        {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "institutionName": {
      "type": "string"
    },
    "addressLine": {
      "type": "string"
    },
    "city": {
      "type": "string"
    },
    "postCode": {
      "type": "string"
    },
    "country": {
      "type": "string"
    }
  },
  "required": ["organizationName"]
}"""

        # Make the API call to ChatGPT
        logger.info("call openai ...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": "You are a NER tool. You will be provided with one line of a text containing an institutional affiliation for the author. "
                                  "Your task is to split the line into \n - institution name(s) (separated by comma if several) \n - adresss line,\n - post code,\n - city, \n - and country.\n Some fields may be missing except institution name."},
                      {"role": "system",
                       "content": "Please extract the text as it is. Do not change/update/fix text, like adding punctuation or etc."},
                      {"role": "system",
                       "content": "Do not duplicate information. Just one exception: if you see that the country is emty and you think that the country could be determined from a city or from an institution name then, please, fill it"},
                      {"role": "system",
                       "content": "All provided text must be packed into the output. Nothing could be skipped or ignored!."},                       
                      {"role": "system", "content": "The output must be a valid json matching the schema:"},
                      {"role": "system", "content": out_format},
                      {"role": "system",
                       "content": "Do not fill information in the adress line field if there is the same city and/or postal code."},
                      {"role": "user", "content": input},
                      ],
            temperature=0.2,
            max_tokens=256
        )
        logger.info("got resp")
        # Extract and print the model's response
        ner_result = response['choices'][0]['message']['content']
        logger.debug(f"\n{ner_result}\n")
        logger.info("done")
        res = json.loads(ner_result)
        return res, True
    except Exception as e:
        logger.error(f"Error processing line '{inst}': {str(e)}")
    return "", False


if __name__ == "__main__":
    main(sys.argv[1:])
