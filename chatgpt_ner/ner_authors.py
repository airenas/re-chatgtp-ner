import argparse
import sys

import openai
import os

from chatgpt_ner.logger import logger

openai.api_key = os.getenv('OPENAI_API_KEY')


def main(argv):
    logger.info("Starting")
    parser = argparse.ArgumentParser(description="Perform Named Entity Recognition (NER) on text from a file.")
    parser.add_argument("--input", nargs='?', required=True, help="Input file to parse")
    args = parser.parse_args(args=argv)

    try:
        # Read the text from the specified file
        logger.info("read file {}".format(args.input))
        with open(args.input, 'r', encoding='utf-8') as file:
            text = file.read()

        # Define the prompt for ChatGPT
        file_data = f"```latex\n{text}\n```"

        out_format = """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "authors": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "email": {
            "type": "string"
          },
          "institutions": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "institution": {
                  "type": "string"
                },
                "address": {
                  "type": "string"
                }
              },
              "required": ["institution"]
            }
          }
        },
        "required": ["name", "institutions"]
      }
    }
  },
  "required": ["authors"]
}"""

        # Make the API call to ChatGPT
        logger.info("call openai ...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": "You are a NER tool. You will be provided text in a latex format. It is a first page of scientific article. "
                                  "Your task is to extract authors of the article. For each author extract an email, an institution, and address."},
                      {"role": "system", "content": "Please extract the text as it is. Do not change/update/fix text, like adding punctuation or etc."},
                      {"role": "system", "content": "The output must be a valid json matching the schema"},
                      {"role": "system", "content": out_format},
                      {"role": "user", "content": file_data},
                      ],
            temperature=0.2,
            max_tokens=512
        )
        logger.info("got resp")
        # Extract and print the model's response
        ner_result = response['choices'][0]['message']['content']
        print(f"\n{ner_result}\n")
        logger.info("done")
    except Exception as e:
        logger.error(f"Error processing file {args.input}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
