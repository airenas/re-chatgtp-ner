import argparse
import openai
import os

# Your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI API client
openai.api_key = api_key


def perform_ner_on_file(file_path):
    try:
        # Read the text from the specified file
        with open(file_path, 'r', encoding='utf-8') as file:
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": "You are a NER tool. You will be provided text in a latex format. It is a first page of scientific article. "
                                  "Your task is to extract authors of the article. For each author extract an institution and address."},
                      {"role": "system", "content": "The output must be a valid json matching the schema"},
                      {"role": "system", "content": out_format},
                      {"role": "user", "content": file_data},
                      ],
            temperature=0.2,
            max_tokens=512
        )

        # Extract and print the model's response
        ner_result = response['choices'][0]['message']['content']
        print(f"\n{ner_result}\n")
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Named Entity Recognition (NER) on text from a file.")
    parser.add_argument("file_path", type=str, help="Path to the text file to process.")
    args = parser.parse_args()

    perform_ner_on_file(args.file_path)
