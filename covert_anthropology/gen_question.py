import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion
import logging
import time
import json
import openai
import sys
import re

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    filename=f'gen_ancient_running_{time.strftime("%d_%H_%M_%S")}.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Starting the script...")

#############################################
# NEW: Function to split plain text into sections
#############################################
def split_text_into_sections(text, max_words=500):
    """
    Splits the input plain text into sections of approximately max_words.
    The split ensures that each section ends at the end of a sentence,
    determined by punctuation (., !, or ?).

    Returns:
        List of text sections.
    """
    # Split text into sentences (using regex to catch end-of-sentence punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sections = []
    current_section = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        # If adding this sentence exceeds the limit and the current section is not empty, finalize the section.
        if current_word_count + len(words) > max_words and current_section:
            section = " ".join(current_section).strip()
            sections.append(section)
            current_section = []
            current_word_count = 0
        current_section.append(sentence)
        current_word_count += len(words)

    if current_section:
        sections.append(" ".join(current_section).strip())

    logger.info(f"Split text into {len(sections)} sections.")
    return sections

#############################################
# UPDATED: Revised create_prompt function
#############################################
def create_prompt(raw_text, question_type, question_count=5):
    """
    Build the GPT-4 prompt based on the text section and question type.
    
    For each question, the model is asked to provide:
      1. The Question
      2. The Correct Answer
      3. A Detailed Explanation (self-contained without referring back to the section)

    **Output Requirements:**
      - The output must be valid JSON.
      - The JSON should be an array of objects.
      - Each object must contain exactly these three keys: "question", "answer", and "explanation".
      - No extra commentary is allowed outside the JSON.

    Raw Text Section:
    {raw_text}
    """
    # Map internal type to a user-friendly name.
    type_mapping = {
        "single-choice": "Single Choice (SC)",
        "multiple-choice": "Multiple Choice (MC)",
        "fill-in-the-blank": "Fill in the Blank (FB)",
        "judgment": "Judgment (True/False) (J)",
        "short-answer": "Short Answer (SA)"
    }
    question_type_name = type_mapping.get(question_type, question_type)
    
    prompt = f'''
You have the following raw text.
Please analyze it carefully and generate {question_count} questions of type "{question_type_name}."

For each question, provide:
1. The Question
2. The Correct Answer
3. A Detailed Explanation of this Answer.
The explanation should not refer to the raw text or mention "the text says" or "based on the given text".
It should be a self-contained explanation that provides general knowledge or reasoning for why the answer is correct.

**Output Requirements:**
- The output must be valid JSON.
- The JSON should be an array of objects.
- Each object must contain exactly these three keys: "question", "answer", and "explanation".
- Do not include any extra commentary or text outside the JSON.
- Example output format:
```
[
    {{
        "question": "x1",
        "answer": "x1",
        "explanation": "x1"
    }},
    {{
        "question": "x2",
        "answer": "x2",
        "explanation": "x2"
    }}
]
```

Raw Text:
{raw_text}
'''
    return prompt



def construct_type1_message_for_txt(section_text, question_type):
    """
    Constructs the message structure for a given text section.
    Always generates a fixed number (5) of questions per section.
    """
    question_count = 5
    input_text = create_prompt(raw_text=section_text, question_type=question_type, question_count=question_count)
    s_prompt = (
        "You are an assistant specializing in generating educational questions. "
        "For each question, provide a detailed, self-contained explanation without referring back to the original section."
    )
    messages = [
        {"role": "system", "content": s_prompt},
        {"role": "user", "content": input_text}
    ]
    return messages, question_count

#############################################
# Adjusted processing for text input
#############################################
def process_text_section(section_text, q_type, api_base, model_name):
    """
    Process a single text section for a given question type.
    """
    try:
        messages, question_count = construct_type1_message_for_txt(
            section_text=section_text,
            question_type=q_type
        )
        logger.info(f"Processing text section for type '{q_type}' with expected {question_count} QA pairs...")
        qa_result = get_validated_completion(
            api_base, model_name, messages,
            expected_count=question_count, 
            max_tokens=4096, temperature=0.7, max_retries=50
        )
    except Exception as e:
        qa_result = f"[Error calling LLM] {str(e)}"
        logger.error(f"Failed to process text section for type '{q_type}': {str(e)}")
    
    logger.info(f"Completed processing text section for type '{q_type}'.")
    return {"qa": qa_result, "input": messages[1]["content"]}

import re

def extract_json(text):
    # Remove any leading/trailing whitespace
    text = text.strip()
    # Check if the text starts with [ or {, if not try to extract a JSON block
    if not (text.startswith('[') or text.startswith('{')):
        # Attempt to extract JSON block using regex: find the first '[' and last ']'
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            text = text[start:end+1]
    return text

def split_generated_output(text, expected_count):
    try:
        json_text = extract_json(text)
        qa_pairs = json.loads(json_text)
        if (isinstance(qa_pairs, list) and
            all(isinstance(item, dict) and {"question", "answer", "explanation"} <= set(item.keys()) 
                for item in qa_pairs)):
            if len(qa_pairs) >= expected_count:
                return qa_pairs
    except Exception as e:
        print("JSON parsing error:", e)
        # Proceed with regex fallback below if JSON parsing fails
        pass

    # Fallback: Use regex to capture blocks that start with "Question:", "Answer:", and "Explanation:".
    pattern = (
        r"Question\s*\d*:\s*(?P<question>.+?)\s*"
        r"Answer\s*\d*:\s*(?P<answer>.+?)\s*"
        r"Explanation\s*\d*:\s*(?P<explanation>.+?)(?=(?:Question\s*\d*:)|$)"
    )
    matches = re.findall(pattern, text, flags=re.DOTALL)
    results = []
    for match in matches:
        q, a, exp = match
        results.append({
            "question": q.strip(),
            "answer": a.strip(),
            "explanation": exp.strip()
        })
    return results

def get_validated_completion(api_base, model_name, messages, expected_count, max_tokens=4096, temperature=0.7, max_retries=5):
    """
    Calls the chat_completion API and verifies that the returned text contains
    at least the expected number of QA pairs.
    
    If the number of parsed QA pairs is insufficient, the function retries up to
    max_retries times. If after the retries the output still does not meet the
    expected count, it returns whatever was generated (with a warning) or raises an error.
    """
    retries = 0
    while retries < max_retries:
        response_text = chat_completion(api_base, model_name, messages, max_tokens=max_tokens, temperature=temperature)
        qa_pairs = split_generated_output(response_text, expected_count)
        if len(qa_pairs) >= expected_count:
            logger.info(f"Generated {len(qa_pairs)} QA pairs as expected.")
            return qa_pairs
        else:
            retries += 1
            logger.warning(f"Insufficient QA pairs generated ({len(qa_pairs)}/{expected_count}). Retrying {retries}/{max_retries}...")
            logger.warning(json.loads(response_text))
    if qa_pairs:
        logger.warning(f"Returning {len(qa_pairs)} QA pairs after {max_retries} retries, which is less than the expected {expected_count}.")
        return qa_pairs
    else:
        raise Exception("Failed to generate sufficient QA pairs after maximum retries.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset being processed.")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model for vLLM.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for vLLM.")
    parser.add_argument("--output_folder_path", type=str, required=True, help="Folder name for the output.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs for tensor parallel.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the vLLM server.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input TXT file.")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for concurrent processing.")
    parser.add_argument(
        "--question_type",
        type=str,
        default="single-choice",
        choices=[
            "all",
            "single-choice",
            "multiple-choice",
            "fill-in-the-blank",
            "judgment",
            "short-answer"
        ],
        help="Type of questions to generate. Use 'all' to generate for all types."
    )
    parser.add_argument("--host_vllm", action="store_true")
    args = parser.parse_args()

    # Check if the output folder exists; if not, create it.
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)

    # Read the entire text from the input file.
    with open(args.input_file, "r") as f:
        full_text = f.read()

    # Split the text into sections of approximately 500 words.
    sections = split_text_into_sections(full_text, max_words=500)

    # Define the list of question types to process.
    if args.question_type == "all":
        types_to_generate = ["single-choice", "multiple-choice", "fill-in-the-blank", "judgment", "short-answer"]
    else:
        types_to_generate = [args.question_type]

    api_base = f"http://localhost:{args.port}/v1"

    # Optionally, start the vLLM server.
    if args.host_vllm:
        process = start_vllm_server(args.model, args.model_name, args.port, args.gpu)

    # Process each section for each question type.
    for q_type in types_to_generate:
        output_file = os.path.join(args.output_folder_path, f'text_sections_{q_type}.jsonl')
        output_data = []

        # Process each text section.
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [
                executor.submit(process_text_section, section, q_type, api_base, args.model_name)
                for section in sections
            ]
            for i, future in enumerate(futures, start=1):
                result = future.result()
                # Include section index for traceability.
                result["section_index"] = i
                output_data.append(result)
                if i % 100 == 0:
                    write_jsonl(output_file, output_data, append=True)
                    output_data.clear()
        # Save any remaining results.
        if output_data:
            write_jsonl(output_file, output_data, append=True)
        logger.info(f"Output saved to {output_file} for type '{q_type}'.")

    # Stop the vLLM server if it was started.
    if args.host_vllm:
        stop_vllm_server(process)

if __name__ == "__main__":
    main()
