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
    filename=f'gen_law_running_{time.strftime("%d_%H_%M_%S")}.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Starting the script...")

def transform_to_readable_text(data_obj):
    """
    Convert a single JSON object into a readable text block for GPT-4 prompt usage.
    The data_obj is expected to contain keys like:
      - "book_name"
      - "part_title"
      - "part_subtitle"
      - "chapter_title"
      - "chapter_subtitle"
      - "section_number"
      - "section_title"
      - "bibliographic"
      - "sidenote"
      - "content" (a list of paragraphs)
    """
    book_name        = data_obj.get("book_name", "")
    part_title       = data_obj.get("part_title", "")
    part_subtitle    = data_obj.get("part_subtitle", "")
    chapter_title    = data_obj.get("chapter_title", "")
    chapter_subtitle = data_obj.get("chapter_subtitle", "")
    section_number   = data_obj.get("section_number", "")
    section_title    = data_obj.get("section_title", "")
    # bibliographic    = data_obj.get("bibliographic", "")
    sidenote         = data_obj.get("sidenote", "")
    content_list     = data_obj.get("content", [])

    text_block = []
    text_block.append(f"Book Name: {book_name}")
    text_block.append(f"Part Title: {part_title}")
    text_block.append(f"Part Subtitle: {part_subtitle}")
    text_block.append(f"Chapter Title: {chapter_title}")
    text_block.append(f"Chapter Subtitle: {chapter_subtitle}")
    text_block.append(f"Section Number: {section_number}")
    text_block.append(f"Section Title: {section_title}")
    # text_block.append(f"Bibliographic: {bibliographic}")
    text_block.append(f"Sidenote: {sidenote}")
    text_block.append("\nContent:\n")

    for paragraph in content_list:
        text_block.append(paragraph)
        text_block.append("")  # blank line between paragraphs
    
    combine_txt = "\n".join(text_block)
    word_count = len(combine_txt.split())
    logger.info(f"Word count of text block: {word_count}")
    return combine_txt

def create_prompt(raw_text, question_type, question_count):
    """
    Build the GPT-4 prompt based on the question type and question count.
    
    For each question, the model is asked to provide:
      1. The Question
      2. The Correct Answer
      3. A Detailed Explanation (self-contained without referring back to the text)
    
    **Output Requirements:**
      - The output must be valid JSON.
      - The JSON should be an array of objects.
      - Each object must contain exactly these three keys: "question", "answer", and "explanation".
      - Do not include any extra commentary or text outside the JSON.
    
    Raw Text:
    {raw_text}
    """
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

Ensure that your explanation provides a detailed elaboration.
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

def construct_prompt():
    """
    Returns the system prompt with instructions to avoid referring to the original text.
    """
    return (
        "You are an assistant specializing in generating educational questions. "
        "For each question, provide a detailed explanation. The explanation should be self-contained, "
        "concise, and not refer to the original text."
    )

def construct_type1_message(dataset_type, origin_dict, question_type):
    """
    Constructs the message structure for the LLM call.
    
    It converts the input JSON into a readable text and then determines the expected
    number of questions based on the text length:
      - If word count < 500, then generate 5 questions.
      - Otherwise, generate 10 questions.
    
    Returns both the messages (as a list of dicts) and the computed question count.
    """
    raw_text = transform_to_readable_text(origin_dict)
    word_count = len(raw_text.split())
    if word_count < 500:
        question_count = 5
    else:
        question_count = 10
    
    input_text = create_prompt(raw_text=raw_text, question_type=question_type, question_count=question_count)
    s_prompt = construct_prompt()
    messages = [
        {"role": "system", "content": s_prompt},
        {"role": "user", "content": input_text}
    ]
    return messages, question_count


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

def process_record(record, q_type, api_base, model_name, dataset_type):
    """
    Process a single record for a given question type.
    """
    origin_dict = record
    idx = record.get("idx", None)
    try:
        messages, question_count = construct_type1_message(
            dataset_type=dataset_type,
            origin_dict=origin_dict,
            question_type=q_type
        )
        logger.info(f"[INFO] Processing record {idx} for type '{q_type}' with expected {question_count} QA pairs...")
        qa_result = get_validated_completion(
            api_base, model_name, messages,
            expected_count=question_count, max_tokens=8096 if question_count >5 else 4096, temperature=0.7, max_retries=50
        )
    except Exception as e:
        qa_result = f"[Error calling LLM] {str(e)}"
        logger.error(f"[ERROR] Failed to process record {idx} for type '{q_type}': {str(e)}")
    
    logger.info(f"[INFO] Completed record {idx} for type '{q_type}'.")
    return {"idx": idx, "qa": qa_result, "origin_dict": origin_dict, "input": messages[1]["content"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset being processed.")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model for vLLM.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for vLLM.")
    parser.add_argument("--output_folder_path", type=str, required=True, help="Folder name for the output.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs for tensor parallel.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the vLLM server.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
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
    parser.add_argument(
        "--question_count",
        type=int,
        default=5,
        help="Default number of questions to generate if not determined by text length."
    )
    
    parser.add_argument("--host_vllm", default=False)
    args = parser.parse_args()
    
    # Check if the output folder exists; if not, create it.
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)
    
    # Define the list of question types to process.
    if args.question_type == "all":
        types_to_generate = ["fill-in-the-blank", "judgment", "short-answer"]
    else:
        types_to_generate = [args.question_type]
    
    # Read input data.
    data_list = list(read_jsonl(args.input_jsonl))
    
    api_base = f"http://localhost:{args.port}/v1"
    
    # Start vLLM server for the chosen model.
    if args.host_vllm:
        process = start_vllm_server(args.model, args.model_name, args.port, args.gpu)
    
    # Process for each question type separately.
    for q_type in types_to_generate:
        # Create an output filename with the question type appended.
        input_basename = os.path.basename(args.input_jsonl).replace(".jsonl", "")
        output_file = os.path.join(args.output_folder_path, f'type1_{input_basename}_{q_type}.jsonl')
        
        # Load existing results (if any) to avoid reprocessing records.
        if os.path.exists(output_file):
            logger.info(f"[INFO] Loading existing results from {output_file} for type '{q_type}'")
            existing_results = list(read_jsonl(output_file))
            existing_ids = {record["idx"] for record in existing_results}
            records_to_process = [record for record in data_list if record.get("idx") not in existing_ids]
            logger.info(f"[INFO] {len(records_to_process)} new records will be processed for type '{q_type}'.")
        else:
            logger.info(f"[INFO] No existing results found for type '{q_type}'. Processing all records.")
            records_to_process = data_list[:]
        
        output_data = []
        
        def save_partial_results():
            if output_data:
                write_jsonl(output_file, output_data, append=True)
                output_data.clear()
        
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_record, record, q_type, api_base, args.model_name, args.dataset_type)
                       for record in records_to_process]
            pre_time = time.time()
            for i, future in enumerate(futures, start=1):
                output_data.append(future.result())
                # Save intermediate results every 2000 records.
                if i % 200 == 0:
                    current_time = time.time()
                    save_partial_results()
                    logger.info(f"[INFO] Processed {i} records for type '{q_type}' in {current_time - pre_time:.2f}s.")
                    pre_time = current_time
        
        # Save any remaining records for the current question type.
        save_partial_results()
        logger.info(f"[INFO] Output saved to {output_file} for type '{q_type}'.")
    
    # Stop the vLLM server.
    if args.host_vllm:
        stop_vllm_server(process)

if __name__ == "__main__":
    main()
