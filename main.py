from google import genai
from dotenv import load_dotenv
import os
from pathlib import Path
from PIL import Image
import re
import time
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, CreateCachedContentConfig
import pandas as pd
import json
import sys
import argparse
import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
import yaml

# Create a thread pool for CPU-bound tasks like image loading
thread_pool = ThreadPoolExecutor(max_workers=10)

async def get_image(img_file: str) -> Image.Image:

    loop = asyncio.get_running_loop()
    try:
        img = await loop.run_in_executor(thread_pool, lambda: Image.open(img_file))

        # Ensure image is loaded before manipulating
        img.load()

        # Resize image so that longest side is 768 pixels, not more than that
        w, h = img.size
        ar = w/h
        max_side = max(w, h)
        # Only resize if longest side is more than 768
        if max_side > 768:
            scale = 768 / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            try:
                # Try the newer approach first
                img = await loop.run_in_executor(
                    thread_pool, 
                    lambda: img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                )
            except AttributeError:
                img = await loop.run_in_executor(
                    thread_pool, 
                    lambda: img.resize((new_w, new_h), Image.LANCZOS)
                )
        new_w, new_h = img.size
        nar = new_w/new_h
        print(f"New width: {new_w}, new height: {new_h}, old aspect ratio : {ar}, new aspect ratio: {nar}")

        return img
    except Exception as e:
        print(f"Error fetching {img_file}: {e}")
        return None  # type: ignore

async def gen_response(api: str, img: Image.Image, prompt_text: str, search_tool:bool) -> tuple[dict, float, Any, Any, str]:
    """
    Generate AI response using a thread pool to handle the synchronous API call.
    This allows multiple API calls to run concurrently without blocking the event loop.
    """
    client = genai.Client(api_key=api)
    model_id = "gemini-2.0-flash"

    google_search_tool = Tool(
        google_search=GoogleSearch(),
    )

    start_time = time.time()
    loop = asyncio.get_running_loop()
    
    # Initialize variables to avoid reference errors
    input_token_count = 0
    output_token_count = 0
    search_tool_used = "No"

    try:
        if search_tool:
            response = await loop.run_in_executor(
                thread_pool,
                lambda: client.models.generate_content(
                    model=model_id,
                    contents=[img, prompt_text],
                    config=GenerateContentConfig(
                        temperature=0.0,
                        seed=42,
                        tools=[google_search_tool],
                        response_modalities=["TEXT"],
                    )
                )
            )
        else:
            response = await loop.run_in_executor(
                thread_pool,
                lambda: client.models.generate_content(
                    model=model_id,
                    contents=[img, prompt_text],
                    config=GenerateContentConfig(
                        temperature=0.0,
                        seed=42
                    )
                )
            )
        
        # Check if response indicates an error
        if hasattr(response, 'error') and response.error:
            raise Exception(f"{response.error.get('code', 'Unknown')} {response.error.get('status', 'ERROR')}: {response.error.get('message', 'Unknown error')}")

        end = time.time() - start_time
        
        if response.usage_metadata:
            output_token_count = response.usage_metadata.candidates_token_count
            input_token_count = response.usage_metadata.prompt_token_count
        else:
            output_token_count = 0
            input_token_count = 0

        if response.candidates and response.candidates[0].grounding_metadata:
            grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
        else:
            grounding_chunks = None

        if grounding_chunks is None:
            search_tool_used = "No"
        elif grounding_chunks is not None:
            search_tool_used = "Yes"
            
        print(f"Grounding google Search tool used or not: {search_tool_used}")
        # print(response)
        
        if response.text:
            raw_response = response.text.strip()
            
            # More comprehensive JSON cleaning
            cleaned_json = re.sub(r'```(?:json|JSON)?\s*\n?', '', raw_response)
            cleaned_json = re.sub(r'\n?```\s*$', '', cleaned_json)
            
            # Extract JSON from response (find first { to last })
            start_brace = cleaned_json.find('{')
            end_brace = cleaned_json.rfind('}')
            
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                cleaned_json = cleaned_json[start_brace:end_brace+1]
            
            # FIX: Replace incorrect boolean values
            cleaned_json = re.sub(r'\bTRUE\b', 'true', cleaned_json)
            cleaned_json = re.sub(r'\bFALSE\b', 'false', cleaned_json)
            
            # Also handle other common JSON issues
            cleaned_json = re.sub(r',\s*}', '}', cleaned_json)  # Remove trailing commas before }
            cleaned_json = re.sub(r',\s*]', ']', cleaned_json)  # Remove trailing commas before ]
            
            try:
                parsed_json = json.loads(cleaned_json)
                return parsed_json, end, input_token_count, output_token_count, search_tool_used
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for image: {e}")
                print(f"Raw response: {raw_response[:500]}...")
                print(f"Cleaned JSON attempt: {cleaned_json[:500]}...")
                
                # Additional debugging: print the problematic section
                error_pos = getattr(e, 'pos', 0)
                if error_pos > 0:
                    start_debug = max(0, error_pos - 50)
                    end_debug = min(len(cleaned_json), error_pos + 50)
                    print(f"Error around position {error_pos}: '{cleaned_json[start_debug:end_debug]}'")
                
                return {"Data": {"title": "", "description": "", "tags": [], "fields": []}}, round(end, 2), input_token_count, output_token_count, search_tool_used
        else:
            return {"Data": {"title": "", "description": "", "tags": [], "fields": []}}, round(end, 2), input_token_count, output_token_count, search_tool_used
            
    except Exception as e:
        error_message = str(e)
        print(f"Error generating response: {e}")
        
        # Check if it's a retryable error (503, overloaded, etc.)
        if any(keyword in error_message.lower() for keyword in ["503", "overloaded", "unavailable", "rate limit"]):
            # Re-raise retryable errors so the retry logic can handle them
            raise e
        else:
            # For non-retryable errors, return empty result
            return {"Data": {"title": "", "description": "", "tags": [], "fields": []}}, round(time.time() - start_time, 2), input_token_count, output_token_count, search_tool_used

def get_next_filename(dir: str) -> str:
    existing_files = glob.glob(os.path.join(dir, "output_*.csv"))  
    next_num = len(existing_files) + 1  
    return os.path.join(dir, f"output_{next_num:05d}.csv")

def remove_base_category(response_dict: dict, base_cat: list)-> dict:
    print(f"Base categories received: \n{base_cat}")

    if "Data" in response_dict and "tags" in response_dict["Data"]:
        # Filter out tags where tagValue is in base_cat list
        response_dict["Data"]["tags"] = [
            tag for tag in response_dict["Data"]["tags"] 
            if tag.get("tagValue") not in base_cat
        ]
    elif "tags" in response_dict:
        # Direct tags structure
        response_dict["tags"] = [
            tag for tag in response_dict["tags"] 
            if tag.get("tagValue") not in base_cat
        ]
    
    return response_dict

async def process_img(img_name: str, img_dir: str, prompt_text: str, api_key: str, yaml_data_list: str, data_list: dict, search_tool_usage: bool, base_categories: list) -> Dict[str, Any]:
    """Process a single image with the AI model"""
    img_file = os.path.join(img_dir, img_name)
    image = await get_image(img_file)


    if image is None:
        return {
            "Image": img_name,
            "Data List": data_list,
            "Title": "",
            "Description": "",
            "Tags": "",
            "Fields": f"Error: Could not load image",
            "Time": 0.0,
            "Input Token Count": "",
            "Output Token Count": "",
            "Search Tool Used": "",
            "Json Response": ""
        }

    try:
        full_prompt = prompt_text + f'\nThis is the data list. : "Data List": {yaml_data_list}'

        # Generate Response
        response_data, time_taken, input_tokens, output_tokens, search_tool = await gen_response(api_key, image, full_prompt, search_tool_usage)

        new_response = remove_base_category(response_data, base_categories
                                            )
        print(f"Response: {new_response}\n")

    

        print(f"Input Tokens: {input_tokens}")
        print(f"Output Tokens: {output_tokens}")

        print(f"Processed: {img_name} in {time_taken:.2f} seconds")

        # Extract Data
        data = new_response.get("Data", {})
        print(f"Data: {data}")
        title = data.get("title", "")
        # print(f"Title: {title}")
        description = data.get("description", "")
        # print(f"Description: {description}")
        # tags = ', '.join(data.get("tags", []))
        
        # Process tags into a single tag string
        tags_list = data.get("tags", [])
        processed_tags = []

        for tag in tags_list:
            if isinstance(tag, str):
                 # Simple string tag
                processed_tags.append(tag)
            elif isinstance(tag, dict):
                 # Complex tag object
                tag_value = tag.get("tagValue", "")
                tag_type = tag.get("tagType", "")
                is_exist = tag.get("isExist", "")

                tag_obj = f"{{tagValue: '{tag_value}', tagType: '{tag_type}', isExists: {is_exist}}}"
                processed_tags.append(tag_obj)

        tags = ', '.join(processed_tags)
        # print(f"Tags: {tags}")

        # Process fields into a single field string
        fields_list = data.get("fields", [])
        fields_data = []
        for field in fields_list:
            associatedTag = field.get("associatedTag", "")
            field_name = field.get("field_name", "")
            field_type = field.get("field_type", "")
            field_values = field.get("field_value", [])
            # Ensure field_value is treated correctly
            if isinstance(field_values, list):
                field_value_str = ', '.join([''.join(v) if isinstance(v, list) else str(v) for v in field_values])
            else:
                field_value_str = str(field_values)  # Handle non-list cases safely

            field_obj = f"{{associatedTag: '{associatedTag}', field_name: '{field_name}', field_type: '({field_type})', field_value: '{field_value_str}'}}"
            fields_data.append(field_obj)

        fields = ', '.join(fields_data)
        # print(f"Fields: {fields}")

        # Store the complete JSON response
        json_response = json.dumps(new_response)

        # Return the result
        return {
            "Image": img_name,
            "Data List": data_list,
            "Title": title,
            "Description": description,
            "Tags": tags,
            "Fields": fields,
            "Time": time_taken,
            "Input Token Count": input_tokens,
            "Output Token Count": output_tokens,
            "Search Tool Used": search_tool,
            "Json Response": json_response
        }

    except Exception as e:
        print(f"Error in getting response for {img_name}: {e}")
        return {
            "Image": img_name,
            "Data List": data_list,
            "Title": "",
            "Description": "",
            "Tags": "",
            "Fields": f"Error: {str(e)}",
            "Time": 0.0,
            "Input Token Count": "",
            "Output Token Count": "",
            "Search Tool Used": "",
            "Json Response": ""
        }
    
async def convert_tag_structure(data_list: dict) -> dict:
    """Convert given data json format to nested dictionary format."""
    result = {
        "base category": [],
        "tags": {},
        "fields": data_list.get("fields", [])
    }
    
    # Extract the tagTypeAndValues list
    tag_list = data_list.get("tagTypesAndValues", [])

    # Base Category tags:
    category_tags = []

    # Convert each tag entry to key-value pair
    for tag_entry in tag_list:
        tag_name = tag_entry.get("tag")
        tag_type = tag_entry.get("type")
        child_tags = tag_entry.get("childTags", [])  
    
        if tag_name:
            if tag_type == "CATEGORY":
                category_tags.append(tag_name)
                result["tags"][tag_name] = child_tags
            else:
                result["tags"][tag_name] = child_tags

    result["base category"] = category_tags
    
    return result

async def convert_json_to_yaml(data: dict)-> str:
    # data_js = data.loads(data)

    yaml_data = yaml.safe_dump(data, default_flow_style=False)

    return yaml_data

async def generate_cache_token(api_key: str, prompt: str)-> str:
    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.0-flash"

    cache = client.caches.create(
    model=model_id,
    config=CreateCachedContentConfig(
        display_name='gen_instructions', # used to identify the cache
        system_instruction=(
            'You are an expert image analyzer, and your job is to '
            'generate types, description, tags and fields as per given instrauctions for an object in image.'
            ),
        contents=[prompt],
        ttl="300s",
        )
    )

    return cache.name
    
async def process_csv_file(csv_file_path: str, img_dir: str, api_key: str, prompt_text: str, max_concurrent: int = 5) -> None:
    """Process all images from a CSV file with controlled concurrency"""
    print(f"Generating Responses...")

    df = pd.read_csv(csv_file_path)
    df = df[df['Image'] != 'SUMMARY']

    # Check if SearchTool column exists, if not, default to False
    if 'SearchTool' not in df.columns:
        df['SearchTool'] = False

    results = []
    total_time = 0
    input_total_tokens_all = 0
    output_total_tokens_all = 0
    successful_images = 0
    total_images = len(df)
    search_tool_total_usage = 0

    # print(max_concurrent)
    
    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_process(img_name: str, yaml_list: str, data_list: dict, search_tool: bool, base_categories: list):
        """Process an image with a semaphore to limit concurrency"""
        async with semaphore:
            # Add small delay between requests to reduce server load
            await asyncio.sleep(0.5)  # 500ms delay between requests
            result = await process_img(img_name, img_dir, prompt_text, api_key, yaml_list, data_list, search_tool, base_categories)
            return result

     # Create tasks for all images
    tasks = []
    for index, row in df.iterrows():
        img_name = row['Image']
        if img_name == 'SUMMARY':
            continue
            
        # categories = row['Base Category'].split(', ') if isinstance(row['Base Category'], str) else []
        # req_fields = row['Required Fields'].split(', ') if isinstance(row['Required Fields'], str) else []

        img_data_list = row["Data List"]

        # print(f" Tag list: \n:{tag_list}")
        
        # Determine search tool usage
        search_tool = row['SearchTool']
        print(f"\nJSON given:{img_data_list}")
        js_list = json.loads(img_data_list)

        # print(f"Tag List: \n{js_list}")

        new_data_list = await convert_tag_structure(js_list)

        base_categories = new_data_list["base category"]

        # print(f"New Tag List: \n{new_data_list}")
        # print(f"\nbase category: {base_categories}")

        yaml_data_list = await convert_json_to_yaml(new_data_list)

        # print(f"Tag list in yaml: \n{yaml_data_list}")
        # print(f"Type of yaml tag list:{type(yaml_data_list)}")

        # print(f"Prompt text: {prompt_text}")
        
        # cache_name = await generate_cache_token(api_key, prompt_text)
        # print(f"cache name: {cache_name}, type: {type(cache_name)}")

        
        # Convert to boolean if it's not already
        if isinstance(search_tool, str):
            search_tool = search_tool.lower() in ['true', '1', 'yes']
        
        print(f"Queueing {index+1}: {img_name} (Search Tool: {search_tool})") #type:ignore
        
        task = limited_process(img_name, yaml_data_list, img_data_list, search_tool, base_categories)
        tasks.append(task)
    
    # Run all tasks concurrently with the semaphore controlling max concurrency
    results = await asyncio.gather(*tasks)
        
    # Update metrics
    for result in results:
        time_taken = float(result.get("Time", 0))
        input_token_count = result.get("Input Token Count", 0)
        output_token_count = result.get("Output Token Count", 0)
        search_tool_usage = result.get("Search Tool Used", 0)
        if time_taken > 0 and result.get("Title", ""):
            total_time += time_taken
            successful_images += 1

        if input_token_count:
            input_total_tokens_all += input_token_count
        if output_token_count:
            output_total_tokens_all += output_token_count
        
        if search_tool_usage == "Yes":
            search_tool_total_usage += 1
            
    
    # Calculating average time
    avg_time = total_time / successful_images if successful_images > 0 else 0
    avg_input_tokens = int(input_total_tokens_all / successful_images) if successful_images > 0 else 0
    avg_output_tokens = int(output_total_tokens_all / successful_images) if successful_images > 0 else 0
    
    # Create a new DataFrame with results to ensure proper types
    results_df = pd.DataFrame(results)
    
    # Format the Time column to have 2 decimal places
    results_df['Time'] = results_df['Time'].apply(lambda x: f"{x:.2f}")
    
    # Add summary statistics
    summary_df = pd.DataFrame([{
        "Image": "SUMMARY",
        "Data List": f"Total Images: {total_images}",
        "Title": f"Successfully Processed: {successful_images}",
        "Description": f"Failed: {total_images - successful_images}",
        "Tags": f"Total: {total_time:.2f}s",
        "Fields": f"Average: {avg_time:.2f}s",
        "Time": f"Total Input Tokens: {int(input_total_tokens_all)}",
        "Input Token Count": f"Average Input Tokens: {int(avg_input_tokens)}",
        "Output Token Count": f"Total Output Tokens: {int(output_total_tokens_all)}",
        "Search Tool Used": f"Average Output Tokens: {int(avg_output_tokens)}",
        "Json Response": f"Total Search Tool Usage: {int(search_tool_total_usage)}"
    }])
    
    # Combine the results with the summary
    results_df = pd.concat([results_df, summary_df], ignore_index=True)
    
    # Save the results to CSV
    results_df.to_csv(csv_file_path, index=False)
    
    # Print summary statistics
    print(f"\nSUMMARY:")
    print(f"Total Images: {total_images}")
    print(f"Successfully Processed: {successful_images}")
    print(f"Failed: {total_images - successful_images}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Image: {avg_time:.2f} seconds")
    print(f"Total Input Tokens: {int(input_total_tokens_all)}")
    print(f"Average Input Tokens per Image: {int(avg_input_tokens)}")
    print(f"Total Output Tokens: {int(output_total_tokens_all)}")
    print(f"Average Output Tokens per Image: {int(avg_output_tokens)}")
    print(f"Total Search Tool Usage: {int(search_tool_total_usage)}")
    
    print(f"Processing complete! Results saved in {csv_file_path}.")

async def main_async():
    start_final = time.time()
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-f", "--file", required=False, help="Name of the CSV file")
    parser.add_argument("-c", "--concurrent", type=int, default=2, help="Maximum number of concurrent requests")
    
    args = parser.parse_args()
    
    csv_file = args.file
    max_concurrent = args.concurrent  # Maximum number of concurrent requests

    
    if csv_file is not None:
        print(f"Received csv file: {csv_file}")
    elif csv_file is None:
        print("CSV file is not provided. Exiting...")
        sys.exit(1)
    
    prompt_text = """
    Analyze the given image, list of base category, tags and fields to generate: title, description, tags, fields. Title should be concise, accurate. If there is a brand name or title, identify it. Max length 100 characters.
    
    Description should focus only on the foreground object(s) and no more than 80 words, providing relevant details about their characteristics, actions, or context while ignoring the background. If you are able to identify known product in image, provide its details rather than a simple captioning of what is visible. E.g., for a known book by an author, mention when the book was released, publisher, genre, and short summary of what the book is about rather than just captioning the text and image on the front cover of the book. Avoid using demonstrative pronouns ('this is', 'these are').  
    
    Tags and Fields are always NON EMPTY in output.

    Generate 7-10 hierarchical tags for objects using the provided YAML tag hierarchy. Structure tags from broad categories to specific attributes. Use attached yaml list for tag hierarchy structure.

    For tags, do not include base category in the final output as these should be excluded. Progress from broad to specific attributes, but base category can only be a tagType not tagValue. There can be multiple hierarchy level taggings, for each base category type.. The tagType for a tag which is present in yaml should not be changed to different tagType. You can add NEW TAG which is not in structure at any hierarical level if no relevant tag is present in yaml. For a new tag which is not in the tag structure, for its tagType, consider hierarchy tags from all base categories in the YAML. Each tag requires: tagValue (max 50 chars), tagType (hierarchy level, max 50 chars), isExist (true or false - MUST BE LOWERCASE) (true if tag exists in YAML, false if its a new tag). Always try to generate atleast one new tag. Values within basecategory cannot be included in fields tagValue.
    
    ALWAYS generate relevant field name, type, value and associatedTag based on the objects and tags provided. Field names should ALWAYS be in list format.
    
    Field should ALWAYS be present and NON EMPTY. If field value can't be determined, provide best possible field related to tags, and also provided that 'associatedTag', there cab be multiple 'associatedTags' for a field. Field type can be: TEXT, NUMBER, DATE, LOCATION. Categorize alphanumeric values ('12 AB') and numbers with units ('15 mm') as TEXT. Numeric values ('123 45') should have whitespace removed and be categorized as NUMBER. If possible, add LOCATION type Field. Use INDIAN METRIC UNITS, not North American units for field values you return. AVOID GENERATING RANDOM AND UNNECESSARY FIELDS. Each field value must be only single value, no multiple entries. AVOID ADDING TITLE AND DESCRIPTION IN FIELDS. You can generate fields on your own as well. Try giving 3 -4 field values. 

    While giving field values, only generate field if you are sure of the value, do not hallucinate, ONLY add field, if correct field value is vaailable with you with surety. 

    Each field must have: Meaningful 'field_name' (max 50 characters), Valid 'field_type', Non-empty 'field_value' (max 500 characters)
            
    Identify relevant attributes. Use search tool only if necessary but if it is used make sure to provide as many relevant results. In case of a known product, identify model number, SKU, etc. E.g., for image of a book - make sure to always search and fetch the title, author, publisher, ISBN number. Then, provide other relevant details from the web that will be useful to know for the user. Make sure to provide fields that are relevant to the category type provided. For example, height, length, width are not usually provided for the category type 'car'. 
    
    IMPORTANT JSON FORMAT REQUIREMENTS:
    - Use lowercase boolean values: true/false (NOT TRUE/FALSE)
    - Ensure proper JSON syntax with correct commas and brackets
    - Do not include trailing commas
    - All string values must be properly quoted
    
    Return the response strictly in following JSON format, without additional text, explanation, or preamble:
    {
    "Data": {
        "title": "To Kill a Mockingbird by Harper Lee",
        "description": "A gripping, heart-wrenching, and wholly remarkable tale of coming-of-age in a South poisoned by virulent prejudice. It views a world of great beauty and savage inequities through the eyes of a young girl, as her father—a crusading local lawyer—risks everything to defend a black man unjustly accused of a terrible crime.",
        "tags": [
            {
                "tagValue": "CLASSIC",
                "tagType": "BOOKS",
                "isExist": true
            },
            {
                "tagValue": "LITERATURE",
                "tagType": "CLASSIC",
                "isExist": true
            },
            {
                "tagValue": "NOVEL",
                "tagType": "LITERATURE",
                "isExist": true
            },
            {
                "tagValue": "THRILLER",
                "tagType": "NOVEL",
                "isExist": false
            }
        ],
        "fields": [
            {   
                "associatedTag":"NOVEL",
                "field_name": "ISBN",
                "field_type": "NUMBER",
                "field_value": "9780061120084"
            },
            {   
                "associatedTag": "THRILLER",
                "field_name": "Genre",
                "field_type": "TEXT",
                "field_value": "Thriller"
            }
        ]
    }
    }
    """
    # Set current working directory
    dir = Path(os.getcwd())

    # Load environment variables from .env file
    ENV_PATH :Path= dir / '.env'
    load_dotenv(ENV_PATH)

    # Image file directory
    img_dir :str = os.path.join(dir, 'asset_images')

    # CSV file directory
    csv_file_dir = os.path.join(dir, "csv_files")

    # Get API URL from environment variables
    GEMINI_API_KEY :str = os.environ["GEMINI_API_KEY"]

    if csv_file is not None:
        csv_file_path = os.path.join(csv_file_dir, csv_file)
        await process_csv_file(csv_file_path, img_dir, GEMINI_API_KEY, prompt_text, max_concurrent)

    end_final = time.time() - start_final
    print(f"Total time taken in processing: {end_final}")

def main():
    asyncio.run(main_async())

if __name__ == '__main__':
    main()