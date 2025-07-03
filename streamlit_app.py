import streamlit as st
import pandas as pd
import base64
import os
import subprocess
from io import BytesIO
from PIL import Image
import csv
import sys
from streamlit.runtime.uploaded_file_manager import UploadedFile 
import re
import zipfile
import shutil
from typing import Optional
import json

# Directories for saving images and CSV
IMAGE_SAVE_DIR = "asset_images"
CSV_SAVE_DIR = "csv_files"

st.set_page_config(page_title="D'Kluter Data Acqusition", layout="wide")

# Ensure the directories exist
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(CSV_SAVE_DIR, exist_ok=True)

# Sample CSV Format
sample_data = pd.DataFrame({
    "Image": ["Image1.png", "Image2.jpg"],
    "Data List": ["Json tag list 1", "Json tag list 2"],
    "Title": ["", ""],
    "Description": ["", ""],
    "Tags": ["", ""],
    "Fields": ["", ""]
})

# Function to convert an uploaded image to Base64 (for preview in Streamlit)
def image_to_base64(uploaded_file):
    image = Image.open(uploaded_file)
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Convert all images to PNG for consistency
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# Function to save uploaded images
def save_uploaded_image(uploaded_file:UploadedFile)->str:
    file_path = os.path.join(IMAGE_SAVE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save image as file
    return file_path  # Return saved file path

# Function to save uploaded CSV
def save_uploaded_csv(uploaded_file:UploadedFile)->str:
    csv_file_path = os.path.join(CSV_SAVE_DIR, uploaded_file.name)
    with open(csv_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return csv_file_path

# Function to extract images from a zip file
def extract_images_from_zip(zip_file:UploadedFile)->list:
    # Clean the image directory first
    shutil.rmtree(IMAGE_SAVE_DIR, ignore_errors=True)
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
    extracted_files = []
    
    with zipfile.ZipFile(BytesIO(zip_file.getbuffer())) as z:
        for filename in z.namelist():
            # Only extract image files
            lower_filename = filename.lower()
            if lower_filename.endswith(('.jpg', '.jpeg', '.png')):
                # Extract only files, not directories
                if not filename.endswith('/'):
                    z.extract(filename, IMAGE_SAVE_DIR)
                    # Strip any directory structure and just get the basename
                    base_filename = os.path.basename(filename)
                    # If needed, move files from nested directories to the root image directory
                    if filename != base_filename:
                        extracted_path = os.path.join(IMAGE_SAVE_DIR, filename)
                        target_path = os.path.join(IMAGE_SAVE_DIR, base_filename)
                        # Create parent directories if they don't exist
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        # Move the file if it's not already there
                        if os.path.exists(extracted_path) and extracted_path != target_path:
                            shutil.move(extracted_path, target_path)
                    
                    extracted_files.append(base_filename)
    
    return extracted_files

def get_csv_download_link(df: pd.DataFrame, file_path: str)-> str:
    """
    Generates a download link for the processed CSV file.
    """
    csv = df.to_csv(index=False)  # Convert DataFrame to CSV string
    b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV as Base64
    filename = os.path.basename(file_path)
    
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Output CSV</a>'

def get_sample_csv_download_link(df):
    csv = df.to_csv(index=False)  # Convert DataFrame to CSV
    b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV as Base64
    return f'<a href="data:file/csv;base64,{b64}" download="sample_template.csv">📥 Download Example CSV</a>'
    
# Function to display results from CSV
def display_results(csv_path:str)->Optional[pd.DataFrame]:
    if os.path.exists(csv_path):
        # Read the CSV file
        updated_df = pd.read_csv(csv_path)
        
        # Extract summary row (last row) and remove it from the main dataframe
        if "SUMMARY" in updated_df["Image"].values:
            summary_row = updated_df[updated_df["Image"] == "SUMMARY"].iloc[0].to_dict()
            results_df = updated_df[updated_df["Image"] != "SUMMARY"].copy()
        else:
            summary_row = None
            results_df = updated_df.copy()
        
        # Display results
        st.subheader("Analysis Results")
        
        # Display each image with its analysis
        for i, (_, row) in enumerate(results_df.iterrows()):
            with st.container():
                st.markdown(f"### Image {i + 1}")
                
                # Create columns for image, content, and tags
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    # Display image
                    image_path = os.path.join(IMAGE_SAVE_DIR, row['Image'])
                    if os.path.exists(image_path):
                        st.image(image_path, width=200)
                    else:
                        st.warning(f"Image not found: {row['Image']}")
                
                with col2:
                    # Display title and description
                    st.markdown(f"**Title:** {row['Title']}")
                    st.markdown(f"**Description:**")
                    st.markdown(f"{row['Description']}")
                    # Display Token Count
                    st.markdown(f"**Input Token Count:** {int(row['Input Token Count'])}")
                    st.markdown(f"**Output Token Count:** {int(row['Output Token Count'])}")

                    # Display Search Tool Usage
                    st.markdown(f"**Search Tool Used:** {row['Search Tool Used']}")
                
                with col3:                  
                    st.markdown("**Tags:**")
                    if isinstance(row['Tags'], str) and row['Tags']:
                        # Remove all brackets and braces, then split by '}, {'
                        tags_clean = row['Tags'].replace('[', '').replace(']', '').replace('{', '').replace('}', '')
                        tag_objects = tags_clean.split(', tagValue:')
                        
                        for i, tag_obj in enumerate(tag_objects, 1):
                            # Add back 'tagValue:' to first part if needed
                            if i > 1 and not tag_obj.startswith('tagValue:'):
                                tag_obj = 'tagValue:' + tag_obj
                                
                            st.markdown(f"**Tag {i}:**")
                            
                            # Split by comma and clean up each part
                            parts = tag_obj.split(',')
                            formatted_parts = []
                            
                            for part in parts:
                                if ':' in part:
                                    key, value = part.split(':', 1)
                                    key = key.strip().strip("'\"")
                                    value = value.strip().strip("'\"")
                                    formatted_parts.append(f"{key}: {value}")
                            
                            st.markdown("```\n" + "\n".join(formatted_parts) + "\n```")
                                    
                    # Display Fields
                    st.markdown("**Fields:**")
                    if 'Fields' in row and isinstance(row['Fields'], str) and row['Fields']:
                        
                        # Split into individual field objects
                        field_objects = row['Fields'].split('}, {')
                        
                        for i, field_obj in enumerate(field_objects):
                            # Clean up the field object string
                            field_obj = field_obj.strip().lstrip('{').rstrip('}')
                            
                            # Simple string splitting approach
                            parts = field_obj.split(', ')
                            
                            # Initialize variables
                            associated_tag = field_name = field_type = field_value = "N/A"
                            
                            # Extract each part
                            for part in parts:
                                part = part.strip()
                                if part.startswith("associatedTag:"):
                                    associated_tag = part.split(":", 1)[1].strip().strip("'\"")
                                elif part.startswith("field_name:"):
                                    field_name = part.split(":", 1)[1].strip().strip("'\"")
                                elif part.startswith("field_type:"):
                                    field_type = part.split(":", 1)[1].strip().strip("'\"()")
                                elif part.startswith("field_value:"):
                                    field_value = part.split(":", 1)[1].strip().strip("'\"")
                            
                            # Display the field
                            st.markdown(f"""
                            **Field {i+1}:**
                            ```
                            Associated Tag: {associated_tag}
                            Field Name:     {field_name}
                            Field Type:     {field_type}
                            Field Value:    {field_value}
                            ```
                            """)

                                    
                st.markdown("---")
        
        if summary_row:
            st.subheader("Summary")
            summary_container = st.container()
            with summary_container:
                cols = st.columns(3)
                
                with cols[0]:
                    st.metric("Total Images", summary_row.get("Data List", "").split(": ")[-1])
                    st.metric("Successfully Processed", summary_row.get("Title", "").split(": ")[-1])
                    st.metric("Failed", summary_row.get("Description", "").split(": ")[-1])
                
                with cols[1]:
                    # Parse time from Title
                    total_time = summary_row.get("Tags", "")
                    if "Total:" in total_time:
                        full_time = total_time.split("Total:")[-1].strip()
                        st.metric("Total Processing Time", full_time)

                    # Parse average time from Description
                    time_text = summary_row.get("Fields", "")
                    if "Average:" in time_text:
                        avg_time = time_text.split("Average:")[-1].strip()
                        st.metric("Average Processing Time per Image", avg_time)

                    # Search Tool Usage
                    search_used = summary_row.get("Json Response", "")
                    if "Total Search Tool Usage:" in search_used:
                        tool_used = search_used.split("Total Search Tool Usage:")[-1].strip()
                        st.metric("Total Responses using Search Tool", tool_used)
                
                with cols[2]:
                    # Parse total input token from Fields
                    input_token_count = summary_row.get("Time", "")
                    if "Total Input Tokens:" in input_token_count:
                        total_input_tokens = input_token_count.split("Total Input Tokens:")[-1].strip()
                        st.metric("Total Input Token Count", total_input_tokens)

                    # Parse average input token from Time
                    avg_input_token_count = summary_row.get("Input Token Count", "")
                    if "Average Input Tokens:" in avg_input_token_count:
                        avg_input_tokens = avg_input_token_count.split("Average Input Tokens:")[-1].strip()
                        st.metric("Avg Input Token Count per Image", avg_input_tokens)

                    # Parse total output token from Input Token Count
                    output_token_count = summary_row.get("Output Token Count", "")
                    if "Total Output Tokens:" in output_token_count:
                        total_output_tokens = output_token_count.split("Total Output Tokens:")[-1].strip()
                        st.metric("Total Output Token Count", total_output_tokens)

                    # Parse average output token from Output Token Count
                    avg_output_token_count = summary_row.get("Search Tool Used", "")
                    if "Average Output Tokens:" in avg_output_token_count:
                        avg_output_tokens = avg_output_token_count.split("Average Output Tokens:")[-1].strip()
                        st.metric("Avg Output Token Count per Image", avg_output_tokens)
        
        # Show raw data in expandable section
        with st.expander("Show CSV Output Data"):
            st.dataframe(updated_df)

        # Add Download Button
        st.markdown(get_csv_download_link(updated_df, csv_path), unsafe_allow_html=True)

        # Display Gemini pricing table
        st.subheader("Gemini API Pricing")
        pricing_data = {
            "Pricing Type": ["Input price (Image/Text/Video)", "Output price", "Context caching price (Text/Image/Video)", "Context caching (storage)", "Grounding with Google Search"],
            "Free Tier": ["Free of charge", "Free of charge", "Free of charge", "Free of charge, up to 1,000,000 tokens of storage per hour [Available March 31, 2025]", "Free of charge, up to 500 RPD"],
            "Paid Tier (per 1M tokens in USD)": ["$0.10", "$0.40", "$0.025 / 1,000,000 tokens [Available March 31, 2025]", "$1.00 / 1,000,000 tokens per hour [Available March 31, 2025]", "1,500 RPD (free), then $35 / 1,000 requests"]
        }

        pricing_df = pd.DataFrame(pricing_data)
        st.table(pricing_df)

        st.markdown("Note: RPD stands for Requests Per Day. It is only applicable if you enabled Google Search Tool.")

        return updated_df
        
    else:
        st.error("CSV file not found!")

        return None

# Function to validate if image files in CSV exist in the extracted files
def validate_csv_images(csv_path:str, extracted_files:list)->list:
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            image_column = df["Image"]
            
            missing_images = []
            for image in image_column:
                if image != "SUMMARY" and image not in extracted_files:
                    missing_images.append(image)
            
            return missing_images
        except Exception as e:
            return ["Error reading CSV: " + str(e)]
    return ["CSV file not found"]

def prepare_csv_with_search_tool(csv_path:str, enable_search_tool:bool)->pd.DataFrame:
    """
    Prepare CSV by adding a SearchTool column based on global and individual settings
    """
    # Read the existing CSV
    df = pd.read_csv(csv_path)
    
    # Remove SUMMARY row if it exists for processing
    summary_row = None
    if "SUMMARY" in df["Image"].values:
        summary_row = df[df["Image"] == "SUMMARY"].iloc[0]
        df = df[df["Image"] != "SUMMARY"]
    
    # Add SearchTool column
    df['SearchTool'] = False
    
    # Iterate through rows to set SearchTool flag
    for index, row in df.iterrows():
        # Check global search tool setting first
        if enable_search_tool:
            # Check if this specific image is NOT disabled in override
            if not st.session_state.override_search_tool.get(row['Image'], False):
                df.at[index, 'SearchTool'] = True
        else:
            # If global search is disabled, use individual flags from CSV upload workflow
            # Prioritize session state csv_search_tool_flags if available
            if hasattr(st.session_state, 'csv_search_tool_flags'):
                df.at[index, 'SearchTool'] = st.session_state.csv_search_tool_flags[index]
    
    # Add back the summary row if it existed
    if summary_row is not None:
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save the updated CSV
    df.to_csv(csv_path, index=False)
    
    return df

# Streamlit UI
st.title("Image Analysis Tool")

# Choose workflow option
workflow = st.radio("Select Workflow", ["Upload Images", "Upload CSV"])

# Checkbox to enable google search tool usage
enable_search_tool = st.checkbox("Enable Google Search Tool", help="When checked, allows the AI to use Google Search for additional context")
if enable_search_tool:
    st.success("Google Search Tool is Enabled for all images.")

# Modify the session state initialization to include a flag to override global search
if "override_search_tool" not in st.session_state:
    st.session_state.override_search_tool = {}
# Store generated results to prevent refresh.
if 'generated_results' not in st.session_state:
    st.session_state.generated_results = None

if workflow == "Upload Images":
    # Textbox for CSV filename
    csv_filename = st.text_input("Enter a CSV filename (without extension)", value="output")

    st.info("Upload single or multiple images!")

    # Upload multiple image files
    uploaded_files = st.file_uploader(
        label="Choose image files ('jpg','jpeg','png')", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True
    )

    # Initialize session state for categories, required fields and google search tool usage, if it doesn't exist
    if "data_list" not in st.session_state:
        st.session_state.data_list = []
    if "search_tool_flags" not in st.session_state:
        st.session_state.search_tool_flags = []

    # Ensure types list is the correct length
    if uploaded_files:
        # Resize the types list if needed
        if len(st.session_state.data_list) < len(uploaded_files):
            # Extend the list with empty strings if there are new images
            st.session_state.data_list.extend([""] * (len(uploaded_files) - len(st.session_state.data_list)))
            st.session_state.search_tool_flags.extend([False] * (len(uploaded_files) - len(st.session_state.search_tool_flags)))
        elif len(st.session_state.data_list) > len(uploaded_files):
            # Trim the list if images were removed
            st.session_state.data_list = st.session_state.data_list[:len(uploaded_files)]
            st.session_state.search_tool_flags = st.session_state.search_tool_flags[:len(uploaded_files)] 
        
        st.subheader("Preview of Data")
        # Display image previews and inputs
        for i, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([1, 1, 1])
            
            # Display image in the first column
            with col1:
                st.image(file, width=150, caption=f"Image {i+1}")

            # Display JSON data input in the second column
            with col2:
                st.markdown(f"**JSON Data for Image {i+1}:**")
                new_data_list = st.text_area(
                f"JSON Data for Image {i+1}", 
                value=st.session_state.data_list[i],
                key=f"data_list_{i+1}",
                height=100,
            )
                st.session_state.data_list[i] = new_data_list
                

            # Display search tool checkbox in the third column
            with col3:
                # If global search is enabled, add an option to override
                if enable_search_tool:
                    # Use a checkbox to override global search setting
                    override = st.checkbox(
                        "Disable Search", 
                        value=st.session_state.override_search_tool.get(file.name, False),
                        key=f"override_search_{i+1}"
                    )
                    # Store the override status
                    st.session_state.override_search_tool[file.name] = override
                else:
                    # If global search is disabled, use existing search tool checkbox
                    st.session_state.search_tool_flags[i] = st.checkbox(
                        f"Search Tool for Image {i+1}", 
                        value=st.session_state.search_tool_flags[i],
                        key=f"search_tool_{i+1}"
                    )

        # Auto-save files and prepare CSV data when files are uploaded
        csv_data = []
        for i, file in enumerate(uploaded_files):
            image_path = save_uploaded_image(file)  # Save each image
            csv_data.append({
                "Image": file.name,
                "Data List": st.session_state.data_list[i],
                "Title": "",
                "Description": "",
                "Tags": "",
                "Fields": ""
            })
                
        # Generate CSV filename
        csv_relative_name = f"{csv_filename}.csv"  
        csv_full_path = os.path.join(CSV_SAVE_DIR, csv_relative_name)

        # Define all field names
        fieldnames = ["Image", "Data List", "Title", "Description", "Tags", "Fields"]

        # Write CSV file
        with open(csv_full_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        # st.success(f"Images saved in '{IMAGE_SAVE_DIR}/' and CSV saved as '{csv_full_path}'")

        # Store the relative CSV name for running the CLI
        st.session_state["csv_name"] = csv_relative_name  
        st.session_state["csv_full_path"] = csv_full_path

    # Button to execute CLI command
    if "csv_name" in st.session_state:
        if st.button("Generate AI Response"):
            with st.spinner("Generating Response..."):
                # Prepare CSV with SearchTool column
                prepare_csv_with_search_tool(st.session_state["csv_full_path"], enable_search_tool)
                
                # Modify command to ALWAYS include the CSV
                command = [sys.executable, "main.py", "-f", st.session_state["csv_name"]]
                
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
                    st.success("Command executed successfully!")

                    # st.text_area("Command Output", result.stdout)

                    print("Command Output", result.stdout)

                    # Display the results
                    display_results(st.session_state["csv_full_path"])
                    
                except subprocess.TimeoutExpired:
                    st.error("The command timed out! Please try again with a smaller dataset or check the system performance.")
                except subprocess.CalledProcessError as e:
                    st.error("Error while executing command.")
                    st.text_area("Error Message", e.stderr)

else:
    # CSV Upload workflow
    st.subheader("Upload CSV File")

    st.markdown("### Example CSV Format:")
    st.dataframe(sample_data)  # Display example CSV structure
    st.markdown(get_sample_csv_download_link(sample_data), unsafe_allow_html=True)
    
    # Upload CSV file
    uploaded_csv = st.file_uploader("Choose a CSV file", type=['csv'])

    st.info("Create a zip file of all the images mentioned in CSV file and upload it here.")

    # Upload ZIP file containing images
    uploaded_zip = st.file_uploader("Upload ZIP file containing images", type=['zip'])
    
     # Variables to track state
    csv_path = None
    extracted_files = []
    csv_data = None

    if uploaded_csv is not None:
        # Save the uploaded CSV
        csv_path = save_uploaded_csv(uploaded_csv)

        # Read the CSV data
        try:
            csv_data = pd.read_csv(csv_path)
            # Initialize data list in session state if needed
            if "csv_data_list" not in st.session_state:
                st.session_state.csv_data_list = csv_data["Data List"].tolist()
            elif len(st.session_state.csv_data_list) != len(csv_data):
                st.session_state.csv_data_list = csv_data["Data List"].tolist()

            # Initialize search tool flags for each image
            if 'csv_search_tool_flags' not in st.session_state:
                st.session_state.csv_search_tool_flags = [False] * len(csv_data)
            elif len(st.session_state.csv_search_tool_flags) != len(csv_data):
                st.session_state.csv_search_tool_flags = [False] * len(csv_data)

        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            
    if uploaded_zip is not None:
        # Extract images from ZIP
        with st.spinner("Extracting images from ZIP file..."):
            extracted_files = extract_images_from_zip(uploaded_zip)
        
        st.success(f"Successfully extracted {len(extracted_files)} images.")

    # Check if both CSV and ZIP have been uploaded
    if csv_path and extracted_files and csv_data is not None:
        # Validate that all images in the CSV exist in the extracted files
        missing_images = validate_csv_images(csv_path, extracted_files)

        if missing_images:
            st.warning("Warning: The following images referenced in the CSV were not found in the ZIP file:")
            for img in missing_images[:10]:  # Show first 10 missing images
                st.write(f"- {img}")
            if len(missing_images) > 10:
                st.write(f"... and {len(missing_images) - 10} more.")

         # Display images with type inputs side by side (similar to Upload Images workflow)
        st.subheader("Preview of Data.")

        # Filter out summary row if it exists
        display_data = csv_data[csv_data["Image"] != "SUMMARY"] if "SUMMARY" in csv_data["Image"].values else csv_data

         # Display each image with its type input
        for i, (_, row) in enumerate(display_data.iterrows()):
            image_name = row['Image']
            image_path = os.path.join(IMAGE_SAVE_DIR, image_name)
            
            # Create two columns layout for each image
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Display image if it exists
                if os.path.exists(image_path):
                    st.image(image_path, width=150, caption=f"Image: {i+1}")
                else:
                    st.warning(f"Image not found: {image_name}")

            with col2:
                # Allow editing the JSON data for each image
                st.markdown(f"**JSON Data for Image {i+1}:**")
                new_data_list = st.text_area(
                f"JSON Data for Image {i+1}", 
                value=row['Data List'],
                key=f"csv_data_list_{i+1}",
                height=100,
                label_visibility="collapsed"
            )

                # Auto-update the JSON data value in session state
                st.session_state.csv_data_list[i] = new_data_list
                
                # Update the CSV data immediately when JSON data list changes
                if new_data_list != row['Data List']:
                    csv_data.at[display_data.index[i], 'Data List'] = new_data_list
                    # Save the updated CSV
                    csv_data.to_csv(csv_path, index=False)

            with col3:
                # If global search is enabled, add an option to override
                if enable_search_tool:
                    # Use a checkbox to override global search setting
                    override = st.checkbox(
                        "Disable Search", 
                        value=st.session_state.override_search_tool.get(image_name, False),
                        key=f"override_search_{i+1}"
                    )
                    # Store the override status
                    st.session_state.override_search_tool[image_name] = override
                else:
                    # If global search is disabled, use search tool checkbox
                    st.session_state.csv_search_tool_flags[i] = st.checkbox(
                        f"Search Tool", 
                        value=st.session_state.csv_search_tool_flags[i],
                        key=f"search_tool_{i+1}"
                    )

        # Auto-save updated types - this happens automatically when the types are changed above
        st.info("Categories and Required Fields are automatically updated as you change them.")

        # Extract just the filename for the CLI command
        csv_filename = os.path.basename(csv_path)

        # Button to process the uploaded CSV
        if st.button("Generate AI Response"):
            with st.spinner("Generating Response..."):
                # Prepare CSV with SearchTool column
                prepare_csv_with_search_tool(csv_path, enable_search_tool)
                
                # Extract just the filename for the CLI command
                csv_filename = os.path.basename(csv_path)
                
                # Modify command to ALWAYS include the CSV
                command = [sys.executable, "main.py", "-f", csv_filename]
                
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
                    st.success("Command executed successfully!")

                    # st.text_area("Command Output", result.stdout)

                    print("Command Output", result.stdout)

                    # Display the results
                    display_results(csv_path)
                    
                except subprocess.TimeoutExpired:
                    st.error("The command timed out! Please try again with a smaller dataset or check the system performance.")
                except subprocess.CalledProcessError as e:
                    st.error("Error while executing command.")
                    st.text_area("Error Message", e.stderr)

