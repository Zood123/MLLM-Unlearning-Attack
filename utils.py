import json
from PIL import Image
from io import BytesIO

def flatten_dataset(df):
    """
    Flatten the dataset such that each question-answer pair becomes a single item.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'image' column with byte data and 'metadata' column with QA pairs.
    
    Returns:
        list: List of dictionaries with image data and each QA pair.
    """
    flattened_data = []

    for idx, row in df.iterrows():
        # Extract the bytes from the 'image' dictionary
        image_data = row['image'].get('bytes')  # Access the image bytes

        # Convert the image bytes to a PIL Image
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            continue

        # Safely load metadata as JSON
        try:
            metadata = json.loads(row['metadata'])  # Using json.loads to parse JSON safely
        except json.JSONDecodeError as e:
            print(f"Error decoding metadata at index {idx}: {e}")
            continue

        for qa_pair in metadata:
            question = qa_pair.get("Question", "")
            answer = qa_pair.get("Answer", "")
            guidance = qa_pair.get("Guidance", "")

            if question and answer:
                flattened_data.append({
                    "image": image,
                    "question": question,
                    "guidance": guidance,
                    "answer": answer
                })

    #print(flattened_data)
    #print(len(flattened_data))
    return flattened_data
