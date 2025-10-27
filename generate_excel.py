import json
import pandas as pd
from pathlib import Path

# Define paths
output_dir = Path("output")
input_json_path = output_dir / "detection_results.json"
output_excel_path = output_dir / "detection_results.xlsx"

# Load detection results from JSON
if not input_json_path.exists():
    print(f"Error: '{input_json_path}' not found. Run 'object_detection.py' first.")
    exit()

# Attempt to load the JSON data
with open(input_json_path, "r") as json_file:
    try:
        detection_results = json.load(json_file)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON: {e}")
        exit()

# Debugging: Check if detection_results is loaded correctly
if not detection_results:
    print("Error: detection_results.json is empty or invalid.")
    exit()

print("\nDetection Results Loaded:")
print(detection_results[:1])  # Print the first entry to inspect

# Prepare data for the first sheet
excel_data = []
for entry in detection_results:
    row = {"Image Name": entry["Image Name"], "Category": entry["Category"]}
    for detection in entry["Detections"]:
        row[detection["Class"]] = detection["Confidence"]
    excel_data.append(row)

# Convert to DataFrame for the first sheet
df_main = pd.DataFrame(excel_data)

# Debugging: Check if df_main is created correctly
print("\nMain DataFrame:")
print(df_main.head())

# Ensure 'Image Number' is extracted from 'Image Name'
if 'Image Name' not in df_main.columns:
    print("Error: 'Image Name' column missing in df_main.")
    exit()

df_main["Image Number"] = df_main["Image Name"].apply(lambda x: int(x.split("_")[0]))

# Sort the DataFrame by Image Number in ascending order
df_main = df_main.sort_values(by="Image Number")

# Clean category names by stripping extra spaces and converting to lowercase
df_main["Category"] = df_main["Category"].str.strip().str.lower()

categories = ["no_fault", "blur_low", "blur_medium", "blur_extreme"]
category_confidences = []

# Iterate over image numbers to extract the confidence score of 'cat' for each category
for image_number in df_main["Image Number"].unique():
    row = {"Image Number": image_number}
    
    for category in categories:
        # Filter the rows for the current image number and category
        filtered_row = df_main[(df_main["Image Number"] == image_number) & (df_main["Category"] == category)]
        
        if not filtered_row.empty:
            # Extract all scores except for 'cat' and convert to numeric
            other_class_scores = filtered_row.drop(columns=['Image Name', 'Image Number', 'Category', 'cat'], errors='ignore').values.flatten()
            other_class_scores = pd.to_numeric(other_class_scores, errors='coerce')  # Convert to numeric, invalid to NaN
            
            # Check if there are non-zero confidence scores for any other class
            has_other_scores = any(score > 0 for score in other_class_scores if not pd.isna(score))
            
            # Assign the confidence score for 'cat' or 0 based on the condition
            if "cat" in filtered_row.columns and not filtered_row["cat"].isna().all():
                if has_other_scores:
                    row[category] = 0  # Other class detected, set to 0
                else:
                    row[category] = filtered_row["cat"].values[0]  # Use cat confidence
            else:
                row[category] = 0  # No cat confidence, set to 0
        else:
            row[category] = 0  # If no rows exist for the category, set to 0

    category_confidences.append(row)

# Create a new DataFrame for the second sheet
df_category_confidence = pd.DataFrame(category_confidences)

# Convert all category columns to numeric
for category in categories:
    df_category_confidence[category] = pd.to_numeric(df_category_confidence[category], errors='coerce')

# Add a column to check if the confidence values satisfy the specified condition
df_category_confidence['Is Ordered'] = (
    (df_category_confidence['no_fault'] >= df_category_confidence['blur_low']) &
    (df_category_confidence['blur_low'] >= df_category_confidence['blur_medium']) &
    (df_category_confidence['blur_medium'] >= df_category_confidence['blur_extreme'])
)

# Reorder the columns to match the specified order
df_category_confidence = df_category_confidence[['Image Number', 'no_fault', 'blur_low', 'blur_medium', 'blur_extreme', 'Is Ordered']]

# Debugging: Check the contents of the second sheet
print("\nSecond Sheet DataFrame (Cat Confidence by Category):")
print(df_category_confidence.head())

# Calculate accuracy percentage
total_rows = len(df_category_confidence)
true_rows = df_category_confidence['Is Ordered'].sum()  # Sum of True values
accuracy_percentage = (true_rows / total_rows) * 100 if total_rows > 0 else 0

# Display accuracy
print(f"\nAccuracy: {accuracy_percentage:.2f}% ({true_rows}/{total_rows} rows are True)")

# Save both sheets to the Excel file
with pd.ExcelWriter(output_excel_path) as writer:
    df_main.to_excel(writer, sheet_name="Detections", index=False)
    df_category_confidence.to_excel(writer, sheet_name="Cat Confidence by Category", index=False)

print(f"Excel file saved to {output_excel_path}")

