import os
import json
import csv

# Directory containing the JSON files
directory = 'C:/Users/Miley/openpose/build3/output/_test/nw'

# List to store the extracted values
data = []

# Indices to remove from the keypoints list
indices_to_remove = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 23, 26, 
                     29, 32, 35, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 
                     56, 59, 62, 65, 68, 71, 74]

# Iterate through the JSON files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        
        # Read the JSON file
        with open(file_path) as f:
            json_data = json.load(f)
        
        # Extract the desired values
        keypoints = json_data['people'][0]['pose_keypoints_2d']
        
        # Remove specific indices from the keypoints list
        keypoints = [val for i, val in enumerate(keypoints) if i not in indices_to_remove]
        
        # Append the values to the data list
        data.append(keypoints)

# Save the data to a CSV file
output_file = 'output.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"Extraction complete. Data saved to {output_file}.")