import json
import pandas as pd

# Load JSON file
def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    rows = []
    # Process diseases
    for disease in data.get("diseases", []):
        rows.append({
            'name': disease.get("name", ""),
            'description': disease.get("description", ""),
            'medications': ', '.join(disease.get("medications", [])),
            'precautions': ', '.join(disease.get("precautions", [])),
            'treatment_options': ', '.join(disease.get("treatment_options", [])) if "treatment_options" in disease else ""
        })
    
    return pd.DataFrame(rows)

# Save as CSV for training
if __name__ == "__main__":
    data = load_data('data.json')  # Make sure data.json is in the same directory
    data.to_csv('processed_data.csv', index=False)
    print("Data processed and saved as 'processed_data.csv'")
