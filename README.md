# Medical_Chatbot

The **Medical Chatbot** is an AI-powered application that provides information about diseases, medications, precautions, and treatments. It uses a fine-tuned BERT model to understand user queries and deliver contextually relevant healthcare information.

## Features

- Provides detailed descriptions of diseases.
- Recommends medications and treatments.
- Shares preventive measures for specific conditions.
- User-friendly interface powered by Flask.

---

## Project Structure

1. **`data.json`**  
   A JSON file containing raw disease data, including names, descriptions, medications, precautions, and treatment options.

2. **`preprocess_data.py`**  
   A script to preprocess the raw JSON data into a CSV format suitable for fine-tuning the BERT model.

3. **`fine_tune_bert.py`**  
   A script to fine-tune the BERT model using the preprocessed data.

4. **`evaluate_model.py`**  
   A script to evaluate the fine-tuned model for accuracy and performance.

5. **`app.py`**  
   A Flask application serving as the user interface for interacting with the chatbot.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- `transformers` library
- Flask
- PyTorch
- Pandas

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/medical-chatbot.git
   cd medical-chatbot
   
### How to Run the Project
Step 1: Clone the Repository
Clone the project repository to your local machine.

git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot

Step 2: Install Required Libraries
Install the Python dependencies 


Step 3: Prepare the Dataset
Convert the raw JSON data into CSV format for training the model. Run the following command:

python preprocess_data.py
This will generate a file named processed_data.csv.

Step 4: Fine-Tune the BERT Model
Train the BERT model using the preprocessed dataset. Run the following command:

python fine_tune_bert.py
The fine-tuned model will be saved in the fine_tuned_model/ directory.

Step 5: Evaluate the Model
Test the fine-tuned model for performance and accuracy. Run the following command:

python evaluate_model.py
Step 6: Start the Flask Application
Launch the user interface to interact with the chatbot. Run the following command:

python app.py

Step 7: Access the Chatbot

Open your browser and navigate to:

http://127.0.0.1:5000
