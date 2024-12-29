from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load disease data from CSV
df = pd.read_csv("processed_data.csv")  # Ensure the CSV file is in the same directory

# List to store chat history
chat_history = []

# Function to get disease info based on user input
def get_disease_info(disease_name, info_type):
    # Search for disease in the CSV
    disease_info = df[df['name'].str.contains(disease_name, case=False, na=False)]
    if not disease_info.empty:
        disease = disease_info.iloc[0]
        if info_type == 'description':
            return f"**Description of {disease_name}:** {disease['description']}"
        elif info_type == 'medications':
            return f"**Medications for {disease_name}:** {disease['medications']}"
        elif info_type == 'precautions':
            return f"**Precautions for {disease_name}:** {disease['precautions']}"
        elif info_type == 'treatment':
            return f"**Treatment Options for {disease_name}:** {disease['treatment_options']}"
    else:
        return "Disease not found. Please check the name or try a different one."

# Function to handle user input and determine bot's response
def handle_input(user_input):
    user_input = user_input.lower().strip()

    # Response for greetings
    if 'hello' in user_input or 'hi' in user_input:
        return "Hello! How can I assist you today? You can ask about diseases, medications, or precautions."

    # Handle thank you response
    elif 'thank you' in user_input:
        return "You're welcome! Feel free to ask more questions."

    # Check for specific disease-related queries
    for disease_name in df['name']:
        if disease_name.lower() in user_input:
            # Check if the user is asking about description, medications, precautions, or treatment
            if 'medications' in user_input:
                return get_disease_info(disease_name, 'medications')
            elif 'precautions' in user_input:
                return get_disease_info(disease_name, 'precautions')
            elif 'treatment' in user_input:
                return get_disease_info(disease_name, 'treatment')
            else:
                return get_disease_info(disease_name, 'description')

    return "I'm sorry, I didn't understand that. Please try asking about a disease."

@app.route('/')
def index():
    # Welcome message
    chat_history.append({"role": "Bot", "message": "Welcome to Disease Information Finder! Feel free to ask questions about diseases, medications, or precautions."})
    return render_template('index.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['user_message']
    
    # Append user message to chat history
    chat_history.append({"role": "User", "message": user_message})
    
    # Get bot's response
    bot_response = handle_input(user_message)
    
    # Append bot response to chat history
    chat_history.append({"role": "Bot", "message": bot_response})

    return jsonify({"bot_response": bot_response, "chat_history": chat_history})

if __name__ == '__main__':
    app.run(debug=True)
