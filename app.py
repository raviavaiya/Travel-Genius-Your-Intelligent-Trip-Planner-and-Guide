from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

app = Flask(__name__)

# Define the prompt template for coding questions
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Your current task is to answer Travel-related questions clearly and concisely."),
        ("user", "Question: {question}")
    ]
)

# Initialize the LLM and output parser
llm = Ollama(model="mistral:7b-instruct-v0.3-q2_K")
output_parser = StrOutputParser()

@app.route('/')
def index():
    return render_template('index.html')  # Ensure your HTML is updated for coding questions

@app.route('/answer', methods=['POST'])  # Changed endpoint to /answer
def answer():
    input_text = request.form['input_text']
    if not input_text.strip():
        return jsonify({"error": "Please enter a coding question."})

    try:
        # Create the chain
        chain = prompt | llm | output_parser

        # Invoke the chain with user input
        raw_answer = chain.invoke({"question": input_text})

        # Format the answer for line-by-line output
        formatted_answer = format_answer_line_by_line(raw_answer)
        return jsonify({"answer": formatted_answer})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})

def format_answer_line_by_line(answer):
    """
    Format the LLM's response to display each line separately.
    - Splits the response by newlines.
    - Wraps each line in HTML <p> tags for display.
    """
    lines = answer.strip().split('\n')
    formatted_lines = ''.join(f"<p>{line.strip()}</p>" for line in lines if line.strip())
    return formatted_lines

if __name__ == '__main__':
    app.run(debug=True)
