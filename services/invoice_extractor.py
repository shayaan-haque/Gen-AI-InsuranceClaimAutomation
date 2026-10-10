from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
import os, re
from dotenv import load_dotenv
import json
load_dotenv()

api_key = os.environ['GROQ_API_KEY'] 

if api_key is None or api_key == "":
    print("Groq API key not set or empty. Please set the environment variable.")
    exit()  # Terminate the program if the API key is not set.


def get_file_content(file):
    text = ""
    if file.filename.endswith(".pdf"):
        pdf = PdfReader(file)
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text()

    return text

def get_bill_info(data):
    prompt = "Act as an expert in extracting information from medical invoices. You are given with the invoice details of a patient. Go through the given document carefully and extract the 'disease' and the 'expense amount' from the data. Return the data in json format = {'disease':"",'expense':""}"
    messages=[
        {"role": "system", 
        "content": prompt}
        ]
    
    user_content = f"INVOICE DETAILS: {data}"

    messages.append({"role": "user", "content": user_content})

    llm = ChatGroq(
                model="qwen/qwen3.8-27b",
                temperature=0.4,
                max_tokens=250)

    response = llm.invoke(messages)
        
    # data = json.loads(response.content) 

    content = response.content.strip()
    
    # Strip markdown code blocks if the LLM added them
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
        
    # Find the start and end of the actual JSON object to ignore conversational text
    try:
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        clean_json_string = content[start_idx:end_idx]
        
        data = json.loads(clean_json_string)
    except json.JSONDecodeError:
        # Fallback if the LLM completely fails to generate JSON
        data = {"disease": None, "expense": None}

    return data
