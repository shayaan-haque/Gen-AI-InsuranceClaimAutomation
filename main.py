# Import necessary libraries
from flask import Flask, render_template, request
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from services.invoice_extractor import get_file_content, get_bill_info
from services.retrieval import get_claim_approval_context, get_general_exclusion_context
from services.llm_chain import check_claim_rejection,PROMPT
import json
import markdown
from langchain_groq import ChatGroq


# Initialize the OpenAI client with the API key
FAISS_PATH = "/faiss"

# Flask App
app = Flask(__name__)

general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease","pregnancy", "substance abuse", "self-inflicted injuries", "sexually transmitted diseases(std)", "pre-existing conditions"]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def msg():
    claim_validation_message = ""
    name = request.form['name']
    address = request.form['address']
    claim_type = request.form['claim_type']
    claim_reason = request.form['claim_reason']
    date = request.form['date']
    medical_facility = request.form['medical_facility']
    medical_bill = request.files['medical_bill']
    total_claim_amount = request.form['total_claim_amount']
    description = request.form['description']

    bill = get_file_content(medical_bill)

    bill_info = get_bill_info(bill)
    # If input amount is more than the bill amount - REJECT
    if bill_info['expense'] != None and int(bill_info['expense']) < int(total_claim_amount) :
        claim_validation_message = "The amount mentioned for claiming is more than the billed amount. Claim Rejected."
        
        return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, total_claim_amount=total_claim_amount, description=description, output=claim_validation_message)
        
    elif bill_info['expense'] != None and int(bill_info['expense']) >= int(total_claim_amount) :
        #Check if the disease is in the exclusion list or not, update the prompt accordingly
        patient_info = f"Name: {name} " + f"\nAddress: {address} " + f"\nClaim type: {claim_type} " + f"\nClaim reason: {claim_reason}" + f"\nMedical facility: {medical_facility} " + f"\nDate : {date} " + f"\nTotal claim amount: {total_claim_amount}" + f"\nDescription: {description}"
        medical_bill_info = f"Medical Bill: {bill}"
        
        validated_prompt = check_claim_rejection(bill_info["disease"], general_exclusion_list,PROMPT)
    
        prompt_template = PromptTemplate(input_variables=["claim_approval_context","patient_info"],template=validated_prompt)
        llm = ChatGroq(model="qwen/qwen3.8-27b")
        llmchain = LLMChain(llm=llm, prompt= prompt_template)
        output = llmchain.run({"claim_approval_context": get_claim_approval_context(), "general_exclusion_context": get_general_exclusion_context(), "patient_info": patient_info, "medical_bill_info":medical_bill_info,"max_amount":total_claim_amount, "disease":bill_info["disease"]})
        
        output = markdown.markdown(output, extensions=['nl2br'])
        
        return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, total_claim_amount=total_claim_amount, description=description, output=output)
        
    else:
        #If no expense value has been extracted
        output = "Please enter a valid Consultation Receipt."
        
        return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, total_claim_amount=total_claim_amount, description=description, output=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)

