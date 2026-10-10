from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import PromptTemplate


PROMPT = """You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and predict if the claim should be accepted or not. Use the following guidelines for your analysis.

1.Verify if the patient has provided all necessary information and all necessary documents
and if you find any incomplete information or required documents are not provided then set INFORMATION criteria as FALSE and REJECT the claim.
if patient has provided all required documents then set INFORMATION criteria as TRUE. 

2. If any disease mentioned in the medical bill of the patient is in the general exclusions list, set EXCLUSION criteria as FALSE and REJECT the claim.

Use this information to verify if the application is valid and to accept or reject the application.

DOCUMENTS FOR CLAIM APPROVAL: {claim_approval_context}
EXCLUSION LIST : {general_exclusion_context}
PATIENT INFO : {patient_info}
MEDICAL BILL : {medical_bill_info}

Use the above information to verify if the application is valid and decide if the application has to be accepted or rejected keeping the guidelines into consideration. 

Generate a detailed report about the claim and procedures you followed for accepting or rejecting the claim and the write the information you used for creating the report. 
Create a report in the following format

Write whether INFORMATION AND EXCLUSION are TRUE or FALSE 
Reject the claim if any of them is FALSE.
Write whether claim is accepted or not. If the claim has been accepted, the maximum amount which can be approved will be {max_amount}

Executive Summary
[Provide a Summary of the report.]

Introduction
[Write a paragraph about the aim of this report, and the state of the approval.]

Claim Details
[Provide details about the submitted claim]

Claim Description
[Write a short description about claim]

Document Verification
[Mentions which documents are submitted and if they are verified.] 

Document Summary
[Give a summary of everything here including the medical reports of the patient]

Please verify for any signs of fraud in the submitted claim if you find the documents required for accepting the claim for the medical treatment.
"""


prompt = PromptTemplate(input_variables=["claim_approval_context", "general_exclusion_context", "patient_info","max_amount"], template=PROMPT)

def check_claim_rejection(claim_reason, general_exclusion_list, prompt_template, threshold=0.4):
    vectorizer = CountVectorizer()
    patient_info_vector = vectorizer.fit_transform([claim_reason])

    for disease in general_exclusion_list:
        disease_vector = vectorizer.transform([disease])
        similarity = cosine_similarity(patient_info_vector, disease_vector)[0][0]
        if float(similarity) > float(threshold):
            
            prompt_template = """You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and give a good rejection. You the following guidelines for your analysis.
            PATIENT INFO : {patient_info}

            Executive Summary
                [Provide a Summary of the report in very short.]

                Introduction
                [Write a paragraph about the aim of this report, and the state of the approval in short.]

                Claim Details
                [Provide details about the submitted claim]

                Claim Description
                [Write a short description about claim]

                Document Verification
                [Mentions which documents are submitted and if they are verified in the following format-
                 1)Consultation Date-A green dot if verified or a red dot , repeat the same for others] 

                Document Summary
                [Give a summary of everything here including the medical reports of the patient in brief]
            
            CLAIM MUST BE REJECTED: Patient has {disease} which is present in the general exclusion list."""
            return prompt_template
    
    return prompt_template