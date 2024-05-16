from transformers import pipeline
import streamlit as st

# Step 1: Load a pre-trained question answering model and tokenizer
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

context = """
Rakesh is a diligent, result-oriented, mild-mannered consulting and delivery professional.He has good business process 
knowledge and practical exposure. He always puts the customer first and provides solutions that benefit the people who 
run the business. Rakesh has worked with several Fortune listed MNCs in US, UK, Europe and Asia to deliver Business. He
is approachable, pragmatic and willing to hear every problem to provide long term fixes.His favourite quote is 
“A hammer shatters glass, but forges steel. The choice is yours!”. He strongly believes in Courage, Confidence, 
Love and Hard work. His LinkedIn address is  “https://www.linkedin.com/in/dineshkumara/“.
We are looking for candidates with good attitude, knowledge and skill in your areas of Specialization. 
Exceptional communication, both written and oral are most welcome. We believe in Employee-First-Entrepreneur-Next philosophy. 
So groom our Employees as business partners who add value to us and their life, with flexible work-life balance.
We have a NO-NONSENSE policy, intolerance towards SPAMMERS and unsolicited senders. Your email addresses are SECURE with
us. We do not sell your contact details to ANY O!NE ! Be Assured! YOUR BUSINESS CONFIDENTIAL DATA 
IS TREAT WITH HIGH LEVEL OF SECURITY AND ACCESSED ON A NEED-ONLY BASIS.This should be a floating icon prominently 
placed on the middle of the screen."""

st.title("Query...")
question = st.text_input("Type the text here : ")
if st.button("Search"):
    result = qa_pipeline(question=question, context=context)
    st.write("Answer:", result['answer'])

