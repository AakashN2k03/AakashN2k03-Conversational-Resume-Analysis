import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text

def get_pdf_text(pdf_docs):

    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

# 2 half
def user_input1(input_text):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(input_text)

    chain = get_conversational_chain1()

    
    response = chain(
        {"input_documents":docs, "question": input_text}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def get_conversational_chain1():

    prompt_template = """
    You, as a seasoned Technical Human Resource Manager, are responsible for assessing the suitability of a candidate's resume for a specific job role.
    Provide a professional evaluation of the candidate's alignment with the job description, emphasizing strengths and weaknesses relevant to the specified job requirements. 
    Ensure that the evaluation reflects the given job role and its associated expectations.

    ""\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
    
# 3 half

def user_input2(input_text):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(input_text)

    chain = get_conversational_chain2()

    
    response = chain(
        {"input_documents":docs, "question": input_text}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def get_conversational_chain2():

    prompt_template = """
    You are an experienced Technical Human Resource Manager tasked with evaluating a resume against a specific job description.
    Provide the percentage of alignment between the candidate's profile and the role. 
    If the candidate is valid, state whether they are a good fit for the role. 
    If invalid, advise them to improve their skills. Ensure the output is concise, limited to 3 lines.""\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

    
def user_input3(input_text):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(input_text)

    chain = get_conversational_chain3()

    
    response = chain(
        {"input_documents":docs, "question": input_text}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def get_conversational_chain3():

    prompt_template = """
   You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against a specific job description. Please list the projects in bullet points format. 
   Ensure that projects relevant to the job description are mentioned with bold letters. In other sections, provide projects from different domains. If no relevant projects are found, advise the candidate to work on projects aligned with the job role.
   ""\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input4(input_text):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(input_text)

    chain = get_conversational_chain4()

    
    response = chain(
        {"input_documents":docs, "question": input_text}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def get_conversational_chain4():

    prompt_template = """
   You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against job description.
   mention the intership company's name  he has done with related to the job decribed,if time duration mentioned,kindly mention it too.
""\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain





def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Resume 😎 ")
    

    user_question = st.text_input("Ask questions from the resume ")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
 


    st.header("Resume Analysis System")

    input_text = st.text_area("Job Description: ", key="input1")
     
    about = st.button("Summarize the resume with job described above",key="button1")
    if about:
         if input_text:
              user_input1(input_text)
 
    percentage = st.button("Percentage match",key="button2")
    if percentage:
         if input_text:
              user_input2(input_text)

    project = st.button("Projects",key="button3")
    if project:
         if input_text:
              user_input3(input_text)

    work = st.button("internships",key="button4")
    if work:
         if input_text:
              user_input4(input_text)


if __name__ == "__main__":
    main()