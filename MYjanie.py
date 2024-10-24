# Importing necessary libraries
import streamlit as st
from langchain.chains import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Setting up Streamlit application
st.title("Identifying Red Flags in a SAAS Document")
st.write("Upload a PDF to see the identified red flags.")
temperature = st.slider("Set your temperature", 0.0, 1.0, 0.7)

# Initialize the LLM with Groq API key
groq_api_key = st.secrets['GROQ_API_KEY']
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=temperature)

# Uploading the PDF documents
uploaded_documents = st.file_uploader("Choose PDF files to upload", type="pdf", accept_multiple_files=True)

# Processing uploaded documents
if uploaded_documents:
    documents = []
    for uploaded_document in uploaded_documents:
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, 'wb') as file:
            file.write(uploaded_document.getvalue())
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=600)
    chunks = text_splitter.split_documents(documents)

    # Define the system prompt with relevant instructions for the assistant
    system_prompt = (
        "You are an assistant specialized in flagging redlines for the customer. "
        "Your task is to help people make informed decisions based on the information provided. "
        "If there are things in the document that should be known to the customer for an informed decision, highlight them. "
        "Highlight only those terms that are not in the customer's favor or must be known for making an informed decision."
    )

    # Create a prompt for the QA chain
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{context}")]
    )

    # Create a question-answer chain for document processing
    question_answer_chain = create_stuff_documents_chain(
        llm=llm, prompt=qa_prompt, document_variable_name="context"
    )

    # Extract important information from document chunks
    processed_documents = []
    for chunk in chunks:
        doc = [Document(page_content=str(chunk.page_content))]
        response = question_answer_chain.invoke({"context": doc})
        processed_documents.append(response)

    # Clean up the processed responses
    cleaned_docs = [Document(page_content=str(doc)) for doc in processed_documents]

    # Set the word limit for summarization
    word_limit = 8000 // len(processed_documents)

    # Create the template for summarizing individual chunks
    chunks_prompt = (
        f"Please summarize the below Terms and Condition Document in less than {word_limit} words:\n"
        "Document: '{text}'\n"
        "Summary:"
    )

    # Template for the map prompt
    map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

    # Template for combining the summaries
    final_prompt = (
        "Provide a final summary of the entire Terms and Conditions document, "
        "highlighting the top 20 critical points the customer should consider for making an informed decision.\n"
        "Document: {text}"
    )
    final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)

    # Load the summarization chain using map-reduce
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=final_prompt_template,
        verbose=True
    )

    # Generate the final output summary
    output_summary = summary_chain.run(cleaned_docs)

    # Display the summarized output
    st.write(output_summary)
