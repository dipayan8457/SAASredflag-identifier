#Importing neccesary libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import StuffDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tabula
import os

#Setting up our env
# from dotenv import load_dotenv
# load_dotenv()

#Getting the Hugging Face Token and GroqApiKey
# os.environ['HF_TOKEN']=st.secrets['HF_TOKEN']
groq_api_key=st.secrets['GROQ_API_KEY']
# embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Setting up streamlit
st.title("Identifying Red flags in a SAAS document")
st.write("Upload pdf to see the red flags")
temperature=st.slider("Set your temperature as you require",0.0,1.0,0.7)

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-70b-versatile",temperature=temperature)

#Uploading the pdf
uploaded_documents=st.file_uploader("Choose A Pdf file to upload",type="pdf",accept_multiple_files=True)
if uploaded_documents:
    documents=[]
    for uploaded_document in uploaded_documents:
        temp_pdf=f"./temp.pdf" 
        with open(temp_pdf,'wb') as file:
            file.write(uploaded_document.getvalue())
            file_name=uploaded_document.name
        loader=PyPDFLoader(temp_pdf)
        docs=loader.load()
        documents.extend(docs)



    text_splitter3=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=600)
    chunks3=text_splitter3.split_documents(documents) 
    # vector_store=FAISS.from_documents(documents=chunks3,embedding=embeddings)
    # faiss_semantic_retriever=vector_store.as_retriever()
    # bm25_retriever = BM25Retriever.from_documents(documents=chunks3)
    # retriever = EnsembleRetriever(retrievers=[bm25_retriever,faiss_semantic_retriever],
    #                                    weights=[0.5,0.5])

    system_prompt = (
                "You are an assistant specialized in flaging the redline for the customer"
                "Your task is to help people make an informed decision"
                "You will be given a piece of information and you will be required to tell what are the things that a customer should be aware of before signing the deal"
                "If you find thing that should be known to the customer for making an informed decision just output that"
                "Output only those things from the terms and condition document that you think must be known to the customer for making an informed decision or things that are not in favour of the customer"
                "Below are some things which are in favour of the customer"
                "\n"
                "License Scope-A broader scope, to include possible use by subsidiaries, affiliates and contractors. Fewer license restrictions, that are fair and reasonable. "
                "Payment Term-Payment in arrears. Longer payment terms with right to dispute payments in good faith (e.g., net 60 after receipt of undisputed invoice). Avoid interest and penalties; or minimize their impact via a written notice requirement and cure periods before any interest or penalties can begin."
                "Service level agreement (SLA)- Robust SLAs, including a right to service credits or refunds for excessive downtime, accessible support channels as well as a right to terminate after a certain number (or length) of incidents."
                "Use of Data/Data Rights-Retain all rights to its data; or grant limited rights to vendor for the use aggregated and anonymized data only."
                "DPA (data privacy addendum)-DPA that requires prompt vendor notice (e.g., 48 hours) in the event of not only an actual security breach, but also any suspected or alleged security breaches; quick remediation (at vendor expense); termination rights for customer; and indemnity for security breach with either unlimited liability or a higher super-cap."
                "Reps and Warranties-Standard, but broader, vendor reps and warranties, e.g., that vendor will comply with applicable laws and industry standards, confidentiality and privacy protections, IP rights (non-infringement), etc."
                "Indemnities-No indemnities given to vendor; or give indemnities with a narrow scope (and include exceptions for modification or misuse of your content or data). Robust indemnities from vendor (e.g., non-infringement, confidentiality & privacy, injury to persons or property, arising from any material breach, etc.)."
                "Limitation on Liability-Uncapped vendor liability, if possible, especially for issues such as indemnities, IP violations, and confidentiality or privacy breaches. May accept super caps if they are reasonable, based on the scope of possible harm, not necessarily proportional to the size of the deal"
                "Termination Rights-Broad termination rights (e.g., due to vendor breach, SLA failures, privacy issues, decrease in service features or functionality, chronic issues, and, if possible, for convenience); with rights to pro-rata refund, if possible."
                "Renewal-Auto renewal may be acceptable, but only with reasonable opt out dates for customer to avoid paying for an unwanted renewal term."
                "Notice Period-Preferred length of notice periods and timelines also varies. Shorter notice requirements for things relating to customer rights. Longer notice periods for any provisions giving the vendor a right to pursue remedies against customer."
                "Insurance-Vendor insured for general liability, errors & omissions/professional liability, cyber liability, and workmen's comp. Plus, an umbrella policy and other applicable coverage based on circumstances (car, shipping, air, etc.)."
                "Pubilicity-Right to approve any use of customer name or logos, including prior approval of use in lists of clients."
                "Assignment-Mutual restriction of assignment, with a mutual exception for Mergers and Acquisitions activity or reorganizations."
                "Pricing Plan-A nominal or economical pricing plan"
                "Others-If asked to sign vendor template contract, review for non-standard terms to avoid, such as Exclusivity ,non-solicitation clauses,Liens and security interests or anything else unusual or non-standard."
            )
    qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{context}"),
                ]
            )
    # qa_chain = load_qa_chain(llm, chain_type="stuff")
    question_answer_chain=create_stuff_documents_chain(llm=llm,prompt=qa_prompt,document_variable_name="context")
    Whole_document=[]
    for i in range(len(chunks3)):
        doc=[Document(page_content=str(chunks3[i].page_content))]
        response = question_answer_chain.invoke({"context": doc})
        Whole_document.append(response)
    cleandocs=[]
    for i in range(len(Whole_document)):
        cleandocs.append(Document(page_content=str(Whole_document[i])))
    # schain=load_summarize_chain(
    # llm=llm,
    # chain_type="refine",
    # verbose=True    
    # )
    # output_summary=schain.run(cleandocs)
    # text=""

    val=8000//len(Whole_document)

    chunks_prompt="""
    Please summarize the below Terms and Condition Document in less than X words:
    Document:'{text}'
    Summary:

    """

    list1=[elm for elm in chunks_prompt]
    for i in range(len(list1)):
        if list1[i]=='X':
            list1[i]=val
    chunks_prompt1="".join([str(elm) for elm in list1])



    map_prompt_template=PromptTemplate(input_variables=['text'],template=chunks_prompt1)
    final_prompt='''
    Provide the final summary of the entire Terms and conditions that should be highlighted to the cutomer for taking an informed and good decision keeping the most important 20 solid points.
    Document:{text}

    '''
    final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)
    summary_chain=load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=final_prompt_template,
    verbose=True
    )
    output_summary=summary_chain.run(cleandocs)

    st.write(output_summary)
    # st.write(chunks_prompt)
