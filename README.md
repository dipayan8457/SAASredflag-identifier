# Identifying Red Flags in SaaS Documents

## Project Description
In the rapidly evolving landscape of Software as a Service (SaaS), customers often encounter complex Terms and Conditions (T&C) that can significantly impact their business operations. This project aims to develop an AI-driven application that identifies critical red flags within SaaS agreements, empowering customers to make informed decisions before entering contracts. By leveraging natural language processing and machine learning, the application streamlines the analysis of legal documents, ensuring that users are aware of potential risks and obligations.

## Aim
The primary objective of this project is to create a user-friendly application that can:
- Automatically extract relevant information from uploaded SaaS T&C documents.
- Identify and flag critical terms that may pose risks or disadvantages to the customer.
- Generate concise summaries highlighting key points, facilitating easier comprehension and decision-making.

## Working
The application is built using **Streamlit** for the web interface and **LangChain** for processing and analyzing the text. Here’s a breakdown of its functionality:

1. **PDF Upload**: Users can easily upload multiple PDF files containing SaaS agreements for analysis.
2. **Document Processing**: 
   - The application utilizes the `PyPDFLoader` to extract text from the uploaded PDFs.
   - Text is divided into manageable chunks using the `RecursiveCharacterTextSplitter` to facilitate efficient processing.
3. **Red Flag Identification**: 
   - A pre-defined system prompt guides the language model to analyze the document content and flag terms that are critical for customers to consider.
   - The application specifically looks for clauses that might be unfavorable to the customer, such as restrictive licensing terms, payment conditions, service level agreements, and data privacy clauses.
4. **Summary Generation**: 
   - The application summarizes the findings, highlighting the top 20 essential points that customers should be aware of before signing the agreement.
   - The summary helps users quickly grasp the most important aspects of the contract.
5. **User Interface**: The app presents the identified risks and generated summaries in a clean, user-friendly interface, enhancing the overall user experience.

## Datasets and Clauses
- The model utilizes various SaaS T&C documents as input for training and evaluation, ensuring it learns from a diverse set of agreements.
- The analysis is based on established guidelines and checklists that outline key clauses and potential risks:
  - [Key Clauses in a SaaS Agreement](https://www.cloudeagle.ai/blogs/key-clauses-you-should-not-miss-in-a-saas-agreement)
  - [Top Legal Issues in a SaaS Agreement](https://www.outsidegc.com/blog/top-15-legal-issues-in-a-saas-agreement)
  - [SaaS Agreement Checklist](https://www.cloudeagle.ai/blogs/saas-agreement-checklist)

## Steps
1. **Setup Environment**: Configure the necessary libraries and API keys, ensuring all dependencies are in place for the application to run smoothly.
2. **Upload Document**: Users can upload their SaaS T&C PDFs directly through the web interface.
3. **Process Document**: The application extracts and splits the text into manageable chunks for detailed analysis.
4. **Analyze Document**: The language model identifies red flags based on the context and predefined criteria.
5. **Display Results**: The findings and summaries are presented to the user, allowing for quick reference and understanding.

## Conclusion
This project demonstrates the application of AI in enhancing legal document reviews for SaaS agreements. By providing a streamlined process for identifying red flags and summarizing critical points, the application aims to reduce the time and costs associated with manual document analysis, ultimately improving customer awareness and decision-making.

## Next Steps
- Enhance model accuracy by incorporating more training data from diverse SaaS agreements.
- Expand functionality to support additional document formats, such as Word documents and text files.
- Integrate user feedback mechanisms to continuously improve the application's performance and usability.

---
