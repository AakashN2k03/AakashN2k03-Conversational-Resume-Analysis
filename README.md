# Conversational-Resume-Analysis
This project utilizes Streamlit to create a web application that facilitates a conversational analysis of uploaded resumes. It leverages Google Generative AI models to process and answer your questions about the resume content.

# Key Features:
Conversational Interface: Ask questions in natural language to gain insights from the resume.
Multiple Analysis Options:
Summarize the resume based on a provided job description.
Calculate the percentage match between the resume and the job description.
Extract relevant projects from the resume.
Identify internship experiences related to the job description.
PDF Support: Upload PDF resumes for processing.
  
# Requirements:
Python 3.x
Streamlit (pip install streamlit)
PyPDF2 (pip install PyPDF2)
langchain (pip install langchain)
Google Generative AI client library (pip install google-generativeai)
dotenv (pip install python-dotenv)

# Setup:
Create a virtual environment (recommended).
Install the required libraries (pip install -r requirements.txt).
Access gemini apikey.
Set the GOOGLE_API_KEY environment variable using dotenv
Create a .env file in your project directory.
Add the line GOOGLE_API_KEY=<your_api_key_here>.
Run the application using streamlit run main.py.

# Usage:
Open the application in your web browser (usually http://localhost:8501).
Upload your PDF resume(s) using the file uploader.
Click the "Submit & Process" button to generate the vector index.
Enter the job description in the "Job Description" text area.
Use the buttons provided to perform different analysis tasks on the resume:
Summarize the resume
Calculate percentage match
Extract projects
Identify internships
Ask questions in natural language related to the resume and job description. The application will attempt to answer them using the processed information.

# Additional Notes:
This is a basic implementation and can be further customized.
Consider adding error handling and input validation.
Explore more advanced prompt engineering techniques for improved conversational analysis.

# Reference
Consider mentioning what I have learnt from Krish Naik's tutorial to personalize the reference and show appreciation for his content.
