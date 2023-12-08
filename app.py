import PyPDF2
from transformers import pipeline
import gradio as gr
from gradio import File, Textbox

import io
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_file_bytes):
    try:
        # Convert bytes to a file stream
        pdf_file_stream = io.BytesIO(pdf_file_bytes)
        text = extract_text(pdf_file_stream)
        if not text:
            print("No text extracted from the PDF.")
            return None
        return text
    except Exception as e:
        print("Error in extract_text_from_pdf with PDFMiner: ", e)
        return None






# Initialize the question-answering pipeline
qa_pipeline = pipeline('question-answering', model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad")

def answer_question(uploaded_file, question):
    try:
        if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)
            if text:
                result = qa_pipeline(context=text, question=question)
                answer = result['answer']
                print("Answer: ", answer)  # Debug print
                return answer
            else:
                return "Failed to extract text from PDF."
        else:
            return "No PDF file uploaded."
    except Exception as e:
        print("Error in answer_question: ", e)  # Print any errors encountered
        return "An error occurred while processing the request."

iface = gr.Interface(
    fn=answer_question,
    inputs=[File(label="Upload PDF", type="binary"), Textbox(label="Question")],
    outputs="text",
    title="PDF Question Answerer",
    description="Upload a PDF and ask questions about its content."
)


if __name__ == "__main__":
    iface.launch()




