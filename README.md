# StepsAI
Overview
This project extracts text from PDF files, chunks the text into manageable pieces, embeds these chunks using Sentence-BERT, and stores the embeddings in a Milvus vector database for efficient retrieval and similarity search.

Prerequisites
To run this project, you need the following software and libraries installed:

Python 3.6 or higher
PyPDF2
NLTK
SentenceTransformers
Scikit-learn
pymilvus
Milvus
You can install the necessary Python libraries using pip:

sh
Copy code
pip install PyPDF2 nltk sentence-transformers scikit-learn pymilvus
Additionally, ensure that you have a running instance of Milvus. You can follow the Milvus installation guide to set it up.

Project Structure
css
.
├── main.py
├── requirements.txt
└── README.md
main.py: The main script that extracts text from PDFs, processes it, and stores the embeddings in Milvus.
requirements.txt: A file listing the Python dependencies.
README.md: This file, providing an overview and instructions.
Detailed Steps
Extract Text from PDF
The function extract_text_from_pdf(file_path) reads the PDF file and extracts text from each page.

Chunk Text
The function chunk_text(text, max_tokens=100) chunks the extracted text into pieces with approximately 100 tokens each, ensuring sentence boundaries are preserved.

Embed Texts
The function embed_texts(texts) uses Sentence-BERT to embed the chunks of text into high-dimensional vectors.

Milvus Operations
Create Collection: create_milvus_collection() connects to Milvus and creates a collection to store the vectors and metadata.
Insert Data: insert_into_milvus(collection, embeddings, chunks, book_title) inserts the embeddings, original text chunks, book titles, and page numbers into the Milvus collection.
Main Execution Flow
The main() function performs the following steps for each textbook:

Extracts the book title from the file path.
Extracts text from the PDF.
Chunks the text into manageable pieces.
Embeds the text chunks.
Inserts the embeddings and metadata into the Milvus collection.
