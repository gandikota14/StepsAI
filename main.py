import PyPDF2
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, MilvusException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Extract content from a PDF file
def extract_text_from_pdf(file_path):
    pdf_reader = PyPDF2.PdfFileReader(open(file_path, "rb"))
    text = ""
    for page_num in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page_num).extract_text()
    return text

# Chunk text into approximately 100 tokens each, preserving sentence boundaries
def chunk_text(text, max_tokens=100):
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) > max_tokens:
            chunks.append(chunk.strip())
            chunk = sentence
        else:
            chunk += " " + sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Embed texts using Sentence-BERT
def embed_texts(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

# Connect to Milvus and create collection
def create_milvus_collection():
    try:
        connections.connect(host='localhost', port='19530')
        collection_name = 'textbook_vectors'

        fields = [
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="book_title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page_number", dtype=DataType.INT64)
        ]

        schema = CollectionSchema(fields, "Textbook vector storage")
        collection = Collection(name=collection_name, schema=schema)
        return collection

    except MilvusException as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        raise

# Insert data into Milvus collection
def insert_into_milvus(collection, embeddings, chunks, book_title):
    entities = [
        {"name": "embeddings", "values": embeddings.tolist()},
        {"name": "text", "values": chunks},
        {"name": "book_title", "values": [book_title] * len(chunks)},
        {"name": "page_number", "values": list(range(1, len(chunks) + 1))}
    ]
    collection.insert(entities)

def main():
    # Paths to the textbooks
    textbook_paths = [
        r'D:\Novels-20210503T110151Z-001\Lord Of The Rings\The two towers.pdf',
        r'D:\Novels-20210503T110151Z-001\Lord Of The Rings\The Return of the King.pdf',
        r'D:\Novels-20210503T110151Z-001\Lord Of The Rings\The Hobbit.pdf'
    ]

    collection = create_milvus_collection()

    for path in textbook_paths:
        # Extract book title from path
        book_title = path.split('\\')[-1]
        
        # Extract, chunk, and embed content
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)
        
        # Insert into Milvus
        insert_into_milvus(collection, embeddings, chunks, book_title)

if __name__ == "__main__":
    main()
