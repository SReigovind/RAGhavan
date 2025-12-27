# Import necessary libraries
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Step 1: Load the Document ---
pdf_path = "documents/paper.pdf"

# Using PyMuPDFLoader which handles columns better
loader = PyMuPDFLoader(pdf_path)

# This loads the PDF and creates a list of "pages"
docs = loader.load()

print(f"Successfully loaded {len(docs)} pages from the PDF.")

# --- Step 2: Split the Text ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)

print(f"Split the document into {len(splits)} chunks.")

print("\n--- First Chunk Preview ---")
print(splits[0].page_content)
print("---------------------------")

# --- Step 3: Create Embeddings and Vector Store ---
print("\nInitializing embedding model... (this might take a moment)")
# We use a pre-trained model to convert text into numbers (vectors)
# "all-MiniLM-L6-v2" is a small, fast, and effective model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vector store
# This takes our text chunks, converts them to vectors using the model,
# and indexes them for fast search.
print("Creating vector store...")
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

print("Successfully created the vector store.")

# --- Step 4: Test the Retrieval ---
# A sample query to see if it finds relevant chunks
query = "What is this paper about?"
docs = vectorstore.similarity_search(query)

print(f"\n--- Query: {query} ---")
print(f"Found {len(docs)} relevant chunks.")
print("\n--- Top Result ---")
print(docs[0].page_content)
print("------------------")