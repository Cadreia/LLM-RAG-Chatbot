import dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "data/chroma_data"

# 1. Create Review Embeddings
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(reviews, OpenAIEmbeddings, persist_directory=REVIEWS_CHROMA_PATH)

# Note: In practice, if you’re embedding a large document, you should use a text splitter. Text splitters break the document into smaller chunks before running them through an embedding model. This is important because embedding models have a fixed-size context window, and as the size of the text grows, an embedding’s ability to accurately represent the text decreases.
# For this example, you can embed each review individually because they’re relatively small.

# 2. Perform Semantic Search over the Review Embeddings
# If you have to reconnect to the vector db,
# reviews_vector_db = Chroma(
#   persist_directory=REVIEWS_CHROMA_PATH,
#   embedding_function=OpenAIEmbeddings()
# )
question = """Has anyone complained about
...            communication with the hospital staff?"""
relevant_docs = reviews_vector_db.similarity_search(question, k=3)

print(relevant_docs[0])


# create reviews_retriever by calling .as_retriever() on reviews_vector_db to create a retriever object that you’ll add to review_chain
reviews_retriever = reviews_vector_db.as_retriever(k=10)