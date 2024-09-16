import sys
import dotenv
from langchain.schema.runnable import RunnablePassthrough

from langchain_groq import ChatGroq
from langchain_core.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser

from retriever import reviews_retriever

print(sys.executable)
dotenv.load_dotenv()

# chat model initialization
model = ChatGroq(model="llama3-8b-8192", temperature=0.5, max_tokens=50)

# 1. Chains and Langchain Expression Language (LCEL)
# A chain is a sequence of calls between objects in Langchain
# We build chains using the LCEL

review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

output_parser = StrOutputParser()

# Chain "review_prompt_template" and "model" together
# Pass retriever to the chain, so that relevant reviews are passed to the prompt as context
# Assigning question to a RunnablePassthrough object ensures the question gets passed unchanged to the next step in the chain.
review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()},
    review_prompt_template | model | output_parser
)


# 2. Retrieval Objects
# Layer another object into review_chain to retrieve documents from a vector database
# The process of retrieving relevant docs using a "Retriever" and passing them to an LLM is called RAG

