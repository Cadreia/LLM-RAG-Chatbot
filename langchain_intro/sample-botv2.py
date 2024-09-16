import sys
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

print(sys.executable)
dotenv.load_dotenv()

# 1. Chat Models
model = ChatGroq(model="llama3-8b-8192", temperature=0.5, max_tokens=50)
messages = [
    SystemMessage("You're an assistant knowledgeable about healthcare. Only answer healthcare-related questions."),
    HumanMessage("How do I care for a wound?")
]
response = model.invoke(messages)
print(response.content)


# # 2. Prompt Templates
# from langchain_core.prompts import ChatPromptTemplate
#
# review_template_str = """Your job is to use patient reviews to answer questions about their experience at a hospital. Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know.
#
# {context}
#
# {question}
# """
#
# review_template = ChatPromptTemplate.from_template(review_template_str)
#
# context = "I had a great stay!"
# question = "Did anyone have a positive experience?"
#
# template_format = review_template.format(context=context, question=question)
#
# print(template_format)


# 3. Enhanced Prompt Template
from langchain_core.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate)

review_system_template_str = """Your job is to use patient reviews to answer questions about their experience at a hospital. Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["context"], template=review_system_template_str))

review_human_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["question"], template="{question}"))

messages = [review_system_prompt, review_human_prompt]
review_prompt_template = ChatPromptTemplate(input_variables=["context", "question"], messages=messages)

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

review_prompt_template.format_messages(context=context, question=question)
print(review_prompt_template.format_messages(context=context, question=question))
