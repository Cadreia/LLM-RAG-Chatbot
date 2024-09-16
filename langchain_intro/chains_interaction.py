from chains import review_chain

import sqlite3
print(sqlite3.sqlite_version)

# context = "I had a great stay!"
# question = "Did anyone have a positive experience?"
#
# response = review_chain.invoke({"context": context, "question": question})

question = "Did anyone have a terrible experience?"
response = review_chain.invoke(question)
print(response)
