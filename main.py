from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model = "llama3.2")

template = """
You are an expert in answering questions about review of 100 restaurants

Here are some relevant reviews: {reviews}

Let's go to with my response: {question}
"""
while True:
    print("\n\n---------------------------------------------")
    question = input("What do you want to know?: ")
    print("---------------------------------------------")

    if question == 'q':
        break
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    reviews = retriever.invoke(question)
    result = chain.invoke({ "reviews": reviews,"question" : question})

    print(result)