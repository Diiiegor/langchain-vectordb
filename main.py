import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()
    query = "What is pinecone in machine learning ?"

    vector_store = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(),
                                             combine_docs_chain=combine_docs_chain)

    result = retrieval_chain.invoke(input={"input": query})
    print(result)
