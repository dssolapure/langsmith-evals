from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Define RAG pipeline builder
def build_rag_pipeline(persist_path="rag_embeddings", model_name="gpt-4o"):
    """Builds and returns a compiled LangGraph workflow for RAG."""

    embeddings = OpenAIEmbeddings()

    # Load FAISS vector store
    vector_store = FAISS.load_local(
        folder_path=persist_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI(model_name=model_name)

    prompt = PromptTemplate(
        template="""
        You are an expert chat assistant designed to answer questions based on provided context.
        Carefully use the context below to answer the question. If the context does not contain enough information,
        respond clearly and politely that you do not know the answer.

        Examples:
        Context: The store is located in Manchester, UK on 29th Avenue Street.
        Question: Which city is the store located in?
        Answer: The store is located in Manchester.

        Context: No information about store location is provided.
        Question: Which city is the store located in?
        Answer: I'm sorry, I don't have enough information to answer that question.

        Context: {context}
        Question: {question}

        Answer:""",
        input_variables=['context', 'question']
    )

    # State definition
    class RagState(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Nodes
    def retrieve(state: RagState):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: RagState):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Define graph
    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# Exposed function
def rag_answer(question: str, persist_path="rag_embeddings", model_name="gpt-4o") -> str:
    """Takes a question and returns the RAG-generated answer."""
    workflow = build_rag_pipeline(persist_path, model_name)
    response = workflow.invoke({"question": question})
    return response["answer"]


# Example usage (only runs if file executed directly, not on import)
if __name__ == "__main__":
    answer = rag_answer("Give me a contact number if possible.")
    print(answer)
