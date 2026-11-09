from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain import hub
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langsmith import evaluate, Client
from dotenv import load_dotenv

# load environment variables
load_dotenv()


# load docs from FAISS
embeddings = OpenAIEmbeddings()

persist_path="rag_embeddings"

vector_store = FAISS.load_local(
    folder_path=persist_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
    ) 

# langgraph app

llm = ChatOpenAI(model_name = 'gpt-4o')

prompt = PromptTemplate(
        template="""
        You are an expert chat assistant designed to answer questions based on provided context.
        Carefully use the context below to answer the question. If the context does not contain enough information,
        respond clearly and politely that you do not know the answer.

        Examples:
        Context: The store is located in Manchester, UK on 29th Avenue Street.
        Question: Which city is the store located in?
        Answer: The dtore is located in Manchester.

        Context: No information about dtore location is provided.
        Question: Which city is the store located in?
        Answer: I'm sorry, I don't have enough information to answer that question.

        Context: {context}
        Question: {question}

        Answer:""",
        input_variables=['context', 'question']
    )

# state
class RagState(TypedDict):
    question: str
    context: List[Document]
    answer: str


# nodes
def retrieve(state: RagState):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: RagState):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# define graph

graph = StateGraph(RagState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()


# test run

input_state = {"question": "What's the price of Americano?"}

response = workflow.invoke(input_state)
print(response["answer"])


"""def run_rag(inputs: dict) -> dict:

    #Run the RAG workflow on a dataset row (dict with at least a 'question' key)
    #and return a dict with 'answer' for evaluation.
    
    question = inputs.get("question")
    input_state = {"question": question}
    response = workflow.invoke(input_state)
    return {"answer": response["answer"]}



# 1. Create and/or select your dataset
client = Client()
dataset_name = "rag-cafe-dataset-duplicate"

# 2. Define an evaluator
def exact_match(outputs: dict, reference_outputs: dict) -> bool:
    return outputs == reference_outputs


# 3. Run an evaluation
# For more info on evaluators, see: https://docs.langchain.com/langsmith/evaluation-concepts

# To evaluate an LCEL chain, replace lambda with chain.invoke
# To evaluate a LangGraph graph, replace lambda with graph.invoke
evaluate(
    run_rag,
    # chain.invoke
    # graph.invoke
    data=client.list_examples(
        dataset_name=dataset_name, 
        example_ids=[   # We pass in a specific list of example_ids
            # TODO: You will need to paste in your own example ids for this to work!
            "2596fdff-a187-4c87-a851-2e7792177723",
            "fcdc9959-5ae4-4176-973d-47e14faaa035"
        ]
    ),
    evaluators=[exact_match],
    experiment_prefix="cafe-rag-datatset 2"
)"""