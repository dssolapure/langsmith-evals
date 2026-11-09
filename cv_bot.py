from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

#environment variables
load_dotenv()
llm = ChatOpenAI(model_name = 'gpt-4o')
#llm = ChatOllama(model="hf.co/MaziyarPanahi/gemma-3-12b-it-GGUF")


# define state
class CVState(TypedDict):
    cv_text: str
    jd_text: str
    cv_skills: str
    jd_skills: str
    missing_skills: str
 
# define functions

def extract_cv(state: CVState):
    messages = [
        SystemMessage(content="You are an expert at extracting technical skills from CVs."),
        HumanMessage(content=f"Extract the key technical skills from the following CV text:\n{state['cv_text']}. Output only the skills in bullet points, no extra words. ")
    ]

    response = llm.invoke(messages).content
    return {'cv_skills' : response}


def extract_jd(state: CVState):
    messages = [
        SystemMessage(content="You are an expert at extracting technical skills from Job Descriptions."),
        HumanMessage(content=f"Extract the key technical skills from the following Job Description text:\n{state['jd_text']}. Output only the skills in bullet points, no extra words.")
    ]

    response = llm.invoke(messages).content
    return {'jd_skills' : response}


def compare_skills(state: CVState):
    messages = [
        SystemMessage(content="You are an expert at comparing skills from CVs and Job Descriptions."),
        HumanMessage(content=f"Compare the following technical skills from a CV and a Job Description and identify skills that are missing in the CV. Output as bullet points. \n CV Skills: {state['cv_skills']} \n JD Skills: {state['jd_skills']}. Output only the skills in bullet points, no extra words.")
    ]

    response = llm.invoke(messages).content
    return {'missing_skills' : response}


# define graph
graph = StateGraph(CVState)

graph.add_node('cv', extract_cv)
graph.add_node('jd', extract_jd)
graph.add_node('compare', compare_skills)

graph.add_edge(START, 'cv')
graph.add_edge('cv', 'jd')
graph.add_edge('jd', 'compare')
graph.add_edge('compare', END)

workflow = graph.compile()


# input data

cv_text = """Data Analyst with experience in the finance domain, proficient in R, SQL, Python, Excel, and Power BI for extracting, analyzing, and visualizing complex financial datasets. Skilled in forecasting, variance analysis, and KPI reporting, delivering actionable insights to support business decisions. Experienced in automating processes, building dashboards, and collaborating with stakeholders to drive data-driven strategies and improve operational efficiency.
"""

jd_text = """Requirements:

Bachelor's degree in a relevant field (e.g. Mathematics, Economics, Statistics, Business, Data Science)
Proficiency in Microsoft Excel; familiarity with PostgreSQL, Python, or R is advantageous
Strong analytical and problem-solving skills with attention to detail
Good written and verbal communication skills
Ability to manage time effectively and work on multiple tasks simultaneously

Desirable Skills:

Experience with data visualization tools such as Power BI or Tableau
Understanding of business processes or industry-specific metrics
Prior internship or placement experience in an analytical role """

# run workflow
input_state = {
    'cv_text': cv_text,
    'jd_text': jd_text}

result = workflow.invoke(
    input_state,
    config={
        "run_name": "cv-bot-run-777",
        "metadata": {"project": "cv-bot", "owner": "vijay"},
        "tags": ["qa"]  # <-- Add this line
    }
)

print(result['missing_skills'])



