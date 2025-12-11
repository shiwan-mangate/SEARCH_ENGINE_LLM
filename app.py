import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

st.title("🚀 Fast LangChain + Groq Chatbot")


CUSTOM_SYSTEM_PROMPT = """
You are a helpful AI assistant that intelligently uses tools.

You have access to the following tools:
- search: for general web or search engine queries and up-to-date information.
- arxiv: for academic and research paper queries.
- wiki: for general knowledge, definitions, explanations of concepts, people, or events.

TOOL USAGE RULES:
- For factual, technical, or knowledge-heavy questions (e.g., 'What is machine learning?',
  'Explain transformers in detail', 'Who is Alan Turing?'), you SHOULD first call an
  appropriate tool (usually `wiki` or `arxiv`) to gather accurate information, and then
  synthesize a detailed final answer.
- For anything related to recent events, news, or time-sensitive information, you MUST
  use the `search` tool.
- You MUST NOT get stuck repeatedly calling tools. Use at most 2 tool calls per user
  question. After that, answer with the information you have.
- Do NOT call the same tool again with nearly the same query if it did not add new value.
- Skip tools only for questions about personal preferences, opinions, or conversational topics.

ANSWER STYLE:
- Provide detailed, structured responses.
- Use bullet points, numbering, and headings when helpful.
- Explain concepts step-by-step, as if teaching a motivated beginner.
- Always combine retrieved information with your own reasoning for clarity.

Your goal is to produce the most accurate, helpful, and detailed answer possible.
"""

# Sidebar API Key Input
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")


@st.cache_resource
def load_llm(api_key: str):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True,
        temperature=0.2,
    )


@st.cache_resource
def load_tools():
    # WIKI
    api_wrapper_wiki = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250,
    )
    wiki = WikipediaQueryRun(
        api_wrapper=api_wrapper_wiki,
        description=(
            "Use this tool to look up general knowledge, definitions, people, places, "
            "companies, and other common concepts. Ideal for questions like "
            "'What is machine learning?' or 'Who is Alan Turing?'."
        ),
    )
    wiki.name = "wiki"

    # ARXIV
    api_wrapper_arxiv = ArxivAPIWrapper(
        max_results=1,
        doc_content_chars_max=250,
    )
    arxiv = ArxivQueryRun(
        api_wrapper=api_wrapper_arxiv,
        description=(
            "Use this tool to search and summarize academic papers on arXiv. "
            "Best for research topics, paper IDs like '1706.03762', and ML/AI papers."
        ),
    )
    arxiv.name = "arxiv"

    # SEARCH
    search = DuckDuckGoSearchRun()
    search.name = "search"
    search.description = (
        "Use this tool for general web or search engine queries, "
        "news, or up-to-date information from the internet."
    )

    return [search, arxiv, wiki]



@st.cache_resource
def load_agent(api_key: str):
    llm = load_llm(api_key)
    tools = load_tools()
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=CUSTOM_SYSTEM_PROMPT,
    )


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])



if prompt := st.chat_input("Ask me anything..."):
    if not api_key:
        st.warning("Please enter your Groq API key")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    agent = load_agent(api_key)

    with st.chat_message("assistant"):
        
        history = st.session_state.messages[-6:]

        
        response = agent.invoke(
            {"messages": history},
            config={"recursion_limit": 8},
        )

        output = response["messages"][-1].content
        st.session_state.messages.append(
            {"role": "assistant", "content": output}
        )

        st.write(output)

