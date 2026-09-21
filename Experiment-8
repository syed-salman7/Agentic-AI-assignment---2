import os
import streamlit as st
from openai import OpenAI

MODEL = "gpt-5-mini"


def extract_sources(response):
    """Collect web sources found by the Research Agent."""
    sources = []
    seen_urls = set()

    for item in response.output:
        if item.type != "message":
            continue

        for content in item.content:
            for annotation in getattr(content, "annotations", []):
                if annotation.type == "url_citation":
                    url = annotation.url
                    title = getattr(annotation, "title", url)

                    if url not in seen_urls:
                        sources.append({"title": title, "url": url})
                        seen_urls.add(url)

    return sources


def run_research_agent(client, task):
    """Agent 1: searches the web and gathers facts."""
    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}],
        instructions="""
You are the Research Agent in a multi-agent system.

Search the web for trustworthy and relevant information about the task.
Return concise research notes with factual findings, useful numbers,
different viewpoints, and limitations. Do not write the final report.
Use multiple reliable sources where possible.
""",
        input=f"Research task: {task}",
    )

    return response.output_text, extract_sources(response)


def run_analyst_agent(client, task, research_notes):
    """Agent 2: evaluates the Research Agent's findings."""
    response = client.responses.create(
        model=MODEL,
        instructions="""
You are the Analyst Agent in a multi-agent system.

Review the research notes received from the Research Agent.
Identify the most important insights, compare viewpoints, identify
possible bias or uncertainty, and explain logical conclusions.

Do not add facts that are not supported by the research notes.
Do not write the final report; provide structured analysis for the
Report Agent.
""",
        input=f"""
Original task:
{task}

Research Agent notes:
{research_notes}
""",
    )

    return response.output_text


def run_report_agent(client, task, research_notes, analysis, sources):
    """Agent 3: converts research and analysis into a final report."""
    reference_list = "\n".join(
        f"{number}. {source['title']} - {source['url']}"
        for number, source in enumerate(sources, start=1)
    )

    response = client.responses.create(
        model=MODEL,
        instructions="""
You are the Report Agent in a multi-agent system.

Create a clear, well-structured final report using only the research
notes and analyst output provided to you.

Use exactly these headings:

# Final Report
## Executive Summary
## Key Findings
## Analysis
## Conclusion
## References

Write in simple academic language. Include only references provided in
the source list. If the evidence is uncertain, state that clearly.
""",
        input=f"""
Original task:
{task}

Research Agent notes:
{research_notes}

Analyst Agent output:
{analysis}

Approved source list:
{reference_list}
""",
    )

    return response.output_text


st.set_page_config(page_title="Multi-Agent Research System", page_icon="🤖")

st.title("🤖 Multi-Agent Research System")
st.caption("Research Agent → Analyst Agent → Report Agent")

api_key = st.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
)

task = st.text_area(
    "Enter a task for the agents",
    height=150,
    placeholder="Example: Analyze the impact of artificial intelligence on education.",
)

if st.button("Run Multi-Agent System", type="primary"):
    if not api_key:
        st.error("Enter your OpenAI API key.")
    elif not task.strip():
        st.error("Enter a task first.")
    else:
        try:
            client = OpenAI(api_key=api_key)

            with st.status("Agents are collaborating...", expanded=True) as status:
                st.write("🔎 Research Agent is searching for information...")
                research_notes, sources = run_research_agent(client, task)

                st.write("📊 Analyst Agent is evaluating the findings...")
                analysis = run_analyst_agent(client, task, research_notes)

                st.write("📝 Report Agent is writing the final report...")
                final_report = run_report_agent(
                    client, task, research_notes, analysis, sources
                )

                status.update(label="Multi-agent task completed", state="complete")

            st.markdown(final_report)

            with st.expander("View Research Agent output"):
                st.write(research_notes)

            with st.expander("View Analyst Agent output"):
                st.write(analysis)

        except Exception as error:
            st.error(f"Could not complete the task: {error}")
