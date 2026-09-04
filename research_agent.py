import os
import streamlit as st
from openai import OpenAI

MODEL = "gpt-5-mini"


def get_sources(response):
    """Extract source URLs returned by the web-search tool."""
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


def generate_report(api_key, topic):
    client = OpenAI(api_key=api_key)

    instructions = """
You are a research agent.

Your task:
1. Search the web for trustworthy, recent, and relevant information.
2. Compare findings from multiple sources.
3. Write a structured report in clear, simple language.

Use exactly these headings:

# Research Report: [topic]
## Executive Summary
## Key Findings
## Analysis
## Conclusion

Rules:
- Use factual, evidence-based statements.
- Do not invent statistics, sources, or citations.
- Mention uncertainty or disagreement between sources when applicable.
- Add citations in the report where useful.
- The application will display the full reference list separately.
"""

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}],
        instructions=instructions,
        input=f"Research topic: {topic}",
    )

    return response.output_text, get_sources(response)


st.set_page_config(page_title="Research Report Agent", page_icon="🔎")

st.title("🔎 Research Report Agent")
st.caption("Searches the web, summarizes findings, and generates a referenced report.")

api_key = st.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
)

topic = st.text_area(
    "Enter a research topic",
    placeholder="Example: Impact of artificial intelligence on education",
)

if st.button("Generate Research Report", type="primary"):
    if not api_key:
        st.error("Enter your OpenAI API key.")
    elif not topic.strip():
        st.error("Enter a research topic.")
    else:
        try:
            with st.spinner("Searching the web and preparing your report..."):
                report, sources = generate_report(api_key, topic)

            st.markdown(report)

            st.divider()
            st.subheader("References")

            if sources:
                for number, source in enumerate(sources, start=1):
                    st.markdown(f"{number}. [{source['title']}]({source['url']})")
            else:
                st.info("No source URLs were returned.")
        except Exception as error:
            st.error(f"Could not generate the report: {error}")
