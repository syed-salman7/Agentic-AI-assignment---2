import os
import streamlit as st
from openai import OpenAI

MODEL = "gpt-4.1-mini"

INSTRUCTIONS = """
You are a defensive cybersecurity log-analysis agent.

Analyze only the supplied security logs or alerts. Your role is to help
a security analyst identify suspicious activity and recommend safe,
defensive next steps. Do not provide attack instructions.

Create a report with exactly these sections:

# Security Alert Analysis
## Overall Severity
Choose one: Informational, Low, Medium, High, or Critical.

## Potential Threats Identified
List each suspicious event and explain why it may be risky.

## Evidence From Logs
Quote the important log lines or fields.

## Recommended Mitigation Steps
Recommend safe actions such as verifying the event, preserving logs,
resetting affected credentials, blocking confirmed malicious indicators,
isolating a host when justified, patching systems, and escalating to the
security team.

## Analyst Notes
State any uncertainty and what extra information is needed.

Rules:
- Do not claim a threat is confirmed unless the logs prove it.
- Clearly distinguish suspicious behavior from confirmed compromise.
- Keep the report concise and understandable.
"""


def analyze_logs(api_key, logs):
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=f"Analyze these security logs:\n\n{logs}",
    )

    return response.output_text


st.set_page_config(page_title="Security Log Analysis Agent", page_icon="🛡️")

st.title("🛡️ Security Log Analysis Agent")
st.caption("Analyzes security logs, classifies threat severity, and suggests defensive mitigation steps.")

api_key = st.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
)

uploaded_file = st.file_uploader(
    "Upload a log file (optional)",
    type=["txt", "log", "csv"],
)

logs = st.text_area(
    "Paste security logs or alerts",
    height=250,
    placeholder="""Example:
2026-09-02 10:20:11 FAILED_LOGIN user=admin ip=203.0.113.10
2026-09-02 10:20:18 FAILED_LOGIN user=admin ip=203.0.113.10
2026-09-02 10:20:30 FAILED_LOGIN user=admin ip=203.0.113.10
2026-09-02 10:21:05 LOGIN_SUCCESS user=admin ip=203.0.113.10""",
)

if uploaded_file:
    logs = uploaded_file.getvalue().decode("utf-8", errors="replace")
    st.text_area("Uploaded log content", logs, height=200, disabled=True)

if st.button("Analyze Security Logs", type="primary"):
    if not api_key:
        st.error("Enter your OpenAI API key.")
    elif not logs.strip():
        st.error("Paste logs or upload a log file first.")
    else:
        try:
            with st.spinner("Analyzing logs and classifying potential threats..."):
                report = analyze_logs(api_key, logs)

            st.markdown(report)

        except Exception as error:
            st.error(f"Could not analyze logs: {error}")
