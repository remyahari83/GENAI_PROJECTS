Resume Evaluation Agent using Agentic AI and Tool Calling

An application that analyses a candidate’s resume against a target job description (JD).

1. Resume Upload
- Upload a resume in PDF format and provide the target job description.

2. Input Guardrails (`guardrails.py`)
- The resume and JD pass through input guardrails to detect potential prompt injection attempts before being sent to the LLM.

3. LLM Analysis and Tool Selection (`prompts.py`, `tools.py`)
- The resume and JD are passed to the LLM along with the system prompt and descriptions of the available tools.
- Available tools include:
  - Keyword frequency analysis.
  - Years of experience extraction.

4. Agentic Tool-Calling Loop (`agent.py`)
- The LLM analyses the resume and JD and decides which tools are required.
- It requests the execution of the selected tools.
- Python executes the corresponding functions and returns the results to the LLM.
- The LLM uses these results for further analysis and can request additional tools.
- This cycle continues until the LLM produces its final response.
- The resume is evaluated with a fit score and detailed analysis highlighting the candidate’s alignment with the job description.

5. Cover Letter Generation
- Generates a job-specific cover letter using the evaluation scorecard and job description.

6. Streamlit GUI
- Built using Streamlit, allowing users to upload resumes, provide job descriptions, and view the fit score, evaluation analysis, cover letter, and execution metrics.

7. LLM Observability and Execution Metrics

A dedicated dashboard panel displays:

- LLM API calls and token consumption (prompt, completion, and total tokens).
- Estimated API cost based on model-specific pricing.
- LLM response latency.
- Individual tool invocation details and execution latency.
- Total execution time and number of LLM turns.

How to Run: (Visual studio code Terminal)

Create virtual environment.
```bash
py -3.11 -m venv myenv
```
Install dependencies:

```bash
python -m pip install -r requirements.txt
```
Place the key in .env directly under JD-Matcher directory. 
Mention like GROQ_API_KEY=<key>

Run the application:

```bash
python -m streamlit run .\main.py
```

#AgenticAI #GenerativeAI #LLM #Python #ToolCalling #LLMObservability #GenAIProjects