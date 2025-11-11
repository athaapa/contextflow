<p align="center">
  <p align="center">
	<img width="128" height="128" src="https://github.com/athaapa/contextflow/blob/74d4dffdb3c0c1777ccd3593b7d52f89f9406afc/images/logo.png" alt="Logo">
  </p>

  <h1 align="center"><b>ContextFlow</b></h1>
  <p align="center">
     Context optimization for AI agents.
    <br />
  </p>
</p>

**ContextFlow** is a fast, open-source Python library that compresses chat, agent, or RAG context before you send it to an expensive LLM. It finds the “needle in the haystack,” scoring every message by utility, keeping only critical facts, summarizing the rest, and dropping noise—preserving quality while saving money.

# Why ContextFlow?
Most LLM applications waste money by sending entire chat histories, long chains of document chunks, or endless boilerplate every API call. Most of it is redundant, repetitive, or irrelevant to the user’s actual goal.

ContextFlow solves this by:
- Ranking every message by utility using a fast LLM batch
- Keeping only high-signal content (order numbers, errors, decisions)
- Aggressively summarizing medium-utility content
- Dropping low-value fluff and polite filler
- Always preserving the last few recent messages for context

# Quickstart
## Installation
```bash
pip install contextflow
```
## Setup
Get your Gemini API Key (for message scoring and summarization):
```.env
export GEMINI_API_KEY="YOUR_KEY_HERE"
```
If you don't have one, you can get one for free [here](https://aistudio.google.com/api-keys).
## Example
```python
from contextflow import ContextFlow

# Chat history (list of {"role": str, "content": str})
messages = [
    # ... up to 50 messages ...
]

cf = ContextFlow()

result = cf.optimize(
    messages=messages,
    agent_goal="Resolve customer shipping inquiry",  # this guides scoring
    max_tokens=400  # your target for output
)

print("Tokens before:", result["analytics"]["tokens_before"])
print("Tokens after:", result["analytics"]["tokens_after"])
print("Reduction:", result["analytics"]["reduction_pct"], "%")
print("Messages ready for LLM:", result["messages"])

# Now use result.messages in your LLM query!
# response = openai.ChatCompletion.create(model="gpt-4", messages=result.messages)
```
# License
MIT License - see [LICENSE](https://github.com/athaapa/contextflow/blob/6b79c95d55c53c40bd8f500a6c921ea394c32666/LICENSE) file for details.
