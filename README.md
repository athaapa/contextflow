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
- Keeping only high-signal content that is relevant to the agent's goal (order numbers, errors, decisions, etc.)
- Aggressively summarizing medium-utility content
- Dropping low-value fluff and filler
- Always preserving the last few recent messages for recency bias

# Quickstart
## Installation
Install the SDK with `pip`:
```bash
pip install contextflow-ai
```
## Setup
By default, ContextFlow uses Gemini 2.5 Flash-Lite which requires a Google API key (you can get one for free [here](https://aistudio.google.com/api-keys)), but ContextFlow also supports a number of other providers.
```.env
export GEMINI_API_KEY="YOUR_KEY_HERE"
export GROQ_API_KEY="YOUR_KEY_HERE"
export OPENAI_API_KEY="YOUR_KEY_HERE"
export ANTHROPIC_API_KEY="YOUR_KEY_HERE"
```
### Disclaimer
ContextFlow is provided "as-is" without warranty. You are responsible for:
- Securing your API keys
- Monitoring your API usage and costs
- Compliance with your LLM provider's terms of service

The maintainers of ContextFlow are not liable for any API costs, security breaches, or damages arising from use of this library.
## Example
```python
from contextflow import ContextFlow

# Chat history (list of {"role": str, "content": str})
messages = [
    # ... up to 50 messages ...
]

cf = ContextFlow(
	scoring_model="anthropic", # You can mix and match providers to see what is working for you 
	summarizing_model="gemini"
)

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
# Security Notice

ContextFlow is a client-side library. **You are responsible for securing your API keys.**

- Never commit API keys to version control.
- Use environment variables (`.env` files) to store keys.
- Monitor your API usage on your provider's dashboard.
- ContextFlow does not transmit your keys to any external server.
# License
MIT License - see [LICENSE](https://github.com/athaapa/contextflow/blob/6b79c95d55c53c40bd8f500a6c921ea394c32666/LICENSE) file for details.
