from core.optimizer import ContextFlow
from utils.tokenizer import count_tokens
from dotenv import load_dotenv

load_dotenv()

optimizer = ContextFlow()

messages = [
    {
        "role": "user",
        "content": "Hi, I ordered something last week but it hasn't arrived yet",
    },
    {
        "role": "assistant",
        "content": "I'm sorry to hear that. I'd be happy to help you track your order. Can you provide your order number?",
    },
    {
        "role": "user",
        "content": "Sure, it's order #12345. I placed it on November 1st",
    },
    {"role": "assistant", "content": "Thank you. Let me look that up for you."},
    {
        "role": "assistant",
        "content": "I found your order. It was shipped on November 5th via USPS Priority Mail. The tracking number is 9405511899223456789012. According to the tracking information, it's currently in transit and is expected to arrive by November 10th at your address in San Francisco, CA.",
    },
    {
        "role": "user",
        "content": "Okay, that's helpful. Is there anything I can do to speed it up?",
    },
    {
        "role": "assistant",
        "content": "Unfortunately, once a package is in transit with USPS, we can't expedite it further. However, if it doesn't arrive by November 10th, please contact us again and we'll look into filing a claim or sending a replacement.",
    },
    {
        "role": "user",
        "content": "Alright, I'll wait until then. Thanks for checking!",
    },
    {
        "role": "assistant",
        "content": "You're very welcome! Is there anything else I can help you with today?",
    },
    {"role": "user", "content": "No, that's all. Have a great day!"},
    {
        "role": "assistant",
        "content": "You too! Feel free to reach out if you need anything else.",
    },
]
agent_goal = "Resolve customer shipping inquiry"

print(count_tokens(messages))
print(count_tokens(messages[-5:]))

optimized = optimizer.optimize(messages, agent_goal, 150)
print(optimized)
print(count_tokens(optimized))
