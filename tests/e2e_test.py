"""
End-to-end test of ContextFlow
"""

from contextflow import ContextFlow
from contextflow.utils.tokenizer import count_tokens
from dotenv import load_dotenv

load_dotenv()

# Sample conversation (customer support)
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {
        "role": "user",
        "content": "I ordered a laptop last week but haven’t received it yet.",
    },
    {
        "role": "assistant",
        "content": "I'm sorry for the delay. Can you share your order number?",
    },
    {"role": "user", "content": "Sure, it’s #9988123."},
    {
        "role": "assistant",
        "content": "Thank you. Let me check the order details.",
    },
    {
        "role": "assistant",
        "content": "The order shows shipped on November 5th via UPS.",
    },
    {
        "role": "user",
        "content": "But the tracking info hasn’t updated since the 7th.",
    },
    {
        "role": "assistant",
        "content": "I see. Can you confirm your shipping address to ensure there’s no error?",
    },
    {"role": "user", "content": "It’s 777 Market St, San Francisco, CA."},
    {"role": "assistant", "content": "Thanks! That matches what we have."},
    {"role": "assistant", "content": "I'm contacting UPS to get an update."},
    {
        "role": "assistant",
        "content": "Meanwhile, would you like me to open a support ticket for this issue?",
    },
    {
        "role": "user",
        "content": "Yes, please. I need this laptop for work urgently.",
    },
    {"role": "assistant", "content": "Understood, I’ll prioritize this case."},
    {
        "role": "user",
        "content": "Is there anything else I need to do on my end?",
    },
    {
        "role": "assistant",
        "content": "No, I’ve got your information. Our team will follow up ASAP.",
    },
    {"role": "user", "content": "Okay, thanks for your help."},
    {
        "role": "assistant",
        "content": "You’re welcome! I’ll message you once I hear from UPS.",
    },
    {"role": "assistant", "content": "Checking for updates now…"},
    {
        "role": "assistant",
        "content": "UPS says the package is delayed at their facility due to high volume.",
    },
    {
        "role": "assistant",
        "content": "Estimated delivery is Friday, November 12th.",
    },
    {
        "role": "user",
        "content": "That’s later than promised. Can anything be done to expedite it?",
    },
    {
        "role": "assistant",
        "content": "UPS can’t guarantee earlier delivery, but I’ll mark your order as urgent.",
    },
    {"role": "user", "content": "Will I get a compensation if it's late?"},
    {
        "role": "assistant",
        "content": "If it isn’t delivered by the 12th, you’ll be eligible for a partial refund.",
    },
    {"role": "user", "content": "Can you notify me if anything changes?"},
    {
        "role": "assistant",
        "content": "Absolutely, I’ll keep you updated via email.",
    },
    {
        "role": "user",
        "content": "Here’s my email, in case you don’t have it: alice.buyer@gmail.com.",
    },
    {"role": "assistant", "content": "Confirmed! That’s what we have on file."},
    {
        "role": "user",
        "content": "Is there a case number for this support ticket?",
    },
    {
        "role": "assistant",
        "content": "Yes, the ticket number is #20241110-7854.",
    },
    {"role": "user", "content": "Thanks, I’ll use that if I need to call."},
    {
        "role": "assistant",
        "content": "Exactly! UPS may also contact you directly.",
    },
    {"role": "user", "content": "I’ll check my email for updates."},
    {
        "role": "assistant",
        "content": "Perfect. Is there anything else I can do for you today?",
    },
    {"role": "user", "content": "No, that covers it."},
    {"role": "assistant", "content": "Thank you for your patience, Alice."},
    {"role": "user", "content": "Thank you for your help!"},
    {"role": "assistant", "content": "Glad to help. Have a wonderful day!"},
    {"role": "user", "content": "You too, bye!"},
    {
        "role": "assistant",
        "content": "Goodbye! Let us know if you need anything else.",
    },
]


print("=" * 60)
print("CONTEXTFLOW END-TO-END TEST")
print("=" * 60)

# Initialize ContextFlow
cf = ContextFlow(scoring_model="gemini", summarizing_model="gemini")

# Define goal
goal = "Resolve customer shipping inquiry"

# Show before state
print("\nBEFORE OPTIMIZATION:")
print(f"   Messages: {len(messages)}")
estimated_tokens_before = count_tokens(messages)
print(f"   Estimated tokens: ~{estimated_tokens_before}")

# Optimize
print("\nOPTIMIZING...")
result = cf.optimize(messages=messages, goal=goal, max_token_count=350)

# Show results
print("\nAFTER OPTIMIZATION:")
print(f"   Messages: {len(result['messages'])}")
print(f"   Tokens: {result['analytics']['tokens_after']}")
print(f"   Reduction: {result['analytics']['reduction_pct']:.1f}%")
print(f"   Tokens saved: {result['analytics']['tokens_saved']}")
print(f"   Time taken: {result['analytics']['time_taken_ms']}ms")

# Show the optimized messages
print("\n OPTIMIZED CONTEXT:")
print("=" * 60)
for i, msg in enumerate(result["messages"], 1):
    role = msg.get("role", "unknown")
    content = msg.get("content", "")  # First 100 chars
    print(f"{i}. [{role}] {content}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
