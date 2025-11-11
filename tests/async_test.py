import asyncio
import time
from dotenv import load_dotenv
from contextflow.core.scorer import MessageScorer

load_dotenv()


async def test_async_parallelization():
    """Test that batches run in parallel, not sequential"""

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
        {
            "role": "assistant",
            "content": "I'm contacting UPS to get an update.",
        },
        {
            "role": "assistant",
            "content": "Meanwhile, would you like me to open a support ticket for this issue?",
        },
        {
            "role": "user",
            "content": "Yes, please. I need this laptop for work urgently.",
        },
        {
            "role": "assistant",
            "content": "Understood, I’ll prioritize this case.",
        },
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
        {
            "role": "assistant",
            "content": "Confirmed! That’s what we have on file.",
        },
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

    goal = "Resolve customer shipping inquiry"

    scorer = MessageScorer(model="groq")

    # Test 1: Sequential (for comparison)
    print("=== SEQUENTIAL TEST ===")
    start = time.time()
    result1 = await scorer._score_batch_async(messages[0:20], goal)
    result2 = await scorer._score_batch_async(messages[20:40], goal)
    sequential_time = time.time() - start
    print(f"Sequential: {sequential_time:.2f}s")

    # Test 2: Parallel (what you want)
    print("\n=== PARALLEL TEST ===")
    start = time.time()
    results = await asyncio.gather(
        scorer._score_batch_async(messages[0:20], goal),
        scorer._score_batch_async(messages[20:40], goal),
    )
    parallel_time = time.time() - start
    print(f"Parallel: {parallel_time:.2f}s")

    # Verify speedup
    print("\n=== RESULTS ===")
    print(f"Speedup: {sequential_time / parallel_time:.1f}x")

    if parallel_time < sequential_time * 0.6:
        print("✅ Async is working! Batches ran in parallel.")
    else:
        print("❌ Async NOT working. Batches ran sequentially.")


# Run test
asyncio.run(test_async_parallelization())
