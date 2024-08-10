import asyncio
import nest_asyncio
from intelisys_current import Intelisys

nest_asyncio.apply()

async def run_tests():
    print("Test 1: Synchronous usage")
    intelisys = Intelisys(json_mode=True, log="DEBUG")
    response = intelisys.chat("Hello, how are you?")
    print(response)
    print("\n" + "-"*50 + "\n")

    print("Test 2: Template usage")
    response = intelisys.template_chat(
        template="Summarize this in {{words}} words: {{text}}",
        render_data={"words": 10, "text": "After carefully reviewing the provided resume, I don't see any explicit mention of a specific army rank. The document describes various military and civilian roles and responsibilities, but does not state a particular rank like Lieutenant, Captain, Major, etc. \nThe experience described suggests the individual has had significant leadership roles in the U.S. Army and Army Reserve, including positions like: Chief of the Central Team in the Army Reserve Sustainment Command Transportation Officer in the US Army Reserve Deployment Support Command Director of Logistics for Area Support Group - Kuwait Contingency Contracting Officer in Southwest Asia. While these roles imply a relatively senior position, without an explicitly stated rank, I cannot confirm any specific army rank from the information provided. The focus seems to be more on describing job responsibilities and accomplishments rather than military rank progression."}
    )
    print(response)
    print("\n" + "-"*50 + "\n")

    print("Test 3: Default template usage")
    intelisys = Intelisys(json_mode=True)
    intelisys.set_default_template("Return JSON, {{ name }}! {{ question }}")
    response = intelisys.template_chat(
        render_data={"name": "Alice", "question": "How are you today?"}
    )
    print(response)
    print("\n" + "-"*50 + "\n")

    print("Test 4: Batch processing")
    intelisys = Intelisys(provider="openai", model="gpt-4")
    questions = ["What is AI?", "How does machine learning work?", "What are neural networks?"]
    responses = intelisys.batch_process(questions)
    for question, response in zip(questions, responses):
        print(f"Q: {question}\nA: {response}\n")
    print("\n" + "-"*50 + "\n")

    print("Test 5: Caching")
    intelisys = Intelisys(provider="openai", model="gpt-4", use_cache=True)
    response1 = intelisys.chat("What is the capital of France?")
    response2 = intelisys.chat("What is the capital of France?")
    print(f"Responses are the same: {response1 == response2}")
    print("\n" + "-"*50 + "\n")

    print("Test 6: Async chat and batch processing")
    async with Intelisys(provider="anthropic", model="claude-3-5-sonnet-20240620", use_async=True) as async_intelisys:
        response = await async_intelisys.async_chat("What's the weather like today?")
        print(f"Async chat response: {response}")

        questions = ["What is AI?", "How does machine learning work?", "What are neural networks?"]
        responses = await async_intelisys.async_batch_process(questions)
        for question, response in zip(questions, responses):
            print(f"Q: {question}\nA: {response}\n")
    print("\n" + "-"*50 + "\n")

    print("Test 7: Image OCR")
    intelisys = Intelisys(provider="openrouter", model="google/gemini-pro-vision-latest")
    result = (intelisys
        .image("/Users/lifsys/Downloads/OCR-Test.jpg")
        .chat("Please provide the complete text in the following image(s).")
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(run_tests())