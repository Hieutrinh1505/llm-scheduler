import asyncio
from openai import OpenAI
import aiohttp

# Synchronous OpenAI client call
client = OpenAI(api_key="no-need", base_url="http://localhost:8000/v1")

response = client.completions.create(
    model="gpt2", prompt="Tell me a joke", max_tokens=100, temperature=0.7
)

print("Response:", response.choices[0].text)
print("\n" + "=" * 50 + "\n")


# Async metrics fetching
async def get_vllm_metrics():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/metrics") as response:
            metrics = await response.text()
            print("vLLM Metrics:")
            print(metrics)


# Run the async function
asyncio.run(get_vllm_metrics())
