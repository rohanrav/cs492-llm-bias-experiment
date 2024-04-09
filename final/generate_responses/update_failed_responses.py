import json
import asyncio
from generate_prompts import get_gpt_response, get_gemini_pro_response, get_llama2_7b_response, get_llama2_7b_uncensored_response


async def update_failed_responses():
    with open("prompts_with_responses.json", "r") as f:
        prompts = json.load(f)

    model_functions = {
        "GPT 3.5": get_gpt_response,
        "Gemini Pro": get_gemini_pro_response,
        "LLama2 7B": get_llama2_7b_response,
        "LLama2 7B Uncensored": get_llama2_7b_uncensored_response
    }

    for prompt in prompts:
        for model, response in prompt["response"].items():
            if "request failed" in response:
                print(f"Recalculating response for {model}...")
                new_response = await model_functions[model](prompt["prompt"])
                prompt["response"][model] = new_response

    with open("prompts_with_responses_updated.json", "w") as f:
        json.dump(prompts, f, indent=2)

if __name__ == '__main__':
    asyncio.run(update_failed_responses())
