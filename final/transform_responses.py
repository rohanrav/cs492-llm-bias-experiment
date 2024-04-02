import json


def transform_json(input_file, output_file):
    with open(input_file, 'r') as f:
        prompts = json.load(f)

    transformed_prompts = []

    for prompt in prompts:
        for model, response in prompt["response"].items():
            transformed_prompt = {
                "race": prompt["race"],
                "gender": prompt["gender"],
                "bio": prompt["bio"],
                "prompt": prompt["prompt"],
                "model": model,
                "response": response
            }
            transformed_prompts.append(transformed_prompt)

    with open(output_file, 'w') as f:
        json.dump(transformed_prompts, f, indent=4)


transform_json('prompts_with_responses.json', 'transformed_prompts.json')
