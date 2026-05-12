import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DATA_PATH = r"D:\LLM\data.json"
OUTPUT_PATH = r"D:\LLM\baseline_results.json"

MODEL_NAME = "gpt2-medium"


def generate_answer(model, tokenizer, prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = full_text[len(prompt):].strip()
    return answer, full_text


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    results = []

    for i, item in enumerate(data, start=1):
        prompt = item["prompt"]
        answer, full_text = generate_answer(model, tokenizer, prompt)

        result = {
            "id": i,
            "prompt": prompt,
            "subject": item["subject"],
            "ground_truth": item["ground_truth"],
            "target_new": item["target_new"],
            "model_answer": answer,
            "full_text": full_text,
            "hit_ground_truth": item["ground_truth"].lower() in answer.lower(),
            "hit_target_new": item["target_new"].lower() in answer.lower()
        }

        results.append(result)

        print("=" * 80)
        print(f"Sample {i}")
        print("Prompt:", prompt)
        print("Ground truth:", item["ground_truth"])
        print("Target new:", item["target_new"])
        print("Model answer:", answer)
        print("Hit ground truth:", result["hit_ground_truth"])
        print("Hit target new:", result["hit_target_new"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nBaseline results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()