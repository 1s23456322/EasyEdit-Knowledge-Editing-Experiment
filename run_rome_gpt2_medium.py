import json
import sys
import gc
import torch

# 让 Python 能找到 EasyEdit 源码
sys.path.insert(0, r"D:\LLM\EasyEdit")

from easyeditor import BaseEditor, ROMEHyperParams


DATA_PATH = r"D:\LLM\data.json"
OUTPUT_PATH = r"D:\LLM\rome_gpt2_medium_results.json"
HPARAMS_PATH = r"D:\LLM\EasyEdit\hparams\ROME\gpt2-medium.yaml"


def generate_answer(model, tokenizer, prompt, max_new_tokens=20):
    """
    使用模型生成回答。
    返回：
    - answer: 去掉 prompt 后的新生成部分
    - full_text: prompt + 生成内容
    """
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 只保留模型在 prompt 后面新生成的部分
    answer = full_text[len(prompt):].strip()

    return answer, full_text


def contains_answer(text, answer):
    """
    判断生成文本中是否包含目标答案。
    """
    if text is None or answer is None:
        return False
    return answer.lower() in text.lower()


def main():
    print("Loading data from:", DATA_PATH)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = []

    for idx, item in enumerate(data, start=1):
        print("\n" + "=" * 100)
        print(f"Editing sample {idx}/{len(data)}")
        print("Prompt:", item["prompt"])
        print("Subject:", item["subject"])
        print("Ground truth:", item["ground_truth"])
        print("Target new:", item["target_new"])

        editor = None
        edited_model = None
        metrics = None

        try:
            # 每条事实都重新加载原始模型
            # 这满足“逐条编辑，每次重置模型权重”的实验要求
            hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)
            editor = BaseEditor.from_hparams(hparams)

            prompts = [item["prompt"]]
            subjects = [item["subject"]]
            ground_truth = [item["ground_truth"]]
            target_new = [item["target_new"]]

            print("Running ROME edit...")

            # 关键修改：
            # keep_original_weight=False
            # 这样 edited_model 会保留编辑后的权重，后面的 generate 才能看到编辑效果
            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=target_new,
                subject=subjects,
                keep_original_weight=False
            )

            tokenizer = editor.tok

            # 1. 原始 prompt 验证：检查 rewrite 是否成功
            edited_answer, edited_full_text = generate_answer(
                edited_model,
                tokenizer,
                item["prompt"],
                max_new_tokens=20
            )

            # 2. rephrase prompt 验证：检查换一种问法是否也能生效
            rephrase_answer, rephrase_full_text = generate_answer(
                edited_model,
                tokenizer,
                item["rephrase_prompt"],
                max_new_tokens=20
            )

            # 3. locality prompt 验证：检查无关知识是否被破坏
            locality_answer, locality_full_text = generate_answer(
                edited_model,
                tokenizer,
                item["locality_prompt"],
                max_new_tokens=20
            )

            result = {
                "id": idx,

                "prompt": item["prompt"],
                "subject": item["subject"],
                "ground_truth": item["ground_truth"],
                "target_new": item["target_new"],

                "edited_answer": edited_answer,
                "edited_full_text": edited_full_text,
                "hit_target_new": contains_answer(edited_answer, item["target_new"]),
                "hit_ground_truth": contains_answer(edited_answer, item["ground_truth"]),

                "rephrase_prompt": item["rephrase_prompt"],
                "rephrase_answer": rephrase_answer,
                "rephrase_full_text": rephrase_full_text,
                "rephrase_hit_target_new": contains_answer(
                    rephrase_answer,
                    item["target_new"]
                ),

                "locality_prompt": item["locality_prompt"],
                "locality_ground_truth": item["locality_ground_truth"],
                "locality_answer": locality_answer,
                "locality_full_text": locality_full_text,
                "locality_preserved": contains_answer(
                    locality_answer,
                    item["locality_ground_truth"]
                ),

                "metrics": metrics,
                "error": None
            }

            print("\nEdit finished.")
            print("Edited answer:", edited_answer)
            print("Hit target new:", result["hit_target_new"])
            print("Hit ground truth:", result["hit_ground_truth"])

            print("\nRephrase answer:", rephrase_answer)
            print("Rephrase hit target new:", result["rephrase_hit_target_new"])

            print("\nLocality answer:", locality_answer)
            print("Locality preserved:", result["locality_preserved"])

            if metrics is not None:
                try:
                    print("\nEasyEdit pre rewrite_acc:", metrics[0]["pre"]["rewrite_acc"])
                    print("EasyEdit post rewrite_acc:", metrics[0]["post"]["rewrite_acc"])
                except Exception:
                    print("Metrics exists, but rewrite_acc format is unexpected.")

        except Exception as e:
            print("Edit failed:", repr(e))

            result = {
                "id": idx,

                "prompt": item.get("prompt"),
                "subject": item.get("subject"),
                "ground_truth": item.get("ground_truth"),
                "target_new": item.get("target_new"),

                "edited_answer": None,
                "edited_full_text": None,
                "hit_target_new": False,
                "hit_ground_truth": False,

                "rephrase_prompt": item.get("rephrase_prompt"),
                "rephrase_answer": None,
                "rephrase_full_text": None,
                "rephrase_hit_target_new": False,

                "locality_prompt": item.get("locality_prompt"),
                "locality_ground_truth": item.get("locality_ground_truth"),
                "locality_answer": None,
                "locality_full_text": None,
                "locality_preserved": False,

                "metrics": metrics,
                "error": repr(e)
            }

        all_results.append(result)

        # 每条样本结束后保存一次，防止中途报错导致结果丢失
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print("\nCurrent results saved to:", OUTPUT_PATH)

        # 清理显存和内存
        try:
            del editor
        except Exception:
            pass

        try:
            del edited_model
        except Exception:
            pass

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll ROME results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()