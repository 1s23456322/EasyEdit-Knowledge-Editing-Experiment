import json
import sys
import io
import gc
import torch
import logging
import contextlib

sys.path.insert(0, r"D:\LLM\EasyEdit")

from easyeditor import BaseEditor, ROMEHyperParams


DATA_PATH = r"D:\LLM\data.json"
HPARAMS_PATH = r"D:\LLM\EasyEdit\hparams\ROME\gpt2-medium.yaml"


def generate_answer(model, tokenizer, prompt, max_new_tokens=20):
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
    answer = full_text[len(prompt):].strip()
    return answer, full_text


def hit(text, target):
    if text is None or target is None:
        return False
    return target.lower() in text.lower()


def main():
    # 屏蔽 EasyEdit / transformers 的普通日志
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("easyeditor").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 90)
    print("ROME Single Fact Editing Summary - gpt2-medium")
    print("=" * 90)
    print(f"Dataset: {DATA_PATH}")
    print(f"Total samples: {len(data)}")
    print("=" * 90)

    success_count = 0

    for idx, item in enumerate(data, start=1):
        prompt = item["prompt"]
        subject = item["subject"]
        ground_truth = item["ground_truth"]
        target_new = item["target_new"]

        # 每条样本重新加载模型，保证“逐条编辑，每次重置模型权重”
        hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            editor = BaseEditor.from_hparams(hparams)

        tokenizer = editor.tok
        model = editor.model

        # 编辑前输出
        before_answer, before_full_text = generate_answer(model, tokenizer, prompt)

        # 执行 ROME 编辑，屏蔽内部日志
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            metrics, edited_model, _ = editor.edit(
                prompts=[prompt],
                ground_truth=[ground_truth],
                target_new=[target_new],
                subject=[subject],
                keep_original_weight=False
            )

        # 编辑后输出
        after_answer, after_full_text = generate_answer(edited_model, tokenizer, prompt)

        before_hit = hit(before_answer, target_new)
        after_hit = hit(after_answer, target_new)

        if after_hit:
            success_count += 1

        print(f"\n[{idx}/10] Subject: {subject}")
        print(f"Prompt: {prompt}")
        print(f"Target new: {target_new}")
        print(f"Before edit: {before_answer}")
        print(f"Before hit target: {before_hit}")
        print(f"After edit: {after_answer}")
        print(f"After hit target: {after_hit}")

        try:
            print(f"EasyEdit rewrite_acc: {metrics[0]['pre']['rewrite_acc']} -> {metrics[0]['post']['rewrite_acc']}")
        except Exception:
            print("EasyEdit rewrite_acc: unavailable")

        print("-" * 90)

        del editor
        del model
        del edited_model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("Final Summary")
    print("=" * 90)
    print(f"Total samples: {len(data)}")
    print(f"Successful edits: {success_count}/{len(data)}")
    print(f"Success rate: {success_count / len(data) * 100:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()