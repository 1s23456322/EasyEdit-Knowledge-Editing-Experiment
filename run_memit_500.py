import json
import sys
import time
import gc
import io
import csv
import torch
import logging
import subprocess
import threading
import contextlib
from pathlib import Path


# 让 Python 找到 EasyEdit 源码
sys.path.insert(0, r"D:\LLM\EasyEdit")

from easyeditor import BaseEditor, MEMITHyperParams


DATA_PATH = Path(r"D:\LLM\counterfact_500_easyedit.json")
HPARAMS_PATH = r"D:\LLM\EasyEdit\hparams\MEMIT\gpt2-medium.yaml"

OUT_PATH = Path(r"D:\LLM\memit_500_results.json")
GPU_LOG_PATH = Path(r"D:\LLM\memit_gpu_log.csv")

# 你的 RTX 3050 Laptop 建议先用 50。
# 如果后续想尝试 100 或 500，改这里即可。
NUM_EDITS = 50

# 用前 20 条计算 ES / PS / NS，避免验证太慢
EVAL_COUNT = 20


def query_gpu_memory_mb():
    """
    使用 nvidia-smi 查询当前显存占用，单位 MB。
    """
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"
            ],
            encoding="utf-8"
        )
        first_line = result.strip().splitlines()[0]
        return int(first_line)
    except Exception:
        return None


class GPUMonitor:
    """
    后台周期性记录 GPU 显存。
    """
    def __init__(self, interval=1.0):
        self.interval = interval
        self.records = []
        self.running = False
        self.thread = None

    def _run(self):
        while self.running:
            mem = query_gpu_memory_mb()
            self.records.append({
                "time": time.time(),
                "gpu_memory_mb": mem
            })
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def peak_memory(self):
        values = [
            r["gpu_memory_mb"]
            for r in self.records
            if r["gpu_memory_mb"] is not None
        ]
        return max(values) if values else None

    def save_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "gpu_memory_mb"])
            writer.writeheader()
            writer.writerows(self.records)


def generate_answer(model, tokenizer, prompt, max_new_tokens=10):
    """
    给定 prompt，使用模型生成回答。
    返回：
    - answer: 去掉 prompt 后的新生成部分
    - full_text: prompt + answer
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
    answer = full_text[len(prompt):].strip()

    return answer, full_text


def contains(text, target):
    """
    判断 text 中是否包含 target。
    """
    if text is None or target is None:
        return False
    return target.lower() in text.lower()


def locality_preserved(before_answer, after_answer):
    """
    近似计算 NS / Locality。

    因为 counterfact_500_easyedit.json 中没有 locality_ground_truth，
    所以不能判断 locality_prompt 的“标准正确答案”是否保持正确。

    这里采用近似方法：
    - 先记录编辑前模型对 locality_prompt 的输出 before_answer
    - 再记录编辑后模型对 locality_prompt 的输出 after_answer
    - 如果 after_answer 中仍包含 before_answer 的第一个词，则认为局部性保持

    这个方法不完美，但适合当前数据格式下的课程实验。
    """
    if before_answer is None or after_answer is None:
        return False

    before_norm = before_answer.lower().strip()
    after_norm = after_answer.lower().strip()

    if before_norm == "" or after_norm == "":
        return False

    before_tokens = before_norm.split()
    if len(before_tokens) == 0:
        return False

    before_first_token = before_tokens[0]
    return before_first_token in after_norm


def main():
    # 减少 transformers / EasyEdit 普通日志
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("easyeditor").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:NUM_EDITS]

    prompts = [x["prompt"] for x in data]
    subjects = [x["subject"] for x in data]
    ground_truth = [x["ground_truth"] for x in data]
    target_new = [x["target_new"] for x in data]

    print("=" * 90)
    print("MEMIT Batch Editing - gpt2-medium")
    print("=" * 90)
    print("Dataset:", DATA_PATH)
    print("Number of edits:", len(data))
    print("Hparams:", HPARAMS_PATH)
    print("=" * 90)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gpu_before = query_gpu_memory_mb()

    monitor = GPUMonitor(interval=1.0)

    print("\nLoading model and MEMIT hparams...")

    hparams = MEMITHyperParams.from_hparams(HPARAMS_PATH)

    # 加载模型
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        editor = BaseEditor.from_hparams(hparams)

    tokenizer = editor.tok
    original_model = editor.model

    print("Model loaded.")
    print("GPU memory before editing:", gpu_before, "MB")

    # ------------------------------------------------------------
    # 编辑前先记录前 EVAL_COUNT 条 locality 输出
    # 用于后续计算 NS / Locality
    # ------------------------------------------------------------
    eval_count = min(EVAL_COUNT, len(data))
    baseline_locality_outputs = {}

    print(f"\nRecording baseline locality outputs for first {eval_count} samples...")

    for i, item in enumerate(data[:eval_count], start=1):
        locality_prompt = item.get("locality_prompt", "")

        if locality_prompt.strip() == "":
            before_answer = ""
            before_full_text = ""
        else:
            before_answer, before_full_text = generate_answer(
                original_model,
                tokenizer,
                locality_prompt,
                max_new_tokens=10
            )

        baseline_locality_outputs[item["id"]] = {
            "before_locality_answer": before_answer,
            "before_locality_full_text": before_full_text
        }

    print("Baseline locality recording finished.")

    # ------------------------------------------------------------
    # 执行 MEMIT 批量编辑
    # ------------------------------------------------------------
    print("\nRunning MEMIT batch editing...")

    start_time = time.time()
    monitor.start()

    error = None
    metrics = None
    edited_model = None

    try:
        # 如果你想看 MEMIT 内部日志，可以去掉这层 redirect。
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=target_new,
                subject=subjects,
                keep_original_weight=False
            )
    except Exception as e:
        error = repr(e)

    monitor.stop()
    end_time = time.time()

    elapsed = end_time - start_time
    gpu_after = query_gpu_memory_mb()
    gpu_peak_nvidia = monitor.peak_memory()

    if torch.cuda.is_available():
        torch_peak_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch_peak_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    else:
        torch_peak_allocated = None
        torch_peak_reserved = None

    monitor.save_csv(GPU_LOG_PATH)

    print("\nMEMIT editing finished.")
    print("Elapsed time:", f"{elapsed:.2f}", "seconds")
    print("GPU memory after editing:", gpu_after, "MB")
    print("Peak GPU memory by nvidia-smi:", gpu_peak_nvidia, "MB")
    print("Torch peak allocated:", torch_peak_allocated, "MB")
    print("Torch peak reserved:", torch_peak_reserved, "MB")

    # ------------------------------------------------------------
    # 验证前 EVAL_COUNT 条，计算 ES / PS / NS
    # ------------------------------------------------------------
    eval_results = []

    es_success = 0
    ps_success = 0
    ns_success = 0

    es_rate = None
    ps_rate = None
    ns_rate = None

    if edited_model is not None:
        print(f"\nValidating first {eval_count} edited facts for ES / PS / NS...")

        for i, item in enumerate(data[:eval_count], start=1):
            item_id = item["id"]

            prompt = item["prompt"]
            rephrase_prompt = item.get("rephrase_prompt", "")
            locality_prompt = item.get("locality_prompt", "")
            target = item["target_new"]

            # ES: 直接编辑 prompt 是否输出 target_new
            es_answer, es_full_text = generate_answer(
                edited_model,
                tokenizer,
                prompt,
                max_new_tokens=10
            )
            es_hit = contains(es_answer, target)

            # PS: 同义改写 prompt 是否输出 target_new
            if rephrase_prompt.strip() == "":
                ps_answer = ""
                ps_full_text = ""
                ps_hit = False
            else:
                ps_answer, ps_full_text = generate_answer(
                    edited_model,
                    tokenizer,
                    rephrase_prompt,
                    max_new_tokens=10
                )
                ps_hit = contains(ps_answer, target)

            # NS: 无关事实是否被破坏
            before_locality_answer = baseline_locality_outputs[item_id]["before_locality_answer"]
            before_locality_full_text = baseline_locality_outputs[item_id]["before_locality_full_text"]

            if locality_prompt.strip() == "":
                after_locality_answer = ""
                after_locality_full_text = ""
                ns_hit = False
            else:
                after_locality_answer, after_locality_full_text = generate_answer(
                    edited_model,
                    tokenizer,
                    locality_prompt,
                    max_new_tokens=10
                )
                ns_hit = locality_preserved(before_locality_answer, after_locality_answer)

            if es_hit:
                es_success += 1
            if ps_hit:
                ps_success += 1
            if ns_hit:
                ns_success += 1

            eval_results.append({
                "id": item_id,
                "subject": item["subject"],
                "prompt": prompt,
                "rephrase_prompt": rephrase_prompt,
                "locality_prompt": locality_prompt,
                "target_new": target,

                "es_answer": es_answer,
                "es_full_text": es_full_text,
                "es_hit": es_hit,

                "ps_answer": ps_answer,
                "ps_full_text": ps_full_text,
                "ps_hit": ps_hit,

                "before_locality_answer": before_locality_answer,
                "before_locality_full_text": before_locality_full_text,
                "after_locality_answer": after_locality_answer,
                "after_locality_full_text": after_locality_full_text,
                "ns_hit": ns_hit
            })

            print("-" * 90)
            print(f"[{i}/{eval_count}] Subject: {item['subject']}")
            print(f"Target new: {target}")
            print(f"ES answer: {es_answer}")
            print(f"ES hit: {es_hit}")
            print(f"PS answer: {ps_answer}")
            print(f"PS hit: {ps_hit}")
            print(f"Locality before: {before_locality_answer}")
            print(f"Locality after:  {after_locality_answer}")
            print(f"NS hit: {ns_hit}")

        es_rate = es_success / eval_count if eval_count > 0 else None
        ps_rate = ps_success / eval_count if eval_count > 0 else None
        ns_rate = ns_success / eval_count if eval_count > 0 else None

        print("\nEvaluation Metrics on first", eval_count, "samples")
        print("=" * 90)
        print(f"ES / Efficacy:       {es_success}/{eval_count} = {es_rate * 100:.2f}%")
        print(f"PS / Generalization: {ps_success}/{eval_count} = {ps_rate * 100:.2f}%")
        print(f"NS / Locality:       {ns_success}/{eval_count} = {ns_rate * 100:.2f}%")
        print("=" * 90)

    else:
        print("\nValidation skipped because edited_model is None.")

    # ------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------
    result = {
        "algorithm": "MEMIT",
        "model": "gpt2-medium",
        "dataset": str(DATA_PATH),
        "num_edits": len(data),

        "elapsed_seconds": elapsed,
        "gpu_memory_before_mb": gpu_before,
        "gpu_memory_after_mb": gpu_after,
        "gpu_memory_peak_nvidia_smi_mb": gpu_peak_nvidia,
        "torch_peak_allocated_mb": torch_peak_allocated,
        "torch_peak_reserved_mb": torch_peak_reserved,

        "validation_count": eval_count,

        "es_success": es_success,
        "ps_success": ps_success,
        "ns_success": ns_success,
        "es_rate": es_rate,
        "ps_rate": ps_rate,
        "ns_rate": ns_rate,

        "metrics": metrics,
        "error": error,
        "eval_results": eval_results
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nSaved result to:", OUT_PATH)
    print("Saved GPU log to:", GPU_LOG_PATH)

    # ------------------------------------------------------------
    # 最终 Summary，适合截图
    # ------------------------------------------------------------
    print("\n" + "=" * 90)
    print("Final Summary")
    print("=" * 90)
    print(f"Algorithm: MEMIT")
    print(f"Model: gpt2-medium")
    print(f"Batch edits: {len(data)}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Peak GPU memory: {gpu_peak_nvidia} MB")
    print(f"Torch peak allocated: {torch_peak_allocated} MB")
    print(f"Torch peak reserved: {torch_peak_reserved} MB")
    print(f"Validation samples: {eval_count}")

    if es_rate is not None:
        print(f"ES / Efficacy:       {es_success}/{eval_count} = {es_rate * 100:.2f}%")
        print(f"PS / Generalization: {ps_success}/{eval_count} = {ps_rate * 100:.2f}%")
        print(f"NS / Locality:       {ns_success}/{eval_count} = {ns_rate * 100:.2f}%")
    else:
        print("ES / Efficacy:       N/A")
        print("PS / Generalization: N/A")
        print("NS / Locality:       N/A")

    print(f"Error: {error}")
    print("=" * 90)

    try:
        del editor
        del edited_model
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()