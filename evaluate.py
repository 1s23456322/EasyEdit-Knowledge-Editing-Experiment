import json
from pathlib import Path


# 这里读取你 Task 3 生成的结果文件
RESULT_PATH = Path(r"D:\LLM\memit_500_results.json")

# 输出评估汇总文件
OUT_PATH = Path(r"D:\LLM\evaluation_summary.json")


def calculate_rate(success_count, total_count):
    if total_count == 0:
        return 0.0
    return success_count / total_count


def main():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        result = json.load(f)

    eval_results = result.get("eval_results", [])

    if not eval_results:
        print("No eval_results found in result file.")
        print("Please make sure run_memit_500.py has finished validation.")
        return

    total = len(eval_results)

    es_success = sum(1 for x in eval_results if x.get("es_hit") is True)
    ps_success = sum(1 for x in eval_results if x.get("ps_hit") is True)
    ns_success = sum(1 for x in eval_results if x.get("ns_hit") is True)

    es_rate = calculate_rate(es_success, total)
    ps_rate = calculate_rate(ps_success, total)
    ns_rate = calculate_rate(ns_success, total)

    summary = {
        "algorithm": result.get("algorithm", "MEMIT"),
        "model": result.get("model", "gpt2-medium"),
        "num_edits": result.get("num_edits"),
        "validation_count": total,

        "ES_success": es_success,
        "ES_total": total,
        "ES_rate": es_rate,

        "PS_success": ps_success,
        "PS_total": total,
        "PS_rate": ps_rate,

        "NS_success": ns_success,
        "NS_total": total,
        "NS_rate": ns_rate,

        "elapsed_seconds": result.get("elapsed_seconds"),
        "gpu_memory_peak_nvidia_smi_mb": result.get("gpu_memory_peak_nvidia_smi_mb"),
        "torch_peak_allocated_mb": result.get("torch_peak_allocated_mb"),
        "torch_peak_reserved_mb": result.get("torch_peak_reserved_mb"),
        "error": result.get("error")
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Comprehensive Evaluation Results")
    print("=" * 80)
    print(f"Algorithm: {summary['algorithm']}")
    print(f"Model: {summary['model']}")
    print(f"Batch edits: {summary['num_edits']}")
    print(f"Validation samples: {total}")
    print("-" * 80)
    print(f"ES / Efficacy:       {es_success}/{total} = {es_rate * 100:.2f}%")
    print(f"PS / Generalization: {ps_success}/{total} = {ps_rate * 100:.2f}%")
    print(f"NS / Locality:       {ns_success}/{total} = {ns_rate * 100:.2f}%")
    print("-" * 80)
    print(f"Elapsed time: {summary['elapsed_seconds']} seconds")
    print(f"Peak GPU memory: {summary['gpu_memory_peak_nvidia_smi_mb']} MB")
    print(f"Torch peak allocated: {summary['torch_peak_allocated_mb']} MB")
    print(f"Torch peak reserved: {summary['torch_peak_reserved_mb']} MB")
    print(f"Error: {summary['error']}")
    print("=" * 80)
    print(f"Saved summary to: {OUT_PATH}")


if __name__ == "__main__":
    main()