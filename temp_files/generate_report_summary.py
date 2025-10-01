import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent

PUBMED_OUTPUT = ROOT / "result_on_server" / "runs" / "outputs-trl-llama32-1b-pubmedqa" / "first_finetune_results.csv"
DREDDIT_OUTPUT = ROOT / "result_on_server" / "runs" / "outputs-trl-llama32-1b-dreaddit" / "second_fine_tuneresults.csv"
FORGETTING_OUTPUT = ROOT / "result_on_server" / "runs" / "outputs-trl-llama32-1b-dreaddit" / "predictions_pubmed_val_from_dreaddit.csv"
PUBMED_TRAINER = ROOT / "result_on_server" / "runs" / "outputs-trl-llama32-1b-pubmedqa" / "checkpoint-123" / "trainer_state.json"
DREDDIT_TRAINER = ROOT / "result_on_server" / "runs" / "outputs-trl-llama32-1b-dreaddit" / "checkpoint-426" / "trainer_state.json"


UNKNOWN = "Unknown"


def normalize_pubmed(value: object) -> str:
    if value is None:
        return UNKNOWN
    text = str(value).strip()
    if not text:
        return UNKNOWN
    low = text.lower()
    letter_map = {"a": "Yes", "b": "No", "c": "Maybe"}
    keyword_map = {"yes": "Yes", "no": "No", "maybe": "Maybe"}

    if low in keyword_map:
        return keyword_map[low]
    if low in letter_map:
        return letter_map[low]

    keyword_match = re.search(r"\b(yes|no|maybe)\b", low)
    if keyword_match:
        return keyword_map[keyword_match.group(1)]

    letter_match = re.search(r"\b([abc])\b", low)
    if letter_match:
        return letter_map[letter_match.group(1)]

    first_token = re.split(r"[\s\.,;:!\?\)\(\[\]\{\}]+", text)[0].strip("()[]{}:.- ")
    if not first_token:
        return UNKNOWN
    low_token = first_token.lower()
    if low_token in letter_map:
        return letter_map[low_token]
    if low_token in keyword_map:
        return keyword_map[low_token]
    return first_token


def normalize_dreaddit(value: object) -> str:
    if value is None:
        return UNKNOWN
    text = str(value).strip()
    if not text:
        return UNKNOWN
    low = text.lower()
    if "no stress" in low or low.startswith("no-stress"):
        return "No Stress"
    if re.search(r"\bstress\b", low):
        return "Stress"
    letter_match = re.search(r"\b([ab])\b", low)
    if letter_match:
        return "Stress" if letter_match.group(1) == "a" else "No Stress"
    return text


def map_dreaddit_label(value: object) -> str:
    if value is None:
        return UNKNOWN
    try:
        return "Stress" if int(float(value)) == 1 else "No Stress"
    except Exception:
        return normalize_dreaddit(value)


def calc_metrics(labels: List[str], preds: List[str]) -> Tuple[float, float, List[Dict[str, object]], Dict[str, Dict[str, int]]]:
    total = len(labels)
    accuracy = sum(1 for gt, pr in zip(labels, preds) if gt == pr) / total if total else 0.0
    label_order = sorted(set(labels) | set(preds))
    confusion = {label: {p_label: 0 for p_label in label_order} for label in label_order}
    for gt, pr in zip(labels, preds):
        confusion.setdefault(gt, {})
        confusion[gt][pr] = confusion[gt].get(pr, 0) + 1

    per_class = []
    macro_sum = 0.0
    active_classes = 0
    for label in label_order:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion.get(other, {}).get(label, 0) for other in label_order if other != label)
        fn = sum(confusion[label].get(other, 0) for other in label_order if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[label].values())
        if support:
            active_classes += 1
            macro_sum += f1
        per_class.append({
            "label": label,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        })
    macro_f1 = macro_sum / active_classes if active_classes else 0.0
    confusion_int = {
        label: {pred_label: int(count) for pred_label, count in preds_dict.items()}
        for label, preds_dict in confusion.items()
    }
    return accuracy, macro_f1, per_class, confusion_int


def summarize_split(
    path: Path,
    label_key: str,
    pred_key: str,
    label_norm: Callable[[object], str],
    pred_norm: Callable[[object], str],
) -> Dict[str, object]:
    if not path.exists():
        return {}

    labels: List[str] = []
    preds: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels.append(label_norm(row.get(label_key)))
            preds.append(pred_norm(row.get(pred_key)))

    accuracy, macro_f1, per_class, confusion = calc_metrics(labels, preds)
    return {
        "samples": len(labels),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "label_distribution": {k: int(v) for k, v in Counter(labels).items()},
        "prediction_distribution": {k: int(v) for k, v in Counter(preds).items()},
        "per_class": per_class,
        "confusion": confusion,
    }


def summarize(pubmed_path: Path, dreaddit_path: Path, forgetting_path: Path) -> Dict[str, object]:
    report: Dict[str, object] = {}

    pubmed_summary = summarize_split(pubmed_path, "final_decision", "prediction", normalize_pubmed, normalize_pubmed)
    if pubmed_summary:
        report["pubmed_test"] = pubmed_summary

    dreaddit_summary = summarize_split(dreaddit_path, "label", "prediction", map_dreaddit_label, normalize_dreaddit)
    if dreaddit_summary:
        report["dreaddit_test"] = dreaddit_summary

    forgetting_summary = summarize_split(
        forgetting_path, "final_decision", "prediction", normalize_pubmed, normalize_pubmed
    )
    if forgetting_summary:
        report["pubmed_val_after_dreaddit"] = forgetting_summary

    return report


def summarize_training(state_path: Path) -> Dict[str, object]:
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as fh:
        state = json.load(fh)
    history = state.get("log_history", [])
    train_logs = [entry for entry in history if "loss" in entry]
    eval_logs = [entry for entry in history if "eval_loss" in entry]

    summary: Dict[str, object] = {
        "epochs": state.get("num_train_epochs"),
        "global_steps": state.get("global_step"),
    }

    if train_logs:
        last = train_logs[-1]
        summary["final_train"] = {
            key: round(float(last[key]), 4)
            for key in ("loss", "mean_token_accuracy", "entropy", "grad_norm")
            if key in last
        }
    if eval_logs:
        best_eval = min(eval_logs, key=lambda item: item.get("eval_loss", float("inf")))
        summary["best_eval"] = {
            key: round(float(best_eval[key]), 4)
            for key in ("eval_loss", "eval_mean_token_accuracy", "eval_entropy")
            if key in best_eval
        }
        summary["best_eval"]["step"] = best_eval.get("step")
    return summary


def main() -> None:
    data_report = summarize(PUBMED_OUTPUT, DREDDIT_OUTPUT, FORGETTING_OUTPUT)
    training_report = {
        "pubmed_training": summarize_training(PUBMED_TRAINER),
        "dreaddit_training": summarize_training(DREDDIT_TRAINER),
    }
    payload = {
        "metrics": data_report,
        "training": training_report,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
