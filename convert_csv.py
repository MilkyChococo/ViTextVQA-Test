import csv
import json
from pathlib import Path


TEST_INPUT_FILE = Path(r"E:\project\vqa_kltn_vitexxt\outputs\predictions\vitextvqa_test_qwen.json")
DEV_INPUT_FILE = Path(r"E:\project\vqa_kltn_vitexxt\outputs\predictions\vitextvqa_dev_qwen.json")
SAMPLE_SUBMISSION_FILE = Path(r"E:\project\vqa_kltn_vitexxt\outputs\predictions\sample_submission.csv")
OUTPUT_FILE = Path(r"E:\project\vqa_kltn_vitexxt\outputs\predictions\submission.csv")
EMPTY_PREDICTION_ANSWER = "Không đủ thông tin"
MISSING_ID_ANSWER = "NONEEE"


def load_answer_map(json_path: Path) -> dict[str, str]:
    if not json_path.exists():
        return {}

    data = json.loads(json_path.read_text(encoding="utf-8"))
    answer_by_id: dict[str, str] = {}
    for ann in data.get("annotations", []):
        qid = str(ann.get("id")).strip()
        answers = ann.get("answers", [])
        answer = answers[0] if len(answers) > 0 else ""
        answer = str(answer).strip()
        if not answer:
            answer = EMPTY_PREDICTION_ANSWER
        if qid:
            answer_by_id[qid] = answer
    return answer_by_id


def main() -> None:
    test_answers = load_answer_map(TEST_INPUT_FILE)
    dev_answers = load_answer_map(DEV_INPUT_FILE)

    merged_answers = dict(dev_answers)
    merged_answers.update(test_answers)

    rows: list[list[str]] = []
    sample_ids: list[str] = []
    missing_ids: list[str] = []

    with SAMPLE_SUBMISSION_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["ID"]).strip()
            sample_ids.append(qid)
            answer = merged_answers.get(qid, MISSING_ID_ANSWER)
            if qid not in merged_answers:
                missing_ids.append(qid)
            rows.append([qid, answer])

    sample_id_set = set(sample_ids)
    extra_test_ids = sorted(qid for qid in test_answers if qid not in sample_id_set)
    extra_dev_ids = sorted(qid for qid in dev_answers if qid not in sample_id_set)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Answer"])
        writer.writerows(rows)

    print(f"Done! Saved to {OUTPUT_FILE}")
    print(f"sample_ids={len(sample_ids)}")
    print(f"test_prediction_ids={len(test_answers)}")
    print(f"dev_prediction_ids={len(dev_answers)}")
    print(f"merged_prediction_ids={len(merged_answers)}")
    print(f"missing_ids_in_prediction={len(missing_ids)}")
    if missing_ids:
        print("missing_id_list=")
        for qid in missing_ids:
            print(qid)
    print(f"extra_test_prediction_ids_not_in_sample={len(extra_test_ids)}")
    if extra_test_ids:
        print("extra_test_prediction_id_list=")
        for qid in extra_test_ids:
            print(qid)
    print(f"extra_dev_prediction_ids_not_in_sample={len(extra_dev_ids)}")
    if extra_dev_ids:
        print("extra_dev_prediction_id_list=")
        for qid in extra_dev_ids:
            print(qid)
    if not DEV_INPUT_FILE.exists():
        print(f"dev_prediction_file_missing={DEV_INPUT_FILE}")


if __name__ == "__main__":
    main()
