"""
Evaluates generated answers against retrieved context using RAGAS metrics:
- faithfulness: is the answer grounded in the retrieved context (no hallucination)?
- answer_relevancy: does the answer actually address the question?
- context_precision: is the retrieved context relevant / well-ranked?
- context_recall: did retrieval surface everything needed (requires ground-truth)?
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


def build_eval_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> Dataset:
    data = {"question": questions, "answer": answers, "contexts": contexts}
    if ground_truths is not None:
        data["ground_truth"] = ground_truths
    return Dataset.from_dict(data)


def run_ragas_eval(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
):
    dataset = build_eval_dataset(questions, answers, contexts, ground_truths)

    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truths is not None:
        metrics.append(context_recall)

    result = evaluate(dataset, metrics=metrics)
    return result.to_pandas()


if __name__ == "__main__":
    # Minimal smoke test with toy data
    df = run_ragas_eval(
        questions=["What is the refund policy?"],
        answers=["Refunds are available within 30 days of purchase."],
        contexts=[["Our refund policy allows returns within 30 days of the purchase date."]],
        ground_truths=["Refunds within 30 days."],
    )
    print(df)
