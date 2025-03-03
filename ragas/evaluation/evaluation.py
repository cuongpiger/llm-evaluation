from typing import List
from ragas import EvaluationDataset, SingleTurnSample


def prepare_evaluation_dataset(questions: List[str], ground_truths: List[str], rag, retriever):
    answers = []
    contexts = []

    for query in questions:
        answers.append(rag.invoke(query))
        contexts.append([docs.page_content for docs in retriever.invoke(query)])

    # Convert dict to dataset
    return EvaluationDataset(samples=[SingleTurnSample(
        user_input=questions[i],
        retrieved_contexts=contexts[i],
        response=answers[i],
        reference=ground_truths[i]
    ) for i in range(len(questions))])