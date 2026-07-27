import json
from langchain_groq import ChatGroq
from judge_prompts import judge_prompt
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

judge = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


def evaluate_with_judge(inputs, outputs, reference_outputs):
    """
    Calls the LLM judge and returns all evaluation metrics.
    """
    messages = judge_prompt.format_messages(
        question=inputs["Query"],
        reference=reference_outputs["Response"],
        context=outputs["context"],
        prediction=outputs["Response"],
    )

    response = judge.invoke(messages)

    text = response.content.strip()

    # Remove markdown if model returns ```json
    text = text.replace("```json", "")
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)

    except Exception:
        return {
            "correctness": {
                "score": 0,
                "reason": "Judge returned invalid JSON."
            },
            "faithfulness": {
                "score": 0,
                "reason": "Judge returned invalid JSON."
            },
            "relevance": {
                "score": 0,
                "reason": "Judge returned invalid JSON."
            },
            "citation_quality": {
                "score": 0,
                "reason": "Judge returned invalid JSON."
            }
        }


def evaluate_with_ragas(inputs, outputs, reference_outputs):
    """
    Evaluate using RAGAS metrics.
    """
    # Prepare dataset for RAGAS
    dataset = Dataset.from_dict({
        "question": [inputs["Query"]],
        "answer": [outputs["Response"]],
        "contexts": [[doc.page_content for doc in reference_outputs.get("sources", [])]],
        "ground_truth": [reference_outputs.get("Response", "")]
    })
    
    # Run RAGAS evaluation
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=judge
    )
    
    return {
        "faithfulness_score": result["faithfulness_score"].iloc[0],
        "answer_relevancy_score": result["answer_relevancy_score"].iloc[0],
        "context_recall_score": result["context_recall_score"].iloc[0]
    }


def correctness(inputs, outputs, reference_outputs):
    result = evaluate_with_judge(inputs, outputs, reference_outputs)
    return {
        "key": "correctness",
        "score": result["correctness"]["score"],
        "comment": result["correctness"]["reason"]
    }


def faithfulness(inputs, outputs, reference_outputs):
    result = evaluate_with_judge(inputs, outputs, reference_outputs)
    return {
        "key": "faithfulness",
        "score": result["faithfulness"]["score"],
        "comment": result["faithfulness"]["reason"]
    }


def relevance(inputs, outputs, reference_outputs):
    result = evaluate_with_judge(inputs, outputs, reference_outputs)
    return {
        "key": "relevance",
        "score": result["relevance"]["score"],
        "comment": result["relevance"]["reason"]
    }


def citation_quality(inputs, outputs, reference_outputs):
    result = evaluate_with_judge(inputs, outputs, reference_outputs)
    return {
        "key": "citation_quality",
        "score": result["citation_quality"]["score"],
        "comment": result["citation_quality"]["reason"]
    }


def answer_not_empty(inputs, outputs):
    answer = outputs.get("Response", "")
    return {
        "key": "answer_not_empty",
        "score": len(answer.strip()) > 0
    }