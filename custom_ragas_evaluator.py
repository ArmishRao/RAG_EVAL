import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize the judge LLM
judge = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Custom prompt templates for RAGAS-style evaluation
faithfulness_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator. Determine if the answer is factually consistent with the context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Score from 0 to 1:
0 = Completely unfaithful (contradicts context or makes up information)
1 = Completely faithful (fully supported by context)

Return ONLY a JSON object with score and reason:
{{
    "score": 0.8,
    "reason": "Brief explanation"
}}
""")

answer_relevancy_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator. Determine how relevant the answer is to the question.

Question:
{question}

Answer:
{answer}

Score from 0 to 1:
0 = Completely irrelevant (doesn't address the question)
1 = Perfectly relevant (directly answers the question)

Return ONLY a JSON object with score and reason:
{{
    "score": 0.9,
    "reason": "Brief explanation"
}}
""")

context_relevancy_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator. Determine how relevant the retrieved context is to the question.

Context:
{context}

Question:
{question}

Score from 0 to 1:
0 = Completely irrelevant context
1 = Perfectly relevant context

Return ONLY a JSON object with score and reason:
{{
    "score": 0.85,
    "reason": "Brief explanation"
}}
""")

answer_correctness_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator. Compare the generated answer with the ground truth.

Question:
{question}

Ground Truth:
{ground_truth}

Generated Answer:
{answer}

Score from 0 to 1:
0 = Completely incorrect
1 = Perfectly correct

Return ONLY a JSON object with score and reason:
{{
    "score": 0.75,
    "reason": "Brief explanation"
}}
""")


def evaluate_faithfulness(question, answer, context):
    """Evaluate faithfulness using LLM judge."""
    try:
        messages = faithfulness_prompt.format_messages(
            context=context,
            question=question,
            answer=answer
        )
        response = judge.invoke(messages)
        # Clean the response
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result.get("score", 0.0)
    except Exception as e:
        print(f"Error in faithfulness evaluation: {e}")
        return 0.0


def evaluate_answer_relevancy(question, answer):
    """Evaluate answer relevancy using LLM judge."""
    try:
        messages = answer_relevancy_prompt.format_messages(
            question=question,
            answer=answer
        )
        response = judge.invoke(messages)
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result.get("score", 0.0)
    except Exception as e:
        print(f"Error in answer relevancy evaluation: {e}")
        return 0.0


def evaluate_context_relevancy(question, context):
    """Evaluate context relevancy using LLM judge."""
    if not context or context.strip() == "":
        return 0.0
    
    try:
        messages = context_relevancy_prompt.format_messages(
            context=context[:2000],  # Limit context length
            question=question
        )
        response = judge.invoke(messages)
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result.get("score", 0.0)
    except Exception as e:
        print(f"Error in context relevancy evaluation: {e}")
        return 0.0


def evaluate_answer_correctness(question, answer, ground_truth):
    """Evaluate answer correctness using LLM judge."""
    if not ground_truth or ground_truth.strip() == "":
        return None
    
    try:
        messages = answer_correctness_prompt.format_messages(
            question=question,
            ground_truth=ground_truth,
            answer=answer
        )
        response = judge.invoke(messages)
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result.get("score", 0.0)
    except Exception as e:
        print(f"Error in answer correctness evaluation: {e}")
        return None


def evaluate_with_custom_ragas(question, answer, contexts, ground_truth=None):
    """
    Evaluate RAG response using custom RAGAS-style metrics.
    
    Args:
        question: The user's question
        answer: The generated answer
        contexts: List of context strings
        ground_truth: Optional reference answer
    
    Returns:
        Dictionary with all metrics
    """
    # Combine contexts
    combined_context = "\n\n".join(contexts) if contexts else ""
    
    # Evaluate each metric
    print("  Evaluating faithfulness...")
    faithfulness_score = evaluate_faithfulness(question, answer, combined_context)
    
    print("  Evaluating answer relevancy...")
    answer_relevancy_score = evaluate_answer_relevancy(question, answer)
    
    print("  Evaluating context relevancy...")
    context_relevancy_score = evaluate_context_relevancy(question, combined_context)
    
    results = {
        "faithfulness_score": faithfulness_score,
        "answer_relevancy_score": answer_relevancy_score,
        "context_relevancy_score": context_relevancy_score,
    }
    
    # Add correctness if ground truth is provided
    if ground_truth:
        print("  Evaluating answer correctness...")
        correctness_score = evaluate_answer_correctness(question, answer, ground_truth)
        if correctness_score is not None:
            results["answer_correctness_score"] = correctness_score
    
    return results


def evaluate_batch(questions, answers, contexts_list, ground_truths=None):
    """
    Evaluate multiple samples.
    """
    results = []
    
    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Evaluating: {question[:50]}...")
        answer = answers[i]
        contexts = contexts_list[i]
        ground_truth = ground_truths[i] if ground_truths else None
        
        scores = evaluate_with_custom_ragas(question, answer, contexts, ground_truth)
        scores["question"] = question
        results.append(scores)
    
    return results