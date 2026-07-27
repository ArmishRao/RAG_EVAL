from langchain_core.prompts import ChatPromptTemplate

judge_prompt = ChatPromptTemplate.from_template("""
You are an expert Pakistani legal evaluator.

Evaluate the generated answer.

Question:
{question}

Reference Answer:
{reference}

Retrieved Context:
{context}

Generated Answer:
{prediction}

Evaluate the following metrics.

1. Correctness
2. Faithfulness
3. Relevance
4. Citation Quality

Scoring:
1 = Very Poor
10 = Excellent

Return ONLY valid JSON.

{{
    "correctness": {{
        "score": 9,
        "reason": "..."
    }},

    "faithfulness": {{
        "score": 10,
        "reason": "..."
    }},

    "relevance": {{
        "score": 8,
        "reason": "..."
    }},

    "citation_quality": {{
        "score": 9,
        "reason": "..."
    }}
}}

No markdown.

Only JSON.

""")