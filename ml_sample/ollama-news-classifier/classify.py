import ollama

from schema import ClassificationResult


PROMPT = """以下のテキストを次の4つのカテゴリのうち最も適切な1つに分類してください。
カテゴリ一覧: {categories}

テキスト:
\"\"\"
{text}
\"\"\"
"""


def classify(text: str, categories: list[str], model_name: str="qwen3:8b") -> ClassificationResult:
    prompt = PROMPT.format(text=text, categories=categories)
    options = {"temperature": 0.0, "num_ctx": 4096}

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "あなたは高精度なテキスト分類器です。指定されたJSONスキーマに従ってのみ回答してください。",
            },
            {"role": "user", "content": prompt},
        ],
        format=ClassificationResult.model_json_schema(),
        options=options,
        think=False,
    )

    return ClassificationResult.model_validate_json(response.message.content)
