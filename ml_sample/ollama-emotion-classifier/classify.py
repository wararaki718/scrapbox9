import ollama

from schema import EmotionResult


PROMPT = """以下の英語ツイートを読み、当てはまる感情を次の11種類からすべて選択してください（複数選択可・該当なしは空リスト）。
感情一覧: {emotion_columns}

ツイート:
\"\"\"
{text}
\"\"\"
"""


def classify(text: str, emotion_columns: list[str], model_name: str="qwen3:8b") -> list[dict[str, str]]:
    prompt = PROMPT.format(text=text, emotion_columns=emotion_columns)
    options = {
        "temperature": 0.0,
        "num_ctx": 4096,
    }
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "あなたは高精度な感情マルチラベル分類器です。指定されたJSONスキーマに従ってのみ回答してください。",
            },
            {"role": "user", "content": prompt},
        ],
        format=EmotionResult.model_json_schema(),
        options=options,
        think=False,
    )
    result = EmotionResult.model_validate_json(response.message.content)
    return list(dict.fromkeys(result.labels))
