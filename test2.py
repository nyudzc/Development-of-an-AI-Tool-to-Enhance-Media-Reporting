import pandas as pd
import time
from sklearn.metrics import classification_report, confusion_matrix
from openai import OpenAI

# ✅ 初始化 OpenAI 客户端（填入你的 OpenAI API Key）
client = OpenAI(api_key="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")

# ✅ 定义对话请求
def query_openai(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个仇恨言论识别助手。请判断以下文本是否包含 hate speech,只返回 hate 或 non-hate 这两个标签，不要附加解释。"
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[EXCEPTION] {str(e)}"

# ✅ 加载测试集
df = pd.read_excel("Test Dataset Spanish.xlsx")
df = df[['id', 'comment_text', 'toxic', 'severe_toxic']].copy()

# ✅ 构造真实标签：toxic 或 severe_toxic 为 1 即为 hate
df["label_true"] = df.apply(lambda row: 1 if row["toxic"] == 1 or row["severe_toxic"] == 1 else 0, axis=1)

# ✅ 开始预测
predictions = []
raw_responses = []

for i, row in df.iterrows():
    text = str(row["comment_text"]) if pd.notna(row["comment_text"]) else ""

    result = query_openai(text)
    raw_responses.append(result)

    result_lower = result.lower()
    if "hate" in result_lower and not "non-hate" in result_lower:
        predictions.append(1)
        print(f"✅ Processing {i+1}/{len(df)} → hate")
    elif "non-hate" in result_lower:
        predictions.append(0)
        print(f"✅ Processing {i+1}/{len(df)} → non-hate")
    else:
        predictions.append(None)
        print(f"⚠️ Processing {i+1}/{len(df)} → 无法提取标签 | 返回: {result[:100]}")

    time.sleep(1.1)  # 限流防止超速

# ✅ 保存预测结果
df["predicted"] = predictions
df["openai_raw"] = raw_responses
df.to_excel("evaluation_openai_predictions.xlsx", index=False)

# ✅ 输出评估结果
valid_df = df[df["predicted"].notna()]
print("\n✅ 有效预测条数:", len(valid_df))

report = classification_report(valid_df["label_true"], valid_df["predicted"])
conf_matrix = confusion_matrix(valid_df["label_true"], valid_df["predicted"])

print("\n=== Classification Report ===\n")
print(report)

print("\n=== Confusion Matrix ===\n")
print(conf_matrix)

# ✅ 保存报告
with open("classification_report_openai.txt", "w", encoding="utf-8") as f:
    f.write(report)
