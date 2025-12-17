import openai
import os

# 请确保已经 export OPENAI_API_KEY=xxxx
openai.api_key = os.environ.get("OPENAI_API_KEY")  # 或写死字符串
openai.api_base = "https://yinli.one/v1"

def test_chat():
    print("Testing ChatCompletion API...")

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say hello in one short sentence."}
            ],
            max_tokens=32,
            temperature=0.7
        )
        print("API 调用成功 ✅")
        print("模型回复:", resp.choices[0].message.content)

    except Exception as e:
        print("❌ API 调用失败：")
        print(e)

if __name__ == "__main__":
    test_chat()
