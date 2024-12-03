import os

import chainlit as cl
from dotenv import load_dotenv
from langchain_community.document_loaders import GitLoader
from openai import OpenAI

# .envファイルのパス
dotenv_path = '/workspaces/chatbot-demo/.env'
load_dotenv(dotenv_path)

# OpenAI APIキーを環境変数から取得
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Make sure it is set in the .env file.")


# OpenAIクライアントの設定
client = OpenAI(api_key=api_key)


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()
print(len(documents))
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)


from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = db.as_retriever()


@cl.on_chat_start
async def start():
    await cl.Message(content='何か質問してください。').send()

settings = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    "presence_penalty": 1.0,
    # ... more settings
}


@cl.on_message
async def on_message(message: cl.Message):
    # ユーザーのメッセージに基づいて、キャラクター情報を組み込む
    context = (
        f"{retriever.invoke(message.content)}\n\n"
        "あなたはとても親切な関西人の秘書です。"
        "おしゃべりが好きで詳細な情報を伝えてくれます。"
        "第一人称は「うち」。"
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": message.content}
        ],
        **settings
    )
    await cl.Message(content=response.choices[0].message.content).send()
