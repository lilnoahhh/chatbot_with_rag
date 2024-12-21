from typing import List
import os
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from openai import OpenAI
import chainlit as cl
import re

# Environment variables setup
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USERNAME = os.environ.get('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Neo4j graph初期化
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=[". ", ", ", "\n", " "] 
)

@cl.on_chat_start
async def on_chat_start():
    files = None

    # ユーザーからのファイルのアップロードを待つ
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    try:
        # PDFから取得
        documents = extract_text_from_pdf(file.path)
        
        if not documents:
            raise ValueError("Could not extract text from PDF")

        # テキストのチャンク化
        split_documents = text_splitter.split_documents(documents)
        
        # knowledge graphの作成
        create_knowledge_graph(split_documents)
        
        msg.content = f"Finished processing `{file.name}`. Ask me anything!"
        await msg.update()

    except Exception as e:
        error_message = f"Error during initialization: {str(e)}"
        await cl.Message(content=error_message).send()

def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """Extract text from English PDF"""
    documents = []
    laparams = LAParams(detect_vertical=False, all_texts=True)
    
    for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=laparams), start=1):
        page_text = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text += element.get_text()
        
        if page_text.strip():
            metadata = {"page": page_num, "source": f"page_{page_num}"}
            doc = Document(page_content=page_text.strip(), metadata=metadata)
            documents.append(doc)
            
    return documents

def create_knowledge_graph(documents: List[Document]):
    """知識グラフの作成（Neo4j Aura対応版）"""
    try:
        # 既存のデータをクリア
        graph.query("MATCH (n) DETACH DELETE n")
        
        # ドキュメントノードの作成
        for doc in documents:
            # ドキュメントノードを作成
            graph.query("""
            CREATE (d:Document {
                text: $text,
                page: $page,
                source: $source
            })
            """, {
                "text": doc.page_content,
                "page": doc.metadata["page"],
                "source": doc.metadata["source"]
            })
            
            # キーワードの抽出と関連付け
            keywords = extract_keywords(doc.page_content)
            for keyword in keywords:
                graph.query("""
                MERGE (k:Keyword {text: $keyword})
                WITH k
                MATCH (d:Document {page: $page})
                CREATE (d)-[:HAS_KEYWORD]->(k)
                """, {
                    "keyword": keyword,
                    "page": doc.metadata["page"]
                })
        
        # 連続するページ間の関係を作成
        graph.query("""
        MATCH (d1:Document), (d2:Document)
        WHERE d1.page = d2.page - 1
        CREATE (d1)-[:NEXT]->(d2)
        """)
        
        print(f"知識グラフを作成しました: {len(documents)}ページ")
        
    except Exception as e:
        print(f"知識グラフ作成中にエラー: {str(e)}")
        raise

def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text"""
    try:
        # OpenAI APIを使用してキーワードを抽出
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract 5 important keywords (mainly nouns and proper nouns) from the text. Return only the keywords separated by commas."},
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        
        keywords = response.choices[0].message.content.split(',')
        return [k.strip() for k in keywords if k.strip()]
        
    except Exception as e:
        print(f"キーワード抽出中にエラー: {str(e)}")
        return []

@cl.on_message
async def main(message: cl.Message):
    try:
        # 関連するドキュメントの取得
        relevant_docs = search_documents(message.content)
        if not relevant_docs:
            await cl.Message(content="No relevant information found.").send()
            return

        # 参照要素を作成
        text_elements = []
        for doc in relevant_docs:
            page_num = doc.metadata["page"]
            relevance = doc.metadata["relevance"]
            context = doc.metadata["context"]
            
            # 前後のページ情報を含む参照テキストを作成
            context_info = []
            if context["prev_page"]:
                context_info.append(f"Previous page: {context['prev_page']}")
            if context["next_page"]:
                context_info.append(f"Next page: {context['next_page']}")
            
            context_text = "\n".join(context_info)
            
            text_elements.append(
                cl.Text(
                    name=f"source_{page_num}",
                    content=f"""
📄 Page {page_num} (Relevance: {relevance})
{context_text}
---
{doc.page_content}
                    """.strip(),
                    display="side"
                )
            )

        # プロンプト生成
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""
        Please answer the question based on the following context:
        
        {context}
        
        Question: {message.content}
        """

        # レスポンスの生成
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        # 参照元の提示
        source_info = [f"p.{doc.metadata['page']}(relevance:{doc.metadata['relevance']})" 
                      for doc in relevant_docs]
        answer += f"\n\nSources: {', '.join(source_info)}"
        
        await cl.Message(content=answer, elements=text_elements).send()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await cl.Message(content=error_message).send()

def search_documents(query: str, k: int = 3) -> List[Document]:
    """ドキュメント検索（Neo4j Aura対応版）"""
    try:
        # クエリからキーワードを抽出
        query_keywords = extract_keywords(query)
        
        # キーワードベースの検索
        results = graph.query("""
        // キーワードを含むドキュメントを検索
        MATCH (d:Document)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE k.text IN $keywords
        
        // キーワードの一致数をカウント
        WITH d, COUNT(DISTINCT k) as keyword_matches
        
        // 前後のページも含める
        OPTIONAL MATCH (d)<-[:NEXT]-(prev:Document)
        OPTIONAL MATCH (d)-[:NEXT]->(next:Document)
        
        // 結果を返す
        RETURN 
            d.text as text,
            d.page as page,
            d.source as source,
            keyword_matches as relevance,
            prev.page as prev_page,
            next.page as next_page
        ORDER BY keyword_matches DESC, d.page
        LIMIT $limit
        """, {
            "keywords": query_keywords,
            "limit": k
        })
        
        documents = []
        for r in results:
            # メタデータの作成
            metadata = {
                "page": r["page"],
                "source": r["source"],
                "relevance": r["relevance"],
                "context": {
                    "prev_page": r["prev_page"],
                    "next_page": r["next_page"]
                }
            }
            
            doc = Document(
                page_content=r["text"],
                metadata=metadata
            )
            documents.append(doc)
            
            print(f"検索結果: ページ {r['page']} (関連度: {r['relevance']})")
        
        return documents
        
    except Exception as e:
        print(f"検索中にエラー: {str(e)}")
        raise

