import pymupdf
import ollama
import numpy
import faiss
from pathlib import Path

def split_text(text,chunk_size=512,overlap=128):
    chunks=[]
    start=0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def cosine_similarity(a, b):
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

BASE_DIR=Path(__file__).parent
pdf_path=BASE_DIR / "test.pdf"
doc=pymupdf.open(pdf_path)
full_text=""
for page_number,page in enumerate(doc):
    text=page.get_text()
    full_text += text
doc.close()

chunks=split_text(full_text)

embeddings = []
for i,chunk in enumerate(chunks):
    response = ollama.embed(model="bge-m3", input=chunk)
    embeddings.append(response['embeddings'][0])
embeddings=numpy.array(embeddings,dtype='float32')

faiss.normalize_L2(embeddings)
dimension=embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

memory=""


while True:
    query=input("请输入你的问题: ")
    print(f"问题: {query}")
    memory+= f"用户: {query}\n"

    response = ollama.embed(model="bge-m3", input=query)
    query_embedding = numpy.array(response['embeddings'][0], dtype='float32')
    faiss.normalize_L2(query_embedding.reshape(-1, 1))

    top_k=min(5, len(chunks))
    scores,indices = index.search(query_embedding.reshape(1, -1), top_k)
    context=''
    for i in range(top_k):
        context += chunks[indices[0][i]] + "\n\n"


    # result = []
    # for i, embedding in enumerate(embeddings):
    #     similarity = cosine_similarity(query_embedding, embedding)
    #     result.append((similarity, i))

    # result.sort(key=lambda x: x[0], reverse=True)

    # top_k = min(5, len(result))
    # context=''
    # for i in range(top_k):
    #     context += chunks[result[i][1]]+"\n\n"

    prompt =f"""
    你是一个PDF学习助手，能够根据用户提供的PDF文档内容回答问题。
    【PDF相关内容】
    {context}

    【用户问题】
    {query}

    【历史对话】
    {memory}

    要求：
    1.优先根据PDF内容回答问题。
    2.如果PDF内容中没有相关信息，请明确告知用户。
    3.不要编造PDF中不存在的信息。
    """

    response=ollama.chat(model="qwen2.5", messages=[{"role": "user", "content": prompt}])
    answer=response['message']['content']
    memory += f"AI: {answer}\n"
    print("="*9)
    print(f"{answer}")
    print("="*9)