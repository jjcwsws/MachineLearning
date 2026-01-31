from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

documents = []  # 用 LangChain 的 Document loader 加载
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    r"D:\PycharmProjects\Machine-learning\medical_docs",  # ← 绝对路径，加 r
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

from sentence_transformers import SentenceTransformer
import torch

# ------------------ 替换原来的 embeddings 创建 ------------------
model_path = r"D:\PycharmProjects\Machine-learning\models\bge-large-zh-v1.5"

print("正在加载 sentence-transformers 模型（本地路径）...")
st_model = SentenceTransformer(
    model_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=False  # 防止它尝试加载远程代码
)

print("模型加载成功！维度:", st_model.get_sentence_embedding_dimension())

# 手动包装成 LangChain 兼容的 embeddings 对象
class LocalBGEEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # texts 是 list[str]，返回 list[list[float]]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,          # bge 系列必须归一化
            show_progress_bar=True,             # 可选，显示进度
            convert_to_numpy=True,
            batch_size=32                       # 根据显存调整
        )
        return embeddings.tolist()

    def embed_query(self, text):
        # 单条查询，返回 list[float]
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        return embedding.tolist()

# 实例化
embeddings = LocalBGEEmbeddings(st_model)

# 测试一下
test_text = "糖尿病是一种慢性代谢性疾病"
test_vec = embeddings.embed_query(test_text)
print("测试 embedding 成功，向量维度:", len(test_vec))
print("向量前5个值:", test_vec[:5])


# ------------------- 加在这里的诊断打印 -------------------
print("\n" + "="*50)
print("=== 诊断信息 开始 ===")

print("documents 数量（原始加载的文档）:", len(documents))
if len(documents) > 0:
    print("第一篇文档预览（前200字符）:", documents[0].page_content[:200])
    print("第一篇文档元数据:", documents[0].metadata)
else:
    print("警告：documents 为空！loader.load() 没有加载到任何文档")

print("chunks 数量（切分后的片段）:", len(chunks))
if len(chunks) > 0:
    print("第一个 chunk 内容预览:", chunks[0].page_content[:100])
    print("第一个 chunk 元数据:", chunks[0].metadata)
else:
    print("警告：chunks 为空！可能是 splitter 切分失败或 documents 本来就空")

# 测试 embedding 是否能正常生成向量
if len(chunks) > 0:
    try:
        test_vector = embeddings.embed_query(chunks[0].page_content)
        print("测试 embedding 成功，向量维度:", len(test_vector))
        print("向量前5个值:", test_vector[:5])
    except Exception as e:
        print("embedding 生成失败:", str(e))
else:
    print("跳过 embedding 测试，因为 chunks 为空")

print("=== 诊断信息 结束 ===")
print("="*50 + "\n")
# ------------------- 诊断打印结束 -------------------



vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index_medical")  # 保存，下次直接 load


print("FAISS 索引创建成功！")
print("索引中的向量数量:", vectorstore.index.ntotal)  # 应该等于 chunks 数量 = 7
print("索引维度:", vectorstore.index.d)               # 应该 = 1024


# model_name = "Qwen/Qwen2-1.5B-Instruct"
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     low_cpu_mem_usage=True
# )
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     temperature=0.7,
#     do_sample=True,
#     device_map="auto"
# )
# llm = HuggingFacePipeline(pipeline=pipe)


# from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

retriever = vectorstore.as_retriever(
    search_type="similarity",  # 或 "mmr"（多样性更好）
    search_kwargs={"k": 4}  # 检索 top-4 相关 chunk
)

prompt = ChatPromptTemplate.from_template(
    """你是一个专业的医疗知识助手。请严格基于以下检索到的上下文内容，用中文回答用户的问题。
    如果上下文里没有足够信息，就诚实地说“根据现有资料无法完整回答”或“我不确定”，不要编造或猜测。

    上下文：
    {context}

    问题：{input}

    回答（清晰、简洁、专业、用中文）："""
)

# # 先加载 llm（放在 chain 之前）
# print("正在加载 LLM (Qwen2-1.5B-Instruct)...")
# model_name = "Qwen/Qwen2-1.5B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     low_cpu_mem_usage=True
# )
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=256,  # 先小一点测试，避免 OOM
#     temperature=0.7,
#     do_sample=True
# )
# llm = HuggingFacePipeline(pipeline=pipe)
# print("LLM 加载成功！")
#
# # 构建 chain
# combine_docs_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
#
# print("RAG chain 构建成功！")

print("正在加载 LLM (Qwen2-1.5B-Instruct)...")

model_name = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    batch_size=1
)

llm = HuggingFacePipeline(pipeline=pipe)

print("LLM 加载成功！")
print("模型实际运行设备:", next(model.parameters()).device)




def ask(question):
    print("\n" + "="*80)
    print(f"问题：{question}")
    try:
        result = rag_chain.invoke({"input": question})
        print("答案：")
        print(result.get("answer", result.get("output", "无答案")))
        print("\n检索到的上下文（前200字符）：")
        for i, doc in enumerate(result["context"], 1):
            print(f"[{i}] 来源：{doc.metadata.get('source', '未知')}")
            print(f"   {doc.page_content[:200]}...")
            print("-"*60)
    except Exception as e:
        print("查询失败：", str(e))
    print("="*80 + "\n")

ask("糖尿病有哪些常见症状？")
ask("高血压患者可以吃什么水果？")