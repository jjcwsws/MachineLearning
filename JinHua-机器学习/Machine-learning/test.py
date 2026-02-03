from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# ================ 1. 加载文档 ================
documents = []  # 用 LangChain 的 Document loader 加载
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    r"D:\PycharmProjects\Machine-learning\medical_docs",  # ← 绝对路径，加 r
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()

# ================ 2. 文本分割 ================
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# ================ 3. 加载 embedding 模型 ================
from sentence_transformers import SentenceTransformer

model_path = r"D:\PycharmProjects\Machine-learning\models\bge-large-zh-v1.5"

print("正在加载 sentence-transformers 模型（本地路径）...")
st_model = SentenceTransformer(
    model_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=False  # 防止它尝试加载远程代码
)

print("模型加载成功！维度:", st_model.get_sentence_embedding_dimension())


# # 手动包装成 LangChain 兼容的 embeddings 对象
# class LocalBGEEmbeddings:
#     def __init__(self, model):
#         self.model = model
#
#     def embed_documents(self, texts):
#         # texts 是 list[str]，返回 list[list[float]]
#         embeddings = self.model.encode(
#             texts,
#             normalize_embeddings=True,  # bge 系列必须归一化
#             show_progress_bar=True,  # 可选，显示进度
#             convert_to_numpy=True,
#             batch_size=32  # 根据显存调整
#         )
#         return embeddings.tolist()
#
#     def embed_query(self, text):
#         # 单条查询，返回 list[float]
#         embedding = self.model.encode(
#             [text],
#             normalize_embeddings=True,
#             convert_to_numpy=True
#         )[0]
#         return embedding.tolist()
#
#
# # 实例化
# embeddings = LocalBGEEmbeddings(st_model)


# 手动包装成 LangChain 兼容的 embeddings 对象（修复 callable 问题）

from langchain_core.embeddings import Embeddings

class LocalBGEEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,  # 批量时关掉进度条，避免干扰
            convert_to_numpy=True,
            batch_size=32
        )
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        return embedding.tolist()

    # 关键修复：让类实例可调用，兼容 LangChain 的 embedding_function 调用
    def __call__(self, text):
        # LangChain 内部会调用 embeddings(text)，这里转发到 embed_query
        return self.embed_query(text)

# 实例化（保持不变）
embeddings = LocalBGEEmbeddings(st_model)



# # ================ 4. 诊断信息 ================
# print("\n" + "=" * 50)
# print("=== 诊断信息 开始 ===")
#
# print("documents 数量（原始加载的文档）:", len(documents))
# if len(documents) > 0:
#     print("第一篇文档预览（前200字符）:", documents[0].page_content[:200])
#     print("第一篇文档元数据:", documents[0].metadata)
# else:
#     print("警告：documents 为空！loader.load() 没有加载到任何文档")
#
# print("chunks 数量（切分后的片段）:", len(chunks))
# if len(chunks) > 0:
#     print("第一个 chunk 内容预览:", chunks[0].page_content[:100])
#     print("第一个 chunk 元数据:", chunks[0].metadata)
# else:
#     print("警告：chunks 为空！可能是 splitter 切分失败或 documents 本来就空")
#
# # 测试 embedding 是否能正常生成向量
# if len(chunks) > 0:
#     try:
#         test_vector = embeddings.embed_query(chunks[0].page_content)
#         print("测试 embedding 成功，向量维度:", len(test_vector))
#         print("向量前5个值:", test_vector[:5])
#     except Exception as e:
#         print("embedding 生成失败:", str(e))
# else:
#     print("跳过 embedding 测试，因为 chunks 为空")
#
# print("=== 诊断信息 结束 ===")
# print("=" * 50 + "\n")

# ================ 5. 创建向量数据库 ================
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index_medical")  # 保存，下次直接 load
#
# print("FAISS 索引创建成功！")
# print("索引中的向量数量:", vectorstore.index.ntotal)  # 应该等于 chunks 数量
# print("索引维度:", vectorstore.index.d)

# 创建 retriever 时，显式传 embed_query 函数（而不是整个对象）
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
    embedding_function=embeddings.embed_query   # ← 关键：传函数，而不是 embeddings 对象
)


# # ================ 6. 加载 LLM 模型 ================
# print("正在加载 LLM (Qwen2-1.5B-Instruct)...")
#
# model_name = "Qwen/Qwen2-1.5B-Instruct"
#
# try:
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         trust_remote_code=True
#     )
#
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         torch_dtype=torch.float16,  # 使用 fp16
#         low_cpu_mem_usage=True,
#         trust_remote_code=True
#     )
#
#     model.eval()
#
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=256,
#         temperature=0.7,
#         do_sample=True,
#         batch_size=1
#     )
#
#     llm = HuggingFacePipeline(pipeline=pipe)
#
#     print("LLM 加载成功！")
#     print("模型实际运行设备:", next(model.parameters()).device)
#
# except Exception as e:
#     print(f"LLM 加载失败: {e}")
#     print("正在尝试使用本地缓存的模型...")

# print("正在加载 LLM (Qwen2-1.5B-Instruct)...")
# model_name = "Qwen/Qwen2-1.5B-Instruct"
# try:
#     tokenizer = AutoTokenizer.from_pretrained( model_name, trust_remote_code=True )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True
#     )
#     model.eval()
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=256,
#         temperature=0.7,
#         do_sample=True,
#         batch_size=1
#     )
#     llm = HuggingFacePipeline(pipeline=pipe)
#     print("LLM 加载成功！")
#     print("模型实际运行设备:", next(model.parameters()).device)
# except Exception as e:
#     print(f"LLM 加载失败: {e}")
#     # 尝试从本地缓存加载
#     try:
#         # 先查找本地缓存路径
#         cache_path = r"C:\Users\YF\.cache\huggingface\hub"
#         # 或者使用 transformers 的默认缓存
#         from transformers.utils import TRANSFORMERS_CACHE
#
#         print(f"Transformers 缓存目录: {TRANSFORMERS_CACHE}")
#
#         # 列出缓存中的模型
#         import os
#
#         if os.path.exists(cache_path):
#             for item in os.listdir(cache_path):
#                 if "Qwen" in item or "qwen" in item:
#                     print(f"找到可能的 Qwen 模型: {item}")
#
#         # 尝试强制本地加载
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             local_files_only=True,
#             trust_remote_code=True
#         )
#
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             local_files_only=True,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True
#         )
#
#         model.eval()
#         pipe = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=256,
#             temperature=0.7,
#             do_sample=True
#         )
#         llm = HuggingFacePipeline(pipeline=pipe)
#         print("从本地缓存加载 LLM 成功！")
#
#     except Exception as e2:
#         print(f"本地缓存加载也失败: {e2}")
#         print("建议：请先单独下载模型，或使用更小的模型")
#         print("可以使用以下命令在终端中先下载模型：")
#         print(f'python -c "from transformers import AutoModel; model = AutoModel.from_pretrained(\'{model_name}\')"')
#         llm = None


# ================ 6. 加载 LLM 模型（本地离线） ================
print("正在加载本地 LLM (Qwen2-1.5B-Instruct)...")
llm_model_path = r"D:\PycharmProjects\Machine-learning\models\Qwen2-1.5B-Instruct"  # ← 这里就是你的本地路径

try:
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_path,
        local_files_only=True,          # 强制只用本地，不要联网
        trust_remote_code=True
    )
    print("tokenizer 加载成功")

    model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        local_files_only=True,
        dtype=torch.float16,      # 如果报错，改成 torch.bfloat16 或 torch.float32
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("模型权重加载成功，device_map:", model.hf_device_map)

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
    print("LLM 完整加载成功！")
    print("模型实际运行设备:", next(model.parameters()).device)

except Exception as e:
    print(f"LLM 加载失败: {e}")
    print("请检查以下几点：")
    print("1. 路径是否完全正确？", llm_model_path)
    print("2. 文件夹里是否有 config.json、model.safetensors（或 pytorch_model.bin）、tokenizer.json 等文件？")
    print("3. 显存是否足够？（1.5B 模型需约 3–4GB 空闲显存）")
    print("4. 尝试改 torch_dtype=torch.float32 测试兼容性")
    llm = None

# ================ 7. 创建 RAG 链 ================
if llm is not None:
    from langchain_core.prompts import ChatPromptTemplate

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # 检索 top-4 相关 chunk
    )

    prompt = ChatPromptTemplate.from_template(
        """你是一个专业的医疗知识助手。请严格基于以下检索到的上下文内容，用中文回答用户的问题。
        如果上下文里没有足够信息，就诚实地说"根据现有资料无法完整回答"或"我不确定"，不要编造或猜测。

        上下文：
        {context}

        问题：{input}

        回答（清晰、简洁、专业、用中文）："""
    )

    # 构建 chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("RAG chain 构建成功！")


    # ================ 8. 测试函数 ================
    def ask(question):
        print("\n" + "=" * 80)
        print(f"问题：{question}")
        try:
            result = rag_chain.invoke({"input": question})
            print("答案：")
            print(result.get("answer", result.get("output", "无答案")))
            print("\n检索到的上下文（前200字符）：")
            for i, doc in enumerate(result["context"], 1):
                print(f"[{i}] 来源：{doc.metadata.get('source', '未知')}")
                print(f"   {doc.page_content[:200]}...")
                print("-" * 60)
        except Exception as e:
            print("查询失败：", str(e))
            import traceback
            traceback.print_exc()
        print("=" * 80 + "\n")


    # ================ 9. 立即测试 ================
#     ask("糖尿病有哪些常见症状？")
#     ask("高血压患者可以吃什么水果？")
# else:
#     print("LLM 未加载，无法创建 RAG 链。请先解决模型加载问题。")


# ================ 交互式问答模式 ================
if llm is not None:
    print("\n" + "="*80)
    print("RAG 系统已就绪！请输入你的医疗问题（输入 'quit' 或 '退出' 结束）：")
    print("="*80)

    while True:
        question = input("你：")
        if question.lower() in ['quit', '退出', 'q', 'exit']:
            print("再见！")
            break
        if not question.strip():
            continue

        print("\n" + "-"*80)
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
        print("-"*80 + "\n")
else:
    print("LLM 未加载，无法进入问答模式。请先解决模型加载问题。")