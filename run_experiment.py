import os
import sys
import glob
from src.chunking import FixedSizeChunker
from src.custom_chunker import ContextAwareCodeChunker
from src.models import Document
from src.store import EmbeddingStore

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("--- BƯỚC 1: CHUẨN BỊ DATA TỪ THƯ MỤC DATA ---")
    data_dir = "data"
    files = glob.glob(os.path.join(data_dir, "python_*.md"))
    
    raw_docs = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            raw_docs.append(Document(
                id=os.path.basename(fpath),
                content=f.read()
            ))
            
    print(f"-> Đã tải xong {len(raw_docs)} file tài liệu Python Markdown.")
    
    print("\n--- BƯỚC 2: TIẾN HÀNH CHUNKING VỚI CHIẾN THUẬT ĐỘC QUYỀN ---")
    chunker = ContextAwareCodeChunker(chunk_size=1000)
    
    chunked_docs = []
    for doc in raw_docs:
        chunks = chunker.chunk(doc.content)
        for i, c in enumerate(chunks):
            # Biến mỗi chunk thành 1 object Document để cho vào store
            chunked_docs.append(Document(
                id=f"{doc.id}_chunk_{i}",
                content=c,
                metadata={"doc_id": doc.id, "source": doc.id}
            ))
            
    print(f"-> Đã băm nhỏ thành {len(chunked_docs)} chunks siêu chất lượng!")
    
    print("\n--- BƯỚC 3: ĐẨY DỮ LIỆU VÀO VECTOR STORE ---")
    store = EmbeddingStore()
    store.add_documents(chunked_docs)
    print(f"-> Store đã sẵn sàng với {store.get_collection_size()} vector.")
    
    # 5 câu hỏi benchmark
    queries = [
        "Python list comprehension khác gì so với for loop khi tạo list mới?",
        "Python module import hoạt động như thế nào, và Module Search Path ảnh hưởng gì?",
        "try-except-finally / exception handling trong Python xử lý lỗi ra sao?",
        "Class, instance, attribute, method trong Python OOP khác nhau thế nào?",
        "Virtual environment và pip giúp quản lý dependency conflict như thế nào?"
    ]
    
    print("\n" + "="*50)
    print("           BẮT ĐẦU CHẠY BENCHMARK")
    print("="*50)
    
    for i, q in enumerate(queries, 1):
        print(f"\n[Câu hỏi {i}]: {q}")
        results = store.search(q, top_k=1)
        if results:
            print(f" -> TÌM THẤY TÀI LIỆU CHUẨN NHẤT: {results[0]['id']}")
            print(" --- NỘI DUNG CHUNK CHÍNH XÁC ---")
            print(f"{results[0]['content'][:300]}...\n[...]")
        else:
            print(" -> (Không tìm thấy kết quả nào)")

if __name__ == "__main__":
    main()
