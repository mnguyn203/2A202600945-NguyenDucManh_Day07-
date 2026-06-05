import json
import re
from typing import Tuple, List, Dict, Any
from src.chunking import RecursiveChunker

class ContextAwareCodeChunker:
    """
    Custom Chunker for Phase 2:
    - Bóc tách Metadata (Bơm ngữ cảnh chống mất gốc).
    - Cắt file theo thẻ tiêu đề Markdown (##, ###).
    - Nhận diện và KHÔNG BAO GIỜ cắt đứt code block (```python ... ```).
    - Gắn liền đoạn giải thích ngay phía trên vào code block.
    """
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.fallback_chunker = RecursiveChunker(chunk_size=chunk_size)
        
    def extract_metadata(self, text: str) -> Tuple[Dict[str, Any], str]:
        # Tách JSON kẹp giữa 2 dấu --- ở đầu file
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            content = match.group(2)
            try:
                metadata = json.loads(json_str)
                return metadata, content
            except json.JSONDecodeError:
                return {}, text
        return {}, text

    def split_by_markdown_headers(self, text: str) -> List[Tuple[str, str]]:
        # Regex tìm header ở đầu dòng
        pattern = re.compile(r'^(#{1,3})\s+(.*)$', re.MULTILINE)
        
        matches = list(pattern.finditer(text))
        if not matches:
            return [("General", text)]
            
        sections = []
        
        # Phần intro trước header đầu tiên
        first_header_start = matches[0].start()
        if first_header_start > 0:
            intro = text[:first_header_start].strip()
            if intro:
                sections.append(("Intro", intro))
                
        for i, match in enumerate(matches):
            header = match.group(2).strip()
            start_pos = match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            section_content = text[start_pos:end_pos].strip()
            if section_content:
                sections.append((header, section_content))
                
        return sections

    def protect_code_blocks(self, section_text: str) -> List[str]:
        # Phân tách code blocks và text thường bằng regex (?s) cho phép . match newline
        pattern = re.compile(r'(```.*?```)', re.DOTALL)
        parts = pattern.split(section_text)
        
        blocks = []
        current_block = ""
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
                
            if part.startswith("```"):
                # Đây là code block -> Ghép chặt với text giải thích phía trước
                current_block += "\n\n" + part
                blocks.append(current_block.strip())
                current_block = ""
            else:
                # Đây là text thường
                if current_block:
                    blocks.append(current_block.strip())
                current_block = part
                
        if current_block:
            blocks.append(current_block.strip())
            
        return blocks

    def chunk(self, text: str) -> List[str]:
        metadata, clean_text = self.extract_metadata(text)
        
        doc_title = metadata.get('doc_title', 'Unknown Document')
        keywords = metadata.get('keywords', [])
        tags = ", ".join(keywords) if keywords else ""
        
        prefix = f"[Ngữ cảnh: {doc_title}]"
        if tags:
            prefix += f" [Từ khóa: {tags}]"
            
        sections = self.split_by_markdown_headers(clean_text)
        
        chunks = []
        for header, content in sections:
            blocks = self.protect_code_blocks(content)
            
            current_chunk = ""
            
            for block in blocks:
                # Nếu ghép vào vẫn bé hơn chunk_size
                if len(current_chunk) + len(block) < self.chunk_size:
                    current_chunk += "\n\n" + block if current_chunk else block
                else:
                    # Nếu quá bự thì chốt sổ chunk hiện tại
                    if current_chunk:
                        full_chunk = f"{prefix} > {header}\n\n{current_chunk}"
                        chunks.append(full_chunk)
                    
                    # Xử lý block mới (có thể block này bản thân nó quá lớn -> fallback)
                    if len(block) > self.chunk_size:
                        sub_chunks = self.fallback_chunker.chunk(block)
                        for sc in sub_chunks:
                            chunks.append(f"{prefix} > {header}\n\n{sc}")
                        current_chunk = ""
                    else:
                        current_chunk = block
                        
            # Xử lý chunk dư cuối cùng
            if current_chunk:
                full_chunk = f"{prefix} > {header}\n\n{current_chunk}"
                chunks.append(full_chunk)
                
        return chunks
