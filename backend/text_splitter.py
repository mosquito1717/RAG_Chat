import re
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document

class TimestampTextSplitter(TextSplitter):
    def split_text(self, text: str):
        # 타임스탬프 패턴 정의 (예: [00:01])
        pattern = r"\[\d{2}:\d{2}\]"
        matches = list(re.finditer(pattern, text))

        if not matches:
            return [text]

        chunks = []
        start_idx = 0

        for match in matches:
            end_idx = match.start()
            if start_idx != end_idx:
                chunks.append(text[start_idx:end_idx].strip())
            start_idx = end_idx

        if start_idx < len(text):
            chunks.append(text[start_idx:].strip())

        return chunks

# 텍스트 정제 및 분할 함수
def clean_text_and_split(text):
    """텍스트 정제 후 타임스탬프 기준으로 분할"""

    # 문단 앞뒤 큰따옴표 제거
    def clean_paragraph(paragraph):
        return re.sub(r'^"(.*)"$', r'\1', paragraph.strip())

    # 문단별 정제
    paragraphs = text.split("\n")
    cleaned_paragraphs = [clean_paragraph(p) for p in paragraphs]
    cleaned_text = "".join(cleaned_paragraphs)

    # 텍스트 분할
    splitter = TimestampTextSplitter()
    chunks = splitter.split_text(cleaned_text)

    # 리스트의 각 문자열 뒤에 붙은 ;를 제거
    cleaned_chunks = [s.rstrip(";") for s in chunks]
    
    
    # 텍스트 리스트를 Document 객체 리스트로 변환
    docs = [Document(page_content=text) for text in cleaned_chunks]

    return docs