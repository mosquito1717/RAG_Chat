# vectorDB에서 유사도만 갖고 검색하는 코드 만들기
import chardet
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from text_splitter import clean_text_and_split
from apikey import API_KEY
import os

os.environ["OPENAI_API_KEY"] = API_KEY

## 다른 폴더에 있는 데이터 가져오기
    
# 파일 경로 지정
# 현재 실행 중인 스크립트 파일의 절대 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
# 상위 디렉토리의 "database" 폴더 경로
database_dir = os.path.join(current_dir, "..", "database")
# 사용할 파일의 전체 경로
file_path = os.path.join(database_dir, "Sejong_In_2024Jan.txt")


    
def load_txt(raw_data):
    encoding_result = chardet.detect(raw_data)
    detected_encoding = encoding_result["encoding"]
    raw_text = raw_data.decode(detected_encoding)
    return raw_text

# 파일 내용 읽기
with open(file_path, "rb") as f:
    raw_data = f.read()
raw_text = load_txt(raw_data)

# 텍스트 스플리팅
def text_splitter(_raw_text):
    return clean_text_and_split(_raw_text)

# 데이터를 분할해 vectorDB에 저장하기 - vectorstore
split_docs = text_splitter(raw_text)
# metadata에 시간을 넣고 content에는 text만 넣는 게 좋을 것 같아

# vectorstore 정의 - 벡터 임베딩 모델 사용
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    split_docs, 
    OpenAIEmbeddings(model='text-embedding-3-small'),
    persist_directory=persist_directory
)
query = "발열"
query_embedding = OpenAIEmbeddings(model='text-embedding-3-small').embed_query(query)

results = vectorstore.similarity_search_by_vector(query_embedding, k=100)

for i, doc in enumerate(results):
    print(f"문서 {i+1}:")
    print(f"내용: {doc.page_content}")
    print(f"메타데이터: {doc.metadata}")
    print("-" * 50)

#print(f"현재 벡터 개수: {vectorstore._collection.count()}")