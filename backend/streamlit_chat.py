import os
import streamlit as st
import chardet
from text_splitter import clean_text_and_split  # text_splitter.py에서 함수 가져오기

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from prompt_config import SYSTEM_PROMPT
from apikey import API_KEY

os.environ["OPENAI_API_KEY"] = API_KEY

@st.cache_resource
def load_txt(raw_data):
    encoding_result = chardet.detect(raw_data)
    detected_encoding = encoding_result["encoding"]
    
    try:
        raw_text = raw_data.decode(detected_encoding)
    except UnicodeDecodeError:
        st.error(f"파일을 {detected_encoding} 인코딩으로 읽을 수 없습니다.")
        st.stop()
    
    st.success(f"✅ {uploaded_file.name} 파일이 성공적으로 처리되었습니다!")
    return raw_text
    
# 텍스트 정제 및 분할
# TextSplitter 역할
def text_splitter(_raw_text):
    return clean_text_and_split(_raw_text) # Document 객체 return

# 시스템 프롬프트 가져오기
def get_system_prompt():
    return SYSTEM_PROMPT

# 벡터 저장소 생성 및 로드
@st.cache_resource
def create_vector_store(_raw_text):
    split_docs = text_splitter(_raw_text)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs, 
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=persist_directory
    )
    return vectorstore

@st.cache_resource
def get_vectorstore(_raw_text):
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
        )
    return create_vector_store(_raw_text)

# Document 객체의 page_content를 Join
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 시스템 설정
@st.cache_resource
def chaining(_raw_text):
    # 업로드한 데이터가 담긴 리스트 : raw_text
    vectorstore = get_vectorstore(_raw_text)
    
    # vectorstore에서 문서를 검색할 수 있는 retriever 객체 생성
    # 검색 요청이 들어오면 벡터 유사도를 기반으로 관련 문서 반환
    retriever = vectorstore.as_retriever()

    # prompt_config.py에서 SYSTEM_PROMPT 가져오기
    qa_system_prompt = get_system_prompt()

    # 프롬프트 템플릿 생성
    # 시스템 프롬프트와 사용자 입력을 결합
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    # LLM 모델 초기화
    llm = ChatOpenAI(model="gpt-4o-mini")

    # RAG 파이프라인 구축
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    """
    retriever로 관련 문서 검색
    format_docs로 검색된 문서들을 하나의 문자열로 변환
    
    context: 검색된 문서를 LLM이 이해할 수 있도록 전달
    input: 사용자의 질문을 그대로 전달
    
    | qa_prompt: context(검색된 문서)와 input(사용자 질문)을 프롬프트에 삽입
    | llm: GPT-4o-mini 모델이 응답을 생성
    | StrOutputParser(): 응답을 문자열로 변환
    """

    return rag_chain # RAG 검색 + LLM 응답 생성 파이프라인(rag_chain) 반환

# Streamlit UI
st.title("💬 Chatbot with File Upload")

# 파일 업로드 기능 추가
uploaded_file = st.file_uploader("파일 업로드 (txt)", type=["txt"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    # 파일 인코딩 감지 및 읽기
    raw_data = uploaded_file.read()
    raw_text = load_txt(raw_data)
    
    rag_chain = chaining(raw_text)
    vectorstore = get_vectorstore(raw_text) # 벡터 스토어 로드
    retriever = vectorstore.as_retriever() # RAG용 검색 엔진

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
        st.chat_message("human").write(prompt_message)
        st.session_state.messages.append({"role": "user", "content": prompt_message})
        
        # vectorStore만 사용한 검색 결과
        with st.expander("🔍 VectorStore 검색 결과 보기"):
            with st.spinner("Retrieving relevant documents..."):
                docs = vectorstore.similarity_search(prompt_message)
                st.write("🔹 **검색된 문서:**")
                st.write(format_docs(docs))
        
        # RAG만 사용한 검색 결과
        with st.expander("🔍 RAG 검색 결과 보기"):
            with st.spinner("Retrieving relevant documents..."):
                docs = retriever.get_relevant_documents(prompt_message)
                st.write("🔹 **검색된 문서:**")
                st.write(format_docs(docs))
        
        # GPT 포함 RAG 결과 (기존 방식)
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt_message)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)

