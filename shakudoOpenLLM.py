
EMB_INSTRUCTOR_XL = "hkunlp/instructor-xl"
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"

LLM_FLAN_T5_XXL = "google/flan-t5-xxl"
LLM_FLAN_T5_XL = "google/flan-t5-xl"
LLM_FASTCHAT_T5_XL = "lmsys/fastchat-t5-3b-v1.0"
LLM_FLAN_T5_SMALL = "google/flan-t5-small"
LLM_FLAN_T5_BASE = "google/flan-t5-base"
LLM_FLAN_T5_LARGE = "google/flan-t5-large"
LLM_FALCON_SMALL = "tiiuae/falcon-7b-instruct"

def create_sbert_mpnet():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})


def create_flan_t5_base(load_in_8bit=False):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )


if config["embedding"] == EMB_SBERT_MPNET_BASE:
    embedding = create_sbert_mpnet()
load_in_8bit = config["load_in_8bit"]
if config["llm"] == LLM_FLAN_T5_BASE:
    llm = create_flan_t5_base(load_in_8bit=load_in_8bit)

def create_falcon_instruct_small(load_in_8bit=False):
        model = "tiiuae/falcon-7b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                tokenizer = tokenizer,
                trust_remote_code = True,
                max_new_tokens=100,
                model_kwargs={
                    "device_map": "auto", 
                    "load_in_8bit": load_in_8bit, 
                    "max_length": 512, 
                    "temperature": 0.01,
                    "torch_dtype":torch.bfloat16,
                    }
            )
        return hf_pipeline


#Step 1: Ingesting the Data into Vector Store (ChromaDB)

# Load the pdf
pdf_path = "wiki_data_short.pdf"
loader = PDFPlumberLoader(pdf_path)
documents = loader.load()

# Split documents and create text snippets
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
texts = text_splitter.split_documents(texts)

persist_directory = config["persist_directory"]
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory

#Step 2: Retrieving Snippets and Prompt Engineering

hf_llm = HuggingFacePipeline(pipeline=llm)
retriever = vectordb.as_retriever(search_kwargs={"k":4})
qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",retriever=retriever)

# Defining a default prompt for flan models
if config["llm"] == LLM_FLAN_T5_SMALL or config["llm"] == LLM_FLAN_T5_BASE or config["llm"] == LLM_FLAN_T5_LARGE:
    question_t5_template = """
    context: {context}
    question: {question}
    answer: 
    """
    QUESTION_T5_PROMPT = PromptTemplate(
        template=question_t5_template, input_variables=["context", "question"]
    )
    qa.combine_documents_chain.llm_chain.prompt = QUESTION_T5_PROMPT

#Step 3: Querying the LLM

question = "what's the reason for financial crisis?"
qa.combine_documents_chain.verbose = True
qa.return_source_documents = True
qa({"query":question,})

#Packaging into a Class

class PdfQA:
    def __init__(self,config:dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None

    ...
# Check out the full script on the Github link on the intro

#We can now initialize and run the PdfQA class with the following code:

# Configuration for PdfQA
config = {"persist_directory":None,
          "load_in_8bit":False,
          "embedding" : EMB_SBERT_MPNET_BASE,
          "llm":LLM_FLAN_T5_BASE,
          "pdf_path":"wiki_data_short.pdf"
          }

# Initialize PdfQA
pdfqa = PdfQA(config=config)
pdfqa.init_embeddings()
pdfqa.init_models()

# Create Vector DB 
pdfqa.vector_db_pdf()

# Set up Retrieval QA Chain
pdfqa.retreival_qa_chain()

# Query the model
question = "what the reason for financial crisis?"
pdfqa.answer_query(question)

#Step 4: Building the Streamlit app 

import streamlit as st
from pdf_qa import PdfQA
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import shutil
from constants import * ## constants.py file can be found in code

# Streamlit app code
st.set_page_config(
    page_title='Q&A Bot for PDF',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
)


if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]:PdfQA = PdfQA() ## Intialisation

## To cache resource across multiple session 
@st.cache_resource
def load_llm(llm,load_in_8bit):

    if llm == LLM_OPENAI_GPT35:
        pass
    elif llm == LLM_FLAN_T5_SMALL:
        return PdfQA.create_flan_t5_small(load_in_8bit)
    elif llm == LLM_FLAN_T5_BASE:
        return PdfQA.create_flan_t5_base(load_in_8bit)
    elif llm == LLM_FLAN_T5_LARGE:
        return PdfQA.create_flan_t5_large(load_in_8bit)
    elif llm == LLM_FASTCHAT_T5_XL:
        return PdfQA.create_fastchat_t5_xl(load_in_8bit)
    elif llm == LLM_FALCON_SMALL:
        return PdfQA.create_falcon_instruct_small(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")

## To cache resource across multiple session
@st.cache_resource
def load_emb(emb):
    if emb == EMB_INSTRUCTOR_XL:
        return PdfQA.create_instructor_xl()
    elif emb == EMB_SBERT_MPNET_BASE:
        return PdfQA.create_sbert_mpnet()
    elif emb == EMB_SBERT_MINILM:
        pass ##ChromaDB takes care
    else:
        raise ValueError("Invalid embedding setting")

with st.sidebar:
    emb = st.radio("**Select Embedding Model**", [EMB_INSTRUCTOR_XL, EMB_SBERT_MPNET_BASE,EMB_SBERT_MINILM],index=1)
    llm = st.radio("**Select LLM Model**", [LLM_FASTCHAT_T5_XL, LLM_FLAN_T5_SMALL,LLM_FLAN_T5_BASE,LLM_FLAN_T5_LARGE,LLM_FLAN_T5_XL,LLM_FALCON_SMALL],index=2)
    load_in_8bit = st.radio("**Load 8 bit**", [True, False],index=1)
    pdf_file = st.file_uploader("**Upload PDF**", type="pdf")

    
    if st.button("Submit") and pdf_file is not None:
        with st.spinner(text="Uploading PDF and Generating Embeddings.."):
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                shutil.copyfileobj(pdf_file, tmp)
                tmp_path = Path(tmp.name)
                st.session_state["pdf_qa_model"].config = {
                    "pdf_path": str(tmp_path),
                    "embedding": emb,
                    "llm": llm,
                    "load_in_8bit": load_in_8bit
                }
                st.session_state["pdf_qa_model"].embedding = load_emb(emb)
                st.session_state["pdf_qa_model"].llm = load_llm(llm,load_in_8bit)        
                st.session_state["pdf_qa_model"].init_embeddings()
                st.session_state["pdf_qa_model"].init_models()
                st.session_state["pdf_qa_model"].vector_db_pdf()
                st.sidebar.success("PDF uploaded successfully")

question = st.text_input('Ask a question', 'What is this document?')

if st.button("Answer"):
    try:
        st.session_state["pdf_qa_model"].retreival_qa_chain()
        answer = st.session_state["pdf_qa_model"].answer_query(question)
        st.write(f"{answer}")
    except Exception as e:
        st.error(f"Error answering the question: {str(e)}")
