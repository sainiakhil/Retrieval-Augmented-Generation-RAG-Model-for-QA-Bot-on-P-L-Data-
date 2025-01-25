import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
import tempfile


@st.cache_resource
def load_model():

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
    )

    # Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")


    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,  # Reduce token generation
    )

    return llm, embed_model

@st.cache_resource
def create_vector_store(_docs, _settings_obj):
  vector_index = VectorStoreIndex.from_documents(
      _docs,
      embed_model=_settings_obj.embed_model,
      node_parser=_settings_obj.node_parser,
      show_progress=True
      )
  return vector_index

def main():
    st.title("Financial Document Q&A Assistant")

    # Sidebar for document upload
    st.sidebar.header("Upload Financial Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    # Load model and embedding
    llm, embed_model = load_model()

    # Configure Settings
    settings = Settings
    settings.llm = llm
    settings.embed_model = embed_model
    settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    settings.num_output = 50
    settings.context_window = 2048
    settings.genrate_kwargs = {"do_sample": False,"temperature": 0.1, "max_new_tokens": 128}

    # Process uploaded documents
    if uploaded_files:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            doc_paths = []
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                doc_paths.append(file_path)

            # Load documents
            reader = SimpleDirectoryReader(input_dir=temp_dir)
            documents = reader.load_data()

            # Create vector store and index
            index = create_vector_store(documents, settings)

            # Query Engine
            query_engine = index.as_query_engine(
                llm=settings.llm,
                similarity_top_k= 5,
                response_mode="compact",
                verbose=True,
                generate_kwargs=settings.genrate_kwargs,
                context_window=settings.context_window,
                num_output=settings.num_output,
                show_progress=True
              )

            # Query input
            query = st.text_input("Enter your financial query:")

            # Display results
            if query:
                with st.spinner("Analyzing documents..."):
                    response = query_engine.query(query)
                    cleaned_response = response.response.split('\n')[0].strip()

                st.header("Original Query:", divider="rainbow")
                st.header(query)
                st.subheader("Response:")
                st.write(cleaned_response)

                st.header("Retrieved Document Chunks:", divider = True)
                for i, node in enumerate(response.source_nodes, 1):
                    st.text(f"Chunk {i}:")
                    st.text(f"Relevance Score: {node.score:.4f}")
                    st.text(f"Text (first 300 chars): {node.text[:300]}...")
                    st.text("-" * 50)

                    

if __name__ == "__main__":
    main()