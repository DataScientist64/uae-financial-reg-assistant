import streamlit as st
from src.ingest import get_or_create_vector_store
from src.retrieve import retrieve, format_context
from src.generate import generate

st.set_page_config(page_title="UAE Financial Regulatory Assistant", page_icon="🏦")
st.title("🏦 UAE Financial Regulatory Assistant")
st.caption("Grounded in real UAE Central Bank regulatory documents.")
st.warning("This tool is for informational purposes only. Not legal or financial advice.")


@st.cache_resource
def load_vector_store():
    return get_or_create_vector_store()


vector_store = load_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about UAE financial regulations...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching regulatory documents..."):
            docs = retrieve(question, vector_store)
            context = format_context(docs)
            answer = generate(question, context)
            st.markdown(answer)
            if docs:
                with st.expander("Sources"):
                    for doc in docs:
                        st.write(
                            f"Page {doc.metadata.get('page', '?')} "
                            f"— {doc.metadata.get('source', '?')}"
                        )

    st.session_state.messages.append({"role": "assistant", "content": answer})