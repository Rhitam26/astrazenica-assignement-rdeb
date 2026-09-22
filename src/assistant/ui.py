"""Streamlit HTTP client. Never imports the backend graph or provider."""

import os

import httpx
import streamlit as st


def page_label(citation):
    pages = sorted(set(citation.get("page_numbers", [])))
    if not pages:
        return "page unavailable"
    if len(pages) == 1:
        return f"p. {pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pp. {pages[0]}–{pages[-1]}"
    return "pp. " + ", ".join(map(str, pages))


def show_assistant(response):
    if response.get("workflow"):
        label = f"Workflow: {response['workflow']}"
        if response.get("latency_ms") is not None:
            label += f" · {response['latency_ms']} ms"
        st.caption(label)
    st.markdown(response["answer"])
    if response.get("legacy"):
        st.caption("Source details unavailable for this older response.")
    for citation in response.get("citations", []):
        with st.expander(f"[{citation['marker']}] {citation['doc_title']} — {page_label(citation)}"):
            st.write(" › ".join(citation["heading_path"]) or "No section metadata")
            st.caption(f"{citation['content_type']} · similarity {citation['score']:.3f}")
            if citation.get("table_id"):
                st.text(f"Table: {citation['table_id']}")
            if citation.get("source_path"):
                st.text(citation["source_path"])
            if citation.get("text_preview"):
                st.caption("Retrieved text")
                st.text(citation["text_preview"])


def api_get(path, **params):
    response = httpx.get(
        os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/") + path,
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def load_conversation(cid):
    try:
        data = api_get(f"/v1/conversations/{cid}")
        history = []
        for message in data["messages"]:
            if message["role"] == "user":
                history.append({"role": "user", "content": message["content"]})
            else:
                metadata = message.get("metadata")
                history.append(
                    {
                        "role": "assistant",
                        "response": {
                            **(metadata or {}),
                            "answer": message["content"],
                            "legacy": metadata is None,
                        },
                    }
                )
        st.session_state.history = history
        st.session_state.conversation_id = data["conversation_id"]
        st.query_params["conversation_id"] = data["conversation_id"]
    except (httpx.HTTPError, ValueError, KeyError):
        st.sidebar.error("Could not load conversation. Select it again or refresh to retry.")


def conversation_sidebar():
    st.sidebar.subheader("Conversations")
    refresh = st.sidebar.button("Refresh")
    if refresh:
        st.session_state.conversations_dirty = True
        requested = st.query_params.get("conversation_id")
        if requested and requested != st.session_state.conversation_id:
            load_conversation(requested)
    if st.session_state.get("conversations_dirty", True):
        try:
            page = api_get("/v1/conversations")
            st.session_state.conversation_list = page["items"]
            st.session_state.next_offset = page["next_offset"]
            st.session_state.conversations_dirty = False
        except (httpx.HTTPError, ValueError, KeyError):
            st.sidebar.warning("History is unavailable. Use Refresh to retry.")
    for item in st.session_state.get("conversation_list", []):
        if st.sidebar.button(
            item["title"], key=f"conversation_{item['conversation_id']}", help=item["updated_at"]
        ):
            load_conversation(item["conversation_id"])
    if st.session_state.get("next_offset") is not None and st.sidebar.button("Load more"):
        try:
            page = api_get("/v1/conversations", offset=st.session_state.next_offset)
            known = {item["conversation_id"] for item in st.session_state.conversation_list}
            st.session_state.conversation_list.extend(
                item for item in page["items"] if item["conversation_id"] not in known
            )
            st.session_state.next_offset = page["next_offset"]
            st.rerun()
        except (httpx.HTTPError, ValueError, KeyError):
            st.sidebar.error("Could not load more conversations. Try again.")


def main():
    st.set_page_config(page_title="DocuSense", page_icon="📚")
    st.title("DocuSense")
    st.caption("Answers grounded in your supplied PDFs")
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.conversation_id = None
        requested = st.query_params.get("conversation_id")
        if requested:
            load_conversation(requested)
    if st.sidebar.button("New conversation"):
        st.session_state.history = []
        st.session_state.conversation_id = None
        st.query_params.pop("conversation_id", None)
        st.rerun()
    conversation_sidebar()
    if st.session_state.conversation_id:
        st.sidebar.caption(f"Conversation: {st.session_state.conversation_id}")
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.markdown(turn["content"])
            else:
                show_assistant(turn["response"])
    if message := st.chat_input("Ask about the documents"):
        with st.chat_message("user"):
            st.markdown(message)
        try:
            with st.spinner("Searching the knowledge base…"):
                result = httpx.post(
                    os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/") + "/v1/chat",
                    json={"message": message, "conversation_id": st.session_state.conversation_id},
                    timeout=900,
                )
                result.raise_for_status()
                response = result.json()
            st.session_state.conversation_id = response["conversation_id"]
            st.query_params["conversation_id"] = response["conversation_id"]
            st.session_state.history.extend(
                [{"role": "user", "content": message}, {"role": "assistant", "response": response}]
            )
            st.session_state.conversations_dirty = True
            st.rerun()
        except (httpx.HTTPError, ValueError, KeyError):
            st.error("The assistant could not complete this request. Check the API service and try again.")


if __name__ == "__main__":
    main()
