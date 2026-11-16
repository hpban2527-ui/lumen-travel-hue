"""Lumen Travel — a Huế guide that only answers from local, cited evidence."""
from pathlib import Path
import html
import os

import streamlit as st
from dotenv import load_dotenv

from firebase_client import AuthError, clear_user_state, ensure_fresh_session, login, register
from guide_loader import GuideError, guide_digest, load_guide
from rag import RAGError, answer_question, clean_base_url, embed_texts

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip()

st.set_page_config(page_title="Lumen Travel | Huế", page_icon=":material/travel_explore:", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{(HERE / 'styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the Huế guide…")
def cached_vectors(_entries, digest, server_url, model="embeddinggemma"):
    # digest, server and model invalidate the cache when data or configuration changes
    contents = [f"{e['title']} | {e['category']} | {' '.join(e['tags'])} | {e['text']}" for e in _entries]
    return embed_texts(server_url, contents)


def authenticated_session():
    user = st.session_state.get("user")
    if user is None:
        return None
    try:
        st.session_state.user = ensure_fresh_session(FIREBASE_API_KEY, user)
        return st.session_state.user
    except AuthError:
        clear_user_state(st.session_state)
        st.warning("Your session expired. Please sign in again.")
        return None


def brand():
    st.markdown('<div class="lm-brand">◌ &nbsp; lumen<span style="color:#A6ABB0">travel</span></div>',
                unsafe_allow_html=True)


def auth_page():
    left, right = st.columns([1.15, .85], gap="large", vertical_alignment="center")
    with left:
        brand()
        st.write("")
        st.write("")
        st.markdown('<span class="lm-badge">HUẾ · VIETNAM &nbsp; / &nbsp; CITY GUIDE</span>', unsafe_allow_html=True)
        st.markdown('<div class="lm-title">Discover the city.<br>Know the source.</div>', unsafe_allow_html=True)
        st.markdown('<p class="lm-muted">A quieter way to explore Huế. Ask about its imperial sites, riverside landmarks and local culture. Every answer stays with our curated guide.</p>', unsafe_allow_html=True)
        st.markdown('<div class="lm-rule"></div>', unsafe_allow_html=True)
        a,b,c=st.columns(3)
        for col,number,label in [(a,"16","Guide entries"),(b,"01","City"),(c,"100%","Source-led")]:
            with col:
                st.markdown(f'<div style="font:700 1.6rem Manrope,sans-serif">{number}</div><div class="lm-muted" style="font-size:.78rem">{label}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="lm-card"><div class="lm-kicker">YOUR JOURNEY STARTS HERE</div><h2 style="margin:10px 0 0;font-size:1.75rem">Welcome to Lumen</h2><p class="lm-muted" style="font-size:.87rem">Sign in to begin exploring the guide.</p></div>',unsafe_allow_html=True)
        sign_in, create = st.tabs(["Sign in", "Create account"])
        with sign_in:
            with st.form("sign_in", clear_on_submit=False):
                email = st.text_input("Email address", placeholder="you@example.com", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                clicked = st.form_submit_button("Sign in  →", type="primary", use_container_width=True)
            if clicked:
                try:
                    st.session_state.user = login(FIREBASE_API_KEY, email.strip(), password)
                    st.session_state.messages = []
                    st.rerun()
                except AuthError as exc:
                    st.error(str(exc))
        with create:
            with st.form("sign_up", clear_on_submit=False):
                email2 = st.text_input("Email address", placeholder="you@example.com", key="signup_email")
                password2 = st.text_input("Password · at least 6 characters", type="password", key="signup_password")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                create_clicked = st.form_submit_button("Create account  →", type="primary", use_container_width=True)
            if create_clicked:
                if password2 != confirm:
                    st.error("Passwords do not match.")
                else:
                    try:
                        st.session_state.user = register(FIREBASE_API_KEY, email2.strip(), password2)
                        st.session_state.messages = []
                        st.rerun()
                    except AuthError as exc:
                        st.error(str(exc))
        if not FIREBASE_API_KEY:
            st.info("Firebase setup needed: configure FIREBASE_API_KEY in your local .env file.")
    st.markdown('<div class="lm-footer">COMPUTATIONAL THINKING · PRACTICE 03 &nbsp; / &nbsp; HUỲNH PHƯƠNG BÌNH AN · 25127009</div>',unsafe_allow_html=True)


def draw_match_details(matches):
    with st.expander(f"Evidence used · {len(matches)} retrieved entries", expanded=False):
        st.caption("Cosine similarity describes vector closeness, not the probability that an answer is correct.")
        for hit in matches:
            e=hit.entry
            st.markdown(f"**[{e['id']}] {e['title']}** &nbsp; · &nbsp; similarity `{hit.score:.3f}`")
            st.caption(e["text"])
            st.markdown(f"[Original source]({e['source']})")
            st.divider()


def chat_page(user):
    try:
        guide = load_guide()
    except GuideError as exc:
        st.error(f"Could not load the guide: {exc}")
        return

    with st.sidebar:
        brand()
        st.caption("YOUR LOCAL GUIDE  /  2026")
        st.markdown('<div class="lm-rule"></div>', unsafe_allow_html=True)
        st.markdown('**The Huế collection**')
        st.caption(f"{len(guide)} hand-curated guide entries")
        categories = sorted({e["category"] for e in guide})
        category = st.selectbox("Browse by theme", ["All topics", *categories])
        shown = guide if category == "All topics" else [x for x in guide if x["category"]==category]
        for e in shown:
            st.markdown(f'<div style="font-size:.87rem;padding:7px 0;border-bottom:1px solid #E9EAEC"><span style="color:#989DA2">{e["id"]}</span> &nbsp; {html.escape(e["title"])}</div>',unsafe_allow_html=True)
        st.markdown('<div class="lm-rule"></div>', unsafe_allow_html=True)
        st.caption("Answers come from the saved guide, not real-time travel data.")
        st.caption(f"Signed in as {user.email}")
        if st.button("Sign out", use_container_width=True):
            clear_user_state(st.session_state)
            st.rerun()

    st.markdown('<div class="lm-kicker">LUMEN TRAVEL &nbsp; / &nbsp; HUẾ CITY GUIDE</div>',unsafe_allow_html=True)
    st.title("Ask a better question.")
    st.markdown('<p class="lm-muted">Explore Huế through 16 short, sourced notes. If we can’t find the answer in the guide, we’ll say so.</p>',unsafe_allow_html=True)
    st.write("")
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    if not st.session_state.messages:
        st.markdown("#### A few places to begin")
        cols = st.columns(3)
        suggestions = ["What is the Imperial City in Huế?", "Which river flows through Huế?", "Tell me about Dong Ba Market."]
        for col, prompt in zip(cols, suggestions):
            with col:
                st.button(prompt, key=f"suggest_{prompt}", use_container_width=True,
                          on_click=lambda text=prompt: st.session_state.update(pending_question=text))
        st.write("")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=":material/travel_explore:" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])
            if msg["role"]=="assistant" and msg.get("matches"):
                draw_match_details(msg["matches"])
    message = st.chat_input("Ask about Huế...")
    if st.session_state.get("pending_question"):
        message = st.session_state.pop("pending_question")
    if message is not None:
        question = message.strip()
        if not question:
            st.warning("Type a question before sending it.")
            return
        if len(question)>600:
            st.warning("Please keep questions under 600 characters.")
            return
        st.session_state.messages.append({"role":"user", "content":question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=":material/travel_explore:"):
            if not OLLAMA_BASE_URL:
                st.error("Ollama server not configured. Set OLLAMA_BASE_URL in .env.")
                return
            with st.spinner("Searching the Huế guide…"):
                try:
                    url=clean_base_url(OLLAMA_BASE_URL)
                    vectors=cached_vectors(guide,guide_digest(),url)
                    result=answer_question(url, question, guide, vectors, k=3)
                except RAGError as exc:
                    st.error(str(exc))
                    return
                st.markdown(result.text)
                draw_match_details(result.matches)
                st.session_state.messages.append({"role":"assistant", "content":result.text,"matches":result.matches})


if __name__ == "__main__":
    user = authenticated_session()
    if user is None:
        auth_page()
    else:
        chat_page(user)
