import json
import os
import uuid

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv


# Load environment variables once
if "config_loaded" not in st.session_state:
    load_dotenv()
    st.session_state.URL = os.getenv("URL", "http://127.0.0.1:8014")

if "unique_id" not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4())


@st.dialog("ℹ️ Connection Error")
def no_internet(text):
    st.markdown(f"{text}")


def get_reponse(prompt: str, messages: list, url: str):
    payload = {"query": prompt.strip(), "last_3_turn": messages}
    try:
        response = requests.post(url=url, data=json.dumps(payload), timeout=30)
        if response.status_code == 200:
            return response.json().get("response")
    except requests.RequestException:
        no_internet(
            """The academic assistant is currently unable to reach its backend server.

            This can happen if the API service is off, the URL is wrong, or the network is unavailable.
            Please try again in a moment.
            """
        )
    return None


@st.dialog("ℹ️ Academic assistant usage policy")
def academic_policy():
    st.markdown(
        """
        **1. 🔒 Privacy**: Conversations are used only for improving the assistant's learning and debugging.
        **2. ⚠️ No Guarantees**: Responses may need checking against the course context.
        **3. 🪪 No Sensitive Info**: Avoid entering private or personal information.
        **4. 🎓 Academic Purpose**: This interface is intended for course, lecture, and research support.

        ✔️ By continuing, you agree to these terms.
        """
    )

    if st.button("AGREE AND CONTINUE", icon="✅"):
        st.session_state.accepted_policy = True
        st.rerun()


def check_internet():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except requests.RequestException:
        return False


def ChatUI():
    if "accepted_policy" not in st.session_state:
        academic_policy()
        st.stop()

    st.info("☑️ This is an academic instructor and lecturer assistant.")
    st.info("💬 I can explain course concepts, lecture notes, research papers, and study topics.")
    st.title("ContextForge")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "history" not in st.session_state:
        st.session_state.history = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]

    for message in st.session_state.messages:
        role = message["role"]
        avatar = ":material/person:" if role == "user" else "✨"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(prompt)

        placeholder = st.empty()
        with placeholder.chat_message("assistant", avatar=":material/lightbulb:"):
            st.markdown("Thinking...")

        response = get_reponse(prompt.strip(), st.session_state.history[-6:], st.session_state.URL + "/chat")

        if response:
            placeholder.empty()
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        placeholder.empty()

    st.markdown('<div id="bottom-anchor"></div>', unsafe_allow_html=True)
    components.html(
        """
        <script>
            var element = window.parent.document.getElementById("bottom-anchor");
            if (element) {
                element.scrollIntoView({behavior: "smooth"});
            }
        </script>
        """,
        height=0,
    )


if "server_ok" not in st.session_state:
    try:
        response = requests.get(url=st.session_state.URL.strip(), timeout=5)
        st.session_state.server_ok = response.status_code == 200
    except requests.RequestException:
        st.session_state.server_ok = False

if not st.session_state.URL:
    print("URL is empty in environment variable")
else:
    if not check_internet():
        no_internet(
            """**⚠️ No Internet Connection Detected**

            The academic assistant requires an active internet connection to fetch live responses and updates.
            """
        )
        st.stop()

    if st.session_state.server_ok:
        ChatUI()
    else:
        no_internet(
            """The academic assistant is currently unable to reach its server.

            This might be due to:
            - Temporary server maintenance
            - Network issues
            - Incorrect backend URL

            Please try again later or contact support.
            """
        )


if __name__ == "__main__":
    ChatUI()




    
