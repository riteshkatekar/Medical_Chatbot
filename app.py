###############################################################
########## with typing #################

# """
# app.py

# Streamlit frontend for stateless Medical Chatbot — conversation is placed inside a dark inner box.
# """

# import streamlit as st
# import time
# import random
# import sys
# import os
# import html as html_module
# from datetime import datetime
# from string import Template

# # --- ensure src importable ---
# ROOT = os.getcwd()
# if ROOT not in sys.path:
#     sys.path.append(ROOT)
# SRC_PATH = os.path.join(ROOT, "src")
# if SRC_PATH not in sys.path:
#     sys.path.append(SRC_PATH)

# # --- import backend inference function (unchanged logic) ---
# try:
#     from chatbot import get_chatbot_response
# except Exception:
#     try:
#         from src.chatbot import get_chatbot_response  # type: ignore
#     except Exception:
#         get_chatbot_response = None

# # --- config ---
# MODELS_DIR = "models"
# TYPING_BASE_SECONDS = 0.45
# MAX_CHARS = 600

# THEME = {
#     "bg_left": "#0f1720",
#     "bg_right": "#071027",
#     "card_bg": "transparent",  # outer card uses transparent background so inner box is the visible black box
#     "card_shadow": "0 20px 60px rgba(0,0,0,0.55)",
#     "msg_box_bg": "rgba(2,6,10,0.86)",  # dark inner box color
#     "bot_bubble": "#57B7FF",
#     "bot_text": "#062135",
#     "user_bubble": "#34D399",
#     "user_text": "#042B16",
# }



# CSS = Template(
#     r"""
# <style>
# :root{
#   --bg-left: $bg_left;
#   --bg-right: $bg_right;
#   --card-bg: $card_bg;
#   --card-shadow: $card_shadow;
#   --bot-bubble: $bot_bubble;
#   --bot-text: $bot_text;
#   --user-bubble: $user_bubble;
#   --user-text: $user_text;
# }

# /* Page */
# html, body, [data-testid="stAppViewContainer"] {
#   min-height: 100vh;
#   margin: 0; padding: 0;
#   background: linear-gradient(90deg, var(--bg-left) 0%, var(--bg-right) 100%);
#   font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
# }

# /* Center wrapper */
# .chat {
#   max-width: 880px;
#   margin: 20px auto;
#   padding: 0 12px;
# }

# /* Card */
# .card {
#   background: var(--card-bg);
#   box-shadow: var(--card-shadow);
#   border-radius: 14px;
#   border: none;
#   padding: 12px;
#   display: flex;
#   flex-direction: column;
#   /* Let card size to content but cap max height to avoid one big black rectangle */
#   height: auto;
#   max-height: calc(100vh - 72px);
#   overflow: hidden;
# }

# /* Header */
# .msg_head { display:flex; gap:12px; align-items:center; padding:6px 4px; margin-bottom:8px; }
# .img_cont img { width:46px; height:46px; border-radius:50%; box-shadow:0 6px 18px rgba(0,0,0,0.35); }
# .user_info span { font-weight:700; color:#f6fbff; font-size:18px; display:block; }
# .user_info p { margin:0; font-size:12px; color:#bcd6ea; }

# /* Scrollable message area - transparent to remove big inner rectangle */
# .msg_card_body {
#   flex: 1 1 auto;
#   overflow-y: auto;
#   padding: 18px;
#   border-radius: 10px;
#   background: transparent;      /* make transparent to avoid large dark box */
#   display: flex;
#   flex-direction: column;
#   gap: 18px;                    /* increased gap to avoid overlap/collision */
#   margin-bottom: 6px;
#   max-height: 60vh;             /* cap the visible area; allows card to shrink when few messages */
# }

# /* Bot bubble (left) */
# .msg_cotainer {
#   background: var(--bot-bubble);
#   color: var(--bot-text);
#   padding: 14px 16px;
#   border-radius: 12px;
#   max-width: 78%;
#   word-wrap: break-word;
#   line-height: 1.45;
#   font-size: 15px;
#   box-shadow: 0 6px 18px rgba(6,33,53,0.12);
#   white-space: pre-wrap;
# }

# /* User bubble (right) */
# .msg_cotainer_send {
#   background: var(--user-bubble);
#   color: var(--user-text);
#   padding: 12px 14px;
#   border-radius: 12px;
#   max-width: 46%;
#   min-width: 90px;
#   word-wrap: break-word;
#   line-height: 1.4;
#   font-size: 15px;
#   box-shadow: 0 6px 12px rgba(4,43,22,0.08);
#   white-space: pre-wrap;
#   text-align: left;
# }

# /* Avatars next to messages */
# .img_cont_msg img { width:36px; height:36px; border-radius:50%; }

# /* Time metadata */
# .msg_time, .msg_time_send { display:block; font-size:11px; color:#9fb6d0; margin-top:6px; }

# /* Footer / Input (anchored inside card) */
# .card-footer { flex: 0 0 auto; padding-top:6px; border-top: 1px solid rgba(255,255,255,0.02); margin-top:8px; }
# .input-group { display:flex; gap:8px; align-items:center; }
# .type_msg {
#   flex: 1;
#   background: transparent;
#   border: 1px solid rgba(255,255,255,0.06);
#   border-radius: 10px;
#   padding: 10px 12px;
#   color: #e6eef8;
#   font-size: 15px;
#   outline: none;
# }
# .type_msg::placeholder { color: rgba(255,255,255,0.45); }

# /* Quick suggestion chips */
# .quick-chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
# .chip {
#   background: rgba(255,255,255,0.03);
#   color: #e6eef8;
#   padding: 6px 10px;
#   border-radius: 999px;
#   font-size: 13px;
#   cursor: pointer;
#   border: 1px solid rgba(255,255,255,0.03);
# }
# .chip:hover { background: rgba(255,255,255,0.06); transform: translateY(-1px); }

# /* Style the Streamlit submit button inside card footer to match send icon */
# .card-footer .stButton button {
#   background: linear-gradient(90deg,#06b6d4,#3b82f6) !important;
#   color: white !important;
#   border: none !important;
#   padding: 10px 12px !important;
#   border-radius: 10px !important;
#   display:flex !important;
#   align-items:center !important;
#   justify-content:center !important;
# }

# /* Responsive tweaks */
# @media (max-width:900px){
#   .card { padding:10px; }
#   .msg_cotainer { max-width: 84%; }
#   .msg_cotainer_send { max-width: 72%; min-width: 64px; }
# }
# </style>
# """
# )

# st.set_page_config(page_title="Medical Chatbot", layout="centered", page_icon="🩺")
# st.markdown(CSS.safe_substitute(THEME), unsafe_allow_html=True)

# # --- session state ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []  # list of dict: sender, text, time
# if "bot_details" not in st.session_state:
#     st.session_state.bot_details = []

# # --- layout ---
# st.markdown('<div class="chat">', unsafe_allow_html=True)
# st.markdown('<div class="card">', unsafe_allow_html=True)

# # header (no stray rectangle)
# st.markdown(
#     """
#     <div class="msg_head">
#       <div class="img_cont"><img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" /></div>
#       <div class="user_info"><span>Medical Chatbot</span><p>Ask me anything!</p></div>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# # message area (the dark inner box)
# st.markdown('<div id="messageFormeight" class="msg_card_body">', unsafe_allow_html=True)

# def _render_messages(history):
#     blocks = []
#     for m in history:
#         sender = m.get("sender")
#         # escape & preserve line breaks; ensure words don't break into characters
#         text = html_module.escape(m.get("text", "")).replace("\n", "<br/>")
#         ts = html_module.escape(m.get("time", ""))
#         if sender == "user":
#             block = f"""
#             <div style="display:flex;justify-content:flex-end;gap:12px;align-items:flex-end;">
#               <div class="msg_cotainer_send">{text}<span class="msg_time_send">{ts}</span></div>
#               <div class="img_cont_msg"><img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" /></div>
#             </div>
#             """
#         else:
#             block = f"""
#             <div style="display:flex;justify-content:flex-start;gap:12px;align-items:flex-start;">
#               <div class="img_cont_msg"><img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" /></div>
#               <div class="msg_cotainer">{text}<span class="msg_time">{ts}</span></div>
#             </div>
#             """
#         blocks.append(block)
#     if blocks:
#         st.markdown("\n".join(blocks), unsafe_allow_html=True)

# _render_messages(st.session_state.chat_history)
# st.markdown("</div>", unsafe_allow_html=True)  # close messages (dark box)

# # --- footer / input --- (use clear_on_submit=True so input clears automatically after submission)
# st.markdown('<div class="card-footer">', unsafe_allow_html=True)
# with st.form("chat_form", clear_on_submit=True):
#     cols = st.columns([1, 12, 2])
#     user_text = cols[1].text_input("", key="user_input", placeholder="Type your message...", max_chars=MAX_CHARS, label_visibility="collapsed")
#     send_pressed = cols[2].form_submit_button("Send")
# st.markdown("</div>", unsafe_allow_html=True)

# # JS helpers: autofocus & scroll-to-bottom
# st.markdown(
#     """
#     <script>
#     (function(){
#       function scrollToBottom(){ var c=document.getElementById('messageFormeight'); if(c){ c.scrollTop = c.scrollHeight; } }
#       function focusInput(){ var i=document.querySelector('input[placeholder="Type your message..."]'); if(i){ i.focus(); } }
#       setTimeout(scrollToBottom,200); setTimeout(focusInput,300); setTimeout(scrollToBottom,700);
#     })();
#     </script>
#     """,
#     unsafe_allow_html=True,
# )

# # --- send handling (backend call preserved) ---
# if send_pressed and user_text and user_text.strip():
#     msg = user_text.strip()
#     now = datetime.now().strftime("%H:%M")

#     # append user message (will show immediately)
#     st.session_state.chat_history.append({"sender": "user", "text": msg, "time": now})

#     # show typing placeholder
#     # placeholder = st.empty()
#     # with placeholder.container():
#     #     st.markdown(
#     #         """
#     #         <div style="display:flex;justify-content:flex-start;gap:12px;align-items:flex-start;">
#     #           <div class="img_cont_msg"><img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" /></div>
#     #           <div class="msg_cotainer"><span style="display:inline-block;width:48px;"><span style="display:inline-block;width:8px;height:8px;margin-right:6px;background:#fff;border-radius:50%;opacity:0.3;animation:typing 1s infinite;"></span><span style="display:inline-block;width:8px;height:8px;margin-right:6px;background:#fff;border-radius:50%;opacity:0.3;animation:typing 1s infinite .2s;"></span><span style="display:inline-block;width:8px;height:8px;margin-right:6px;background:#fff;border-radius:50%;opacity:0.3;animation:typing 1s infinite .4s;"></span></span></div>
#     #         </div>
#     #         <style>@keyframes typing{0%{transform:translateY(0);opacity:0.2}50%{transform:translateY(-6px);opacity:1}100%{transform:translateY(0);opacity:0.2}}</style>
#     #         """,
#     #         unsafe_allow_html=True,
#     #     )

#     placeholder = st.empty()
#     with placeholder.container():
#         st.markdown(
#             """
#         <div style="display:flex;justify-content:flex-start;gap:12px;align-items:flex-start;">
#           <div class="img_cont_msg">
#             <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" />
#           </div>
#           <div class="msg_cotainer">
#             <span style="display:inline-block;width:48px;">
#               <span style="display:inline-block;width:8px;height:8px;margin-right:6px;background:#fff;border-radius:50%;opacity:0.3;animation:typing 1s infinite;"></span>
#               <span style="display:inline-block;width:8px;height:8px;margin-right:6px;background:#fff;border-radius:50%;opacity:0.3;animation:typing 1s infinite .2s;"></span>
#               <span style="display:inline-block;width:8px;height:8px;margin-right:6px;background:#fff;border-radius:50%;opacity:0.3;animation:typing 1s infinite .4s;"></span>
#             </span>
#           </div>
#         </div>

#         <style>
#         @keyframes typing {
#           0%   { transform: translateY(0);   opacity: 0.2; }
#           50%  { transform: translateY(-6px); opacity: 1; }
#           100% { transform: translateY(0);   opacity: 0.2; }
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )


#     # simulate a small delay
#     time.sleep(TYPING_BASE_SECONDS + random.random() * 0.45)

#     # call backend (unchanged)
#     if get_chatbot_response is None:
#         bot_text = "Backend not available for import. Ensure 'src/chatbot.py' is present and exports get_chatbot_response()."
#         result = {"reply": bot_text, "intent": "unknown", "confidence": 0.0, "description": "", "precaution": "", "emergency": False}
#     else:
#         try:
#             result = get_chatbot_response(msg, models_dir=MODELS_DIR, threshold=0.5)
#         except Exception as e:
#             result = {"reply": f"Error while processing: {e}", "intent": "error", "confidence": 0.0, "description": "", "precaution": "", "emergency": False}

#     placeholder.empty()

#     # emergency banner inside card if flagged
#     if result.get("emergency"):
#         st.markdown(
#             "<div style='background:#ff4d4f;color:white;padding:10px;border-radius:8px;font-weight:700;margin:6px 0;'>⚠️ EMERGENCY: This sounds like a medical emergency. Call local emergency services immediately.</div>",
#             unsafe_allow_html=True,
#         )

#     # use backend reply as-is
#     reply = result.get("reply", "") or ""

#     # append bot reply
#     st.session_state.chat_history.append({"sender": "bot", "text": reply, "time": datetime.now().strftime("%H:%M")})
#     st.session_state.bot_details.append({
#         "Intent": result.get("intent", ""),
#         "Confidence": f"{float(result.get('confidence',0.0))*100:.1f}%",
#         "Description": result.get("description", "") or "",
#         "Precaution": result.get("precaution", "") or ""
#     })

#     # scroll to bottom after rerun
#     st.markdown(
#         """
#         <script>
#         (function(){ var c=document.getElementById('messageFormeight'); if(c){ c.scrollTop = c.scrollHeight; } })();
#         </script>
#         """,
#         unsafe_allow_html=True,
#     )

#     # rerun to show updated messages; input is cleared automatically by clear_on_submit=True
#     if hasattr(st, "experimental_rerun"):
#         try:
#             st.experimental_rerun()
#         except Exception:
#             st.rerun()
#     else:
#         st.rerun()

# # close wrappers
# st.markdown("</div></div>", unsafe_allow_html=True)






























############################################################################################

### without typing (best )

"""
app.py

Streamlit frontend for stateless Medical Chatbot — conversation is placed inside a dark inner box.
"""

import streamlit as st
import sys
import os
import html as html_module
from datetime import datetime
from string import Template

# --- ensure src importable ---
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# --- import backend inference function (unchanged logic) ---
try:
    from chatbot import get_chatbot_response
except Exception:
    try:
        from src.chatbot import get_chatbot_response  # type: ignore
    except Exception:
        get_chatbot_response = None

# --- config ---
MODELS_DIR = "models"
MAX_CHARS = 600

THEME = {
    "bg_left": "#0f1720",
    "bg_right": "#071027",
    "card_bg": "transparent",  # outer card uses transparent background so inner box is the visible black box
    "card_shadow": "0 20px 60px rgba(0,0,0,0.55)",
    "msg_box_bg": "rgba(2,6,10,0.86)",  # dark inner box color (unused in CSS here but kept if you reuse)
    "bot_bubble": "#57B7FF",
    "bot_text": "#062135",
    "user_bubble": "#34D399",
    "user_text": "#042B16",
}

CSS = Template(
    r"""
<style>
:root{
  --bg-left: $bg_left;
  --bg-right: $bg_right;
  --card-bg: $card_bg;
  --card-shadow: $card_shadow;
  --bot-bubble: $bot_bubble;
  --bot-text: $bot_text;
  --user-bubble: $user_bubble;
  --user-text: $user_text;
}

/* Page */
html, body, [data-testid="stAppViewContainer"] {
  min-height: 100vh;
  margin: 0; padding: 0;
  background: linear-gradient(90deg, var(--bg-left) 0%, var(--bg-right) 100%);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}

/* Center wrapper */
.chat {
  max-width: 880px;
  margin: 20px auto;
  padding: 0 12px;
}

/* Card */
.card {
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  border-radius: 14px;
  border: none;
  padding: 12px;
  display: flex;
  flex-direction: column;
  /* Let card size to content but cap max height to avoid one big black rectangle */
  height: auto;
  max-height: calc(100vh - 72px);
  overflow: hidden;
}

/* Header */
.msg_head { display:flex; gap:12px; align-items:center; padding:6px 4px; margin-bottom:8px; }
.img_cont img { width:46px; height:46px; border-radius:50%; box-shadow:0 6px 18px rgba(0,0,0,0.35); }
.user_info span { font-weight:700; color:#f6fbff; font-size:18px; display:block; }
.user_info p { margin:0; font-size:12px; color:#bcd6ea; }

/* Scrollable message area - transparent to remove big inner rectangle */
/* Message row spacing — ensures bubbles keep distance */
.msg_card_body {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 18px;
  border-radius: 10px;
  background: transparent;
  display: flex;
  flex-direction: column;
  gap: 22px;                    /* slightly larger gap between message rows */
  margin-bottom: 6px;
  max-height: 60vh;
}

/* each message is a row; .bot/.user helps alignment */
.message-row { margin: 6px 0; display:flex; gap:12px; align-items:flex-start; }
.message-row.bot { justify-content:flex-start; align-items:flex-start; }
.message-row.user { justify-content:flex-end; align-items:flex-end; }

/* add vertical margin inside each bubble so their rounded corners don't touch */
.msg_cotainer, .msg_cotainer_send {
  margin-block: 6px;   /* vertical spacing inside each bubble */
}

/* keep time stamps spaced from bubble edge */
.msg_time, .msg_time_send { display:block; font-size:11px; color:#9fb6d0; margin-top:8px; }


/* Bot bubble (left) */
.msg_cotainer {
  background: var(--bot-bubble);
  color: var(--bot-text);
  padding: 14px 16px;
  border-radius: 12px;
  max-width: 78%;
  word-wrap: break-word;
  line-height: 1.45;
  font-size: 15px;
  box-shadow: 0 6px 18px rgba(6,33,53,0.12);
  white-space: pre-wrap;
}

/* User bubble (right) */
.msg_cotainer_send {
  background: var(--user-bubble);
  color: var(--user-text);
  padding: 12px 14px;
  border-radius: 12px;
  max-width: 46%;
  min-width: 90px;
  word-wrap: break-word;
  line-height: 1.4;
  font-size: 15px;
  box-shadow: 0 6px 12px rgba(4,43,22,0.08);
  white-space: pre-wrap;
  text-align: left;
}

/* Avatars next to messages */
.img_cont_msg img { width:36px; height:36px; border-radius:50%; }

/* Time metadata */
.msg_time, .msg_time_send { display:block; font-size:11px; color:#9fb6d0; margin-top:6px; }

/* Footer / Input (anchored inside card) */
.card-footer { flex: 0 0 auto; padding-top:6px; border-top: 1px solid rgba(255,255,255,0.02); margin-top:8px; }
.input-group { display:flex; gap:8px; align-items:center; }
.type_msg {
  flex: 1;
  background: transparent;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 10px 12px;
  color: #e6eef8;
  font-size: 15px;
  outline: none;
}
.type_msg::placeholder { color: rgba(255,255,255,0.45); }

/* Quick suggestion chips */
.quick-chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
.chip {
  background: rgba(255,255,255,0.03);
  color: #e6eef8;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 13px;
  cursor: pointer;
  border: 1px solid rgba(255,255,255,0.03);
}
.chip:hover { background: rgba(255,255,255,0.06); transform: translateY(-1px); }

/* Style the Streamlit submit button inside card footer to match send icon */
.card-footer .stButton button {
  background: linear-gradient(90deg,#06b6d4,#3b82f6) !important;
  color: white !important;
  border: none !important;
  padding: 10px 12px !important;
  border-radius: 10px !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
}

/* Responsive tweaks */
@media (max-width:900px){
  .card { padding:10px; }
  .msg_cotainer { max-width: 84%; }
  .msg_cotainer_send { max-width: 72%; min-width: 64px; }
}
</style>
"""
)

st.set_page_config(page_title="Medical Chatbot", layout="centered", page_icon="🩺")
st.markdown(CSS.safe_substitute(THEME), unsafe_allow_html=True)

# --- session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dict: sender, text, time
if "bot_details" not in st.session_state:
    st.session_state.bot_details = []

# --- layout ---
st.markdown('<div class="chat">', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

# header (no stray rectangle)
st.markdown(
    """
    <div class="msg_head">
      <div class="img_cont"><img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" /></div>
      <div class="user_info"><span>Medical Chatbot</span><p>Ask me anything!</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# message area (the dark inner box)
st.markdown('<div id="messageFormeight" class="msg_card_body">', unsafe_allow_html=True)

def _render_messages(history):
    blocks = []
    for m in history:
        sender = m.get("sender")
        # escape & preserve line breaks; ensure words don't break into characters
        text = html_module.escape(m.get("text", "")).replace("\n", "<br/>")
        ts = html_module.escape(m.get("time", ""))
        if sender == "user":
            block = f"""
            <div style="display:flex;justify-content:flex-end;gap:12px;align-items:flex-end;">
              <div class="msg_cotainer_send">{text}<span class="msg_time_send">{ts}</span></div>
              <div class="img_cont_msg"><img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" /></div>
            </div>
            """
        else:
            block = f"""
            <div style="display:flex;justify-content:flex-start;gap:12px;align-items:flex-start;">
              <div class="img_cont_msg"><img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" /></div>
              <div class="msg_cotainer">{text}<span class="msg_time">{ts}</span></div>
            </div>
            """
        blocks.append(block)
    if blocks:
        st.markdown("\n".join(blocks), unsafe_allow_html=True)

_render_messages(st.session_state.chat_history)
st.markdown("</div>", unsafe_allow_html=True)  # close messages (dark box)

# --- footer / input --- (use clear_on_submit=True so input clears automatically after submission)
st.markdown('<div class="card-footer">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([1, 12, 2])
    user_text = cols[1].text_input("", key="user_input", placeholder="Type your message...", max_chars=MAX_CHARS, label_visibility="collapsed")
    send_pressed = cols[2].form_submit_button("Send")
st.markdown("</div>", unsafe_allow_html=True)

# JS helpers: autofocus & scroll-to-bottom
st.markdown(
    """
    <script>
    (function(){
      function scrollToBottom(){ var c=document.getElementById('messageFormeight'); if(c){ c.scrollTop = c.scrollHeight; } }
      function focusInput(){ var i=document.querySelector('input[placeholder="Type your message..."]'); if(i){ i.focus(); } }
      setTimeout(scrollToBottom,200); setTimeout(focusInput,300); setTimeout(scrollToBottom,700);
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# --- send handling (backend call preserved; typing removed) ---
if send_pressed and user_text and user_text.strip():
    msg = user_text.strip()
    now = datetime.now().strftime("%H:%M")

    # append user message (will show immediately)
    st.session_state.chat_history.append({"sender": "user", "text": msg, "time": now})

    # Immediately call backend (no typing animation, no artificial delay)
    if get_chatbot_response is None:
        bot_text = "Backend not available for import. Ensure 'src/chatbot.py' is present and exports get_chatbot_response()."
        result = {"reply": bot_text, "intent": "unknown", "confidence": 0.0, "description": "", "precaution": "", "emergency": False}
    else:
        try:
            result = get_chatbot_response(msg, models_dir=MODELS_DIR, threshold=0.5)
        except Exception as e:
            result = {"reply": f"Error while processing: {e}", "intent": "error", "confidence": 0.0, "description": "", "precaution": "", "emergency": False}

    # emergency banner inside card if flagged
    if result.get("emergency"):
        st.markdown(
            "<div style='background:#ff4d4f;color:white;padding:10px;border-radius:8px;font-weight:700;margin:6px 0;'>⚠️ EMERGENCY: This sounds like a medical emergency. Call local emergency services immediately.</div>",
            unsafe_allow_html=True,
        )

    # use backend reply as-is
    reply = result.get("reply", "") or ""

    # append bot reply
    st.session_state.chat_history.append({"sender": "bot", "text": reply, "time": datetime.now().strftime("%H:%M")})
    st.session_state.bot_details.append({
        "Intent": result.get("intent", ""),
        "Confidence": f"{float(result.get('confidence',0.0))*100:.1f}%",
        "Description": result.get("description", "") or "",
        "Precaution": result.get("precaution", "") or ""
    })

    # scroll to bottom after rerun
    st.markdown(
        """
        <script>
        (function(){ var c=document.getElementById('messageFormeight'); if(c){ c.scrollTop = c.scrollHeight; } })();
        </script>
        """,
        unsafe_allow_html=True,
    )

    # rerun to show updated messages; input is cleared automatically by clear_on_submit=True
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()
    else:
        st.rerun()

# close wrappers
st.markdown("</div></div>", unsafe_allow_html=True)
