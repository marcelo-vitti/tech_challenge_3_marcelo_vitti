import streamlit as st
import xgboost

from utils.handle_user_answer import generate_user_answer


model = xgboost.XGBClassifier()
model.load_model("model/models/xgboost_model_v01.json")


st.title("Diabetes Check Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_message = "Please send me the following data to check for diabetes: hba1c, glucose_postprandial, glucose_fasting, family_history_diabetes, age"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = generate_user_answer(prompt, model)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# 8.18, 236, 136, 0, 58 -> 1
# 5.63, 150, 93, 0, 48 -> 0
