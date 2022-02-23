import streamlit as st

import about_page
import signature_page
from Utils import Utils_streamlit

st.markdown("<h1 style='text-align: center; color: green;'>SIGNATURE VERIFICATION</h1>", unsafe_allow_html=True)

app = Utils_streamlit.MultiApp()

st.sidebar.title("SIGNATURE VERIFICATION")
st.sidebar.success("SIGNATURE VERIFICATION using **Artificial Intelligence**!")
app.add_app("Home Page", about_page.app)
app.add_app("Signature Verification", signature_page.app)
app.run()
