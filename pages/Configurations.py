

import streamlit as st


st.markdown("""
    ## Connected with SAP IBP
    ### Configurations 
            """)



URL_of_IBP = st.text_input("IBP URL",'https://my400305.scmibp.ondemand.com/')
PA_of_IBP = st.text_input("Planning Area",'WEPROD')
account_of_IBP = st.text_input("E-mail",'David.jason@westernacher.com')
pw_of_IBP = st.text_input("Password ",'davidxxx0319')

save_config = st.button('SAVE')
if save_config:
    st.write("Change saved")
