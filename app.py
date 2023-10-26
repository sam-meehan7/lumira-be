# Import necessary libraries
from ask_vector_and_supplment_with_ai import answer_question
import streamlit as st

st.title("Ask a Question")

# Using st.form to create a form
with st.form(key='my_form'):
    # Create a text input for the user to enter their question
    question = st.text_input("Please enter your question:")

    # Create a submit button within the form
    submit_button = st.form_submit_button(label='Submit')


if submit_button:
    # Display the answer
    st.subheader("Answer:")
    answer, vector_results = answer_question(question)

    st.write(answer)

    st.subheader("Source Links:")
    for result in vector_results:
        # Extract necessary data from the result's metadata
        title = result.metadata["title"]
        url = result.metadata["url"]
        start_time = result.metadata["start"]

        # Construct the URL with start time appended
        clickable_url = f"{url}&t={int(start_time)}s"

        # Display the clickable link
        st.markdown(f"[{title}]({clickable_url})")


