import streamlit as st

from Model import feature_extraction
from Utils import Utils_streamlit


def scale_score(score, threshold, min_actual=0.0, max_actual=100.0):
    min_score_not_matched = 5.0
    max_score_not_matched = 65.0
    min_score_matched = 85.0
    max_score_matched = 99.0
    if score <= threshold:  # scale between 5-65
        score = (max_score_not_matched - min_score_not_matched) * (score - min_actual) / \
                (threshold - min_actual) + min_score_not_matched
    else:  # scale between 85-100
        score = (max_score_matched - min_score_matched) * (score - threshold) / \
                (max_actual - threshold) + min_score_matched
    return score


def execute(image1, image2, pre_trained_model_option, threshold):
    cosine_similarity = feature_extraction.get_cosine_similarity(image1, image2, pre_trained_model_option)
    score = scale_score(cosine_similarity, threshold)
    cosine_similarity_percentage = f"Matching Score: {score:.2f}%"
    st.success(cosine_similarity_percentage)
    if cosine_similarity < threshold:
        st.warning("The signatures don't match!!")
    else:
        st.success("The signatures match!!")


def app():
    image1_uploader = st.file_uploader('Choose original signature..')
    image2_uploader = st.file_uploader('Choose signature to be verified..')
    image1 = image2 = None
    if image1_uploader:
        image1 = Utils_streamlit.get_image(image1_uploader)
    if image2_uploader:
        image2 = Utils_streamlit.get_image(image2_uploader)

    if image1 is not None and image2 is not None:
        threshold = st.sidebar.number_input('threshold', value=70)
        col1, col2 = st.columns(2)
        col1.image(image1, use_column_width=True, caption="Benchmarked")
        col2.image(image2, use_column_width=True, caption="New Signature")
        model_option = st.sidebar.selectbox('Select the algorithm to be used for feature extraction',
                                            ('VGG16', 'VGG19', 'ResNet50'))
        submit_button = st.sidebar.button(label='Calculate Matching Score')
        if submit_button:
            execute(image1, image2, model_option, threshold)


if __name__ == '__main__':
    pass
