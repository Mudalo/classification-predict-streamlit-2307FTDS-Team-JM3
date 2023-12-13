"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file



# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app

def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
 

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "About Us", "Model Explainer"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.markdown("<h1 style='color: black; font-weight: bold;'>Tweet Classifier</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: black; font-weight: bold;'>Climate change tweet classification</h2>", unsafe_allow_html=True)
        st.info("Introduction")
        # You can read a markdown file from supporting resources folder
        # st.markdown("Some information here")
        st.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +"Climate change refers to the permanent shifts in weather patterns and temperatures as a result of volcanic eruptions or changes in the sun's activity. Since the 1800s, human activities have been the key factor in climate change as a result of burning fossil fuels such as coal, oil, and gas. This leads to a rise in the Earth's temperature due to greenhouse gases, mainly carbon dioxide and methane. The consequences of climate change are deadly, leading to intense droughts, severe fires, rising sea levels, flooding, declining biodiversity, and many more."+"</p>", unsafe_allow_html=True)


        image = Image.open("resources/imgs/Information.png")
        st.image(image, use_column_width=True)
        st.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +"With the worldwide use of social media, with Facebook and Twitter occupying the top two spots for the most used social media applications, there's a gap in gathering crucial information relating to environmental issues using social media. "+"</p>", unsafe_allow_html=True)
        st.info("Purpose")
        st.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +" We combine market analysis, data science, and app development to create innovative web applications that allow companies to access a wide range of consumer sentiment from anywhere in the world with just a tap of a button!"+"</p>", unsafe_allow_html=True)
        st.info("How to use the App")
        markdown_text = """
- **Open predictions in the drop-down menu.**
- **Select model for classification:**
  - *For more information on the models, refer to the model explanation page.*
- **Type/paste text to be classified in the text box.**
- **Click classify to get your results!**
  - *For more information on your result, refer to the model explanation page.*
"""
        st.markdown(markdown_text, unsafe_allow_html=True)
# st.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +" "+"</p>", unsafe_allow_html=True)
         
         
        




        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if the box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page

    # Building out the prediction page
    if selection == "Prediction":
        st.markdown("<h1 style='color: black; font-weight: bold;'>Tweet Classifier</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: black; font-weight: bold;'>Climate change tweet classification</h2>", unsafe_allow_html=True)
        st.info("Prediction with ML Models")
        selected_option = st.selectbox("Select a model:", ["Logistic Regression", "Support Vector", "Decision Tree"])

        if selected_option == "Logistic Regression":
            predictor = joblib.load(open(os.path.join("resources/model_lr.pkl"), "rb"))

        elif selected_option == "Support Vector":
            predictor = joblib.load(open(os.path.join("resources/model_svc.pkl"), "rb"))

        elif selected_option == "Decision Tree":
            predictor = joblib.load(open(os.path.join("resources/model_dt.pkl"), "rb"))

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            prediction = predictor.predict(vect_text)

            if prediction == -1:
                predicted = "anti climate change"
            elif prediction == 0:
                predicted = 'neutral'
            elif prediction == 1:
                predicted = 'pro climate change'
            elif prediction == 2:
                predicted = 'News'

            # When the model has successfully run, will print prediction
            # You can use a dictionary or a similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(predicted))

    # Building About us page
    if selection == "About Us":
        st.markdown("<h1 style='color: green; font-weight: bold;'>Green Data Solutions</h1>", unsafe_allow_html=True)
        #st.markdown("<h2 style='color: black; font-weight: bold;'>Climate change tweet classification</h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: green;'>Who we are</h1>", unsafe_allow_html=True)

        st.markdown("<p style='color: black; font-size: 1.2em; font-weight: bold;'>" +
            "Welcome to Green Data Solutions, a trailblazing data analytics company committed to driving sustainable change through the power of data science. Specializing in Sustainable Data Solutions, we empower businesses to make smarter decisions and foster growth through insightful data analysis." +
            "</p>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        col1.markdown("**<h3 style='color:black'>Vision</h3>**", unsafe_allow_html=True)
        col2.markdown("**<h3 style='color:black'>Mission</h3>**", unsafe_allow_html=True)
        col1.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +"At Green Data Solutions, we're not just about analytics; we're architects of a smarter, greener world, where every data point contributes to a sustainable future."+"</p>", unsafe_allow_html=True)
        col2.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +"Our mission is to promote green technology and combat climate change by extracting meaningful insights that lead to informed, eco-conscious choices."+"</p>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: green;'>The team</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.75, 1.5, 1.75])
        col1.markdown("**<h3 style='color:black'>Name</h3>**", unsafe_allow_html=True)
        col1.info("Mudalo Ramadi")
        col2.markdown("**<h3 style='color:black'>Role</h3>**", unsafe_allow_html=True)
        col2.info("Team Lead")
        col3.markdown("**<h3 style='color:black'>Email</h3>**", unsafe_allow_html=True)
        col3.info("sanelehadebe070@gmail.com")
        col1.info("Precious Ratlhagane")
        col2.info("Project Manager")
        col3.info("sanelehadebe070@gmail.com")
        col1.info("Daluxolo Hadebe")
        col2.info("Technical Lead")
        col3.info("sanelehadebe070@gmail.com")
        col1.info("Thembi Chauke")
        col2.info("Administrator")
        col3.info("sanelehadebe070@gmail.com")
        col1.info("Ivan Cronje")
        col2.info("Data Scientist")
        col3.info("sanelehadebe070@gmail.com")
        col1.info("Carol Ndlovu")
        col2.info("Data Scientist")
        col3.info("sanelehadebe070@gmail.com")
        st.markdown("<h2 style='text-align: center; color: green;'>Clients</h1>", unsafe_allow_html=True)
        image = Image.open("resources/imgs/Clients.jpeg")
        st.image(image,use_column_width=True)




    #Building model explanations page
    if selection == "Model Explainer":
        st.info('Model Selection')
        st.markdown("<h1 style='text-align: center; color: black;'>Logistic Regression</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: black; font-size: 1.2em; font-weight: bold;'>" +
                    "A logistic classifcation model is typically used for predicting the outcome of a classification problem by picking from two possible outcomes. It works by calculating the probability of an observation belonging to a particular class using a threshold that is typically equal to one half." +
            "</p>", unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; color: black;'>Support Vector Classifier</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: black; font-size: 1.2em; font-weight: bold;'>" +
        "A Support Vector Classifier(SVC) model works by finding the most optimal way possible in order to seperate classes by considering the closest point on either side of the boundary. The boundary can either be linear or non-linear depending on the task at hand." +
            "</p>", unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; color: black;'>Decision Tree Classifier</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: black; font-size: 1.2em; font-weight: bold;'>" +
        "A Decision Tree Classifier model works by asking a series of questions concerning the input data and using each answer is used to make a decision depending on the specified features. This is a iterative process that eventually leads to the final classification at the end of the flowchart." +
                    "</p>", unsafe_allow_html=True)
        




   
            

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()

