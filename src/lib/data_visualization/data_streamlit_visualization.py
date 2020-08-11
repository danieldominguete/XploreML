"""
===========================================================================================
Data Visualization with Streamlit Package
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(color_codes=True)
sns.set(style="darkgrid")

MAX_CATEGORIES_FOR_PLOT = 500


class DataVisualization:
    def __init__(self, title="Title", data_param=None):
        """Constructor for this class"""
        self.title = title
        self.data_param = data_param

    def update_page(self, data):

        # Title
        st.title(self.title)

        # Dataframe
        st.subheader("Raw data")
        st.write("Original data table")
        st.write(data)

        # Numerical features
        option = st.selectbox(
            "Select a numerical feature to visualize:", self.data_param.numerical_variables
        )
        st.line_chart(data[option])

        plt.hist(data[option])
        st.pyplot()

        # Numerical features
        optionx = st.selectbox(
            "Select a numerical feature to visualize:",
            self.data_param.numerical_variables,
            key="optionx",
        )
        optiony = st.selectbox(
            "Select a numerical feature to visualize:",
            self.data_param.numerical_variables,
            key="optiony",
        )

        plt.scatter(data[optionx], data[optiony])
        st.pyplot()

        # Categorical features
        option = st.selectbox(
            "Select a categorical feature to visualize:",
            self.data_param.categorical_variables,
        )
        number = st.number_input("Insert a number of top categories", min_value=1)
        hist = data[option].value_counts()[:number].sort_values(ascending=False)
        st.bar_chart(hist)
