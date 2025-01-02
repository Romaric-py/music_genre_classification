import streamlit as st

def run():
    st.header("Classification report for each model")

    columns = st.columns(2, gap='large', )
    with columns[0]:
        st.subheader("LogisticRegression")
        st.markdown('''

| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| blues      | 0.85      | 0.73   | 0.79     |
| classical  | 0.97      | 0.97   | 0.97     |
| country    | 0.85      | 0.73   | 0.79     |
| disco      | 0.70      | 0.63   | 0.67     |
| hiphop     | 0.71      | 0.67   | 0.69     |
| jazz       | 0.80      | 0.93   | 0.86     |
| metal      | 0.84      | 0.87   | 0.85     |
| pop        | 0.81      | 0.87   | 0.84     |
| reggae     | 0.58      | 0.70   | 0.64     |
| rock       | 0.66      | 0.63   | 0.64     |
| **accuracy** |           |        | 0.77     |
| **macro avg** | 0.78      | 0.77   | 0.77     |
| **weighted avg** | 0.78   | 0.77   | 0.77     |
'''
        )
    
    with columns[1]:
        st.subheader("LinearSVC")
        st.markdown('''

| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| blues      | 0.88      | 0.77   | 0.82     |
| classical  | 0.94      | 0.97   | 0.95     |
| country    | 0.77      | 0.80   | 0.79     |
| disco      | 0.64      | 0.47   | 0.54     |
| hiphop     | 0.79      | 0.73   | 0.76     |
| jazz       | 0.80      | 0.93   | 0.86     |
| metal      | 0.81      | 0.83   | 0.82     |
| pop        | 0.79      | 0.87   | 0.83     |
| reggae     | 0.61      | 0.73   | 0.67     |
| rock       | 0.59      | 0.53   | 0.56     |
| **accuracy** |           |        | 0.76     |
| **macro avg** | 0.76      | 0.76   | 0.76     |
| **weighted avg** | 0.76   | 0.76   | 0.76     |

'''
        )