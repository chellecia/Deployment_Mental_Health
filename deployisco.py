import streamlit as st
import pickle
import numpy as np
from deprecated import deprecated
from streamlit_option_menu import option_menu
import streamlit as st
import pickle
import numpy as np
from pycaret.classification import * 
# Muat model
 
model_anxiety = load_model('model_anxiety')
model_stress = load_model('model_stress')
model_depression = load_model('model_depression')


# Define the min and max values for user input

# Streamlit App
st.title("Mental Health Prediction App")

# User input
# Daftar pilihan yang ditampilkan di selectbox
display_options_age = ["18-22", "23-26", "27-30"]
age = st.selectbox("Age", options=display_options_age)
value_options_age = [1, 2, 3]
age = value_options_age[display_options_age.index(age)]

display_options_gender = ["Male", "Female"]
gender = st.selectbox("Gender", options = display_options_gender)
value_options_gender = [1,2]
gender = value_options_gender[display_options_gender.index(gender)]

department = st.selectbox(
    "Select your department:",
    [
        'CS/IT Engineering',
        'Other',
        'EEE/ECE Engineering',
        'Env/Life Sciences',
        'Business/Entrepreneurship',
        'Biological Sciences',
        'Civil Engineering',
        'Mechanical Engineering'
    ]
)


display_options_academic = ["First Year", "Second Year", "Third Year", "Fourth Year"]
academic_year = st.selectbox("Academic Year", options=display_options_academic)
value_options_academic = [1,2,3,4]
academic_year = value_options_academic[display_options_academic.index(academic_year)]

display_options_cgpa = ["Below 2.50", "2.50 - 2.99", "3.00 - 3.39", "3.40 - 3.79", "3.80 - 4.00", "Other"]
cgpa = st.selectbox("CGPA", options=display_options_cgpa)
value_options_cgpa = [1, 2, 3, 4, 5, 6]
cgpa = value_options_cgpa[display_options_cgpa.index(cgpa)]


display_options_scholarship = ["Yes", "No"]
scholarship = st.selectbox("Are you on a scholarship?", display_options_scholarship)
value_options_scholarship = [1,2]
scholarship = value_options_scholarship[display_options_scholarship.index(scholarship)]

# Frequency inputs using sliders
nervous_frequency = st.radio("Nervous Frequency", [0,1,2,3])
worry_frequency = st.radio("Worry Frequency", [0,1,2,3])
relaxation_frequency = st.radio("Relaxation Frequency", [0,1,2,3])
irritation_frequency = st.radio("Irritation Frequency", [0,1,2,3])
overthinking_frequency = st.radio("Overthinking Frequency",[0,1,2,3])
restlessness_frequency = st.radio("Restlessness Frequency", [0,1,2,3])
fear_frequency = st.radio("Fear Frequency", [0,1,2,3])
upset_frequency = st.radio("Upset Frequency", [0,1,2,3,4])
control_frequency = st.radio("Control Frequency", [0,1,2,3,4])
stress_frequency = st.radio("Stress Frequency", [0,1,2,3,4])
coping_frequency = st.radio("Coping Frequency", [0,1,2,3,4])
confidence_frequency = st.radio("Confidence Frequency",[0,1,2,3,4])
life_control_frequency = st.radio("Life Control Frequency",[0,1,2,3,4])
irritation_control_frequency = st.radio("Irritation Control Frequency",[0,1,2,3,4])
top_performance_frequency = st.radio("Top Performance Frequency",[0,1,2,3,4])
anger_frequency = st.radio("Anger Frequency", [0,1,2,3,4])
overwhelm_frequency = st.radio("Overwhelm Frequency", [0,1,2,3,4])
no_interest_frequency = st.radio("No Interest Frequency", [0,1,2,3])
hopeless_frequency = st.radio("Hopeless Frequency", [0,1,2,3])
sleep_issues_frequency = st.radio("Sleep Issues Frequency", [0,1,2,3])
low_energy_frequency = st.radio("Low Energy Frequency", [0,1,2,3])
appetite_issues_frequency = st.radio("Appetite Issues Frequency", [0,1,2,3])
self_worth_frequency = st.radio("Self Worth Frequency",[0,1,2,3])
concentration_issues_frequency = st.radio("concentration_issues_frequency",[0,1,2,3])
slow_movement_frequency = st.radio("Slow Movement Frequency", [0,1,2,3])
suicidal_thoughts_frequency = st.radio("Suicidal Thoughts Frequency", [0,1,2,3])

# Create input array for the model
input_data = np.array([[age , gender , department , academic_year , cgpa , scholarship ,
                        nervous_frequency,
worry_frequency,
relaxation_frequency,
irritation_frequency,
overthinking_frequency,
restlessness_frequency,
fear_frequency,
upset_frequency,
control_frequency,
stress_frequency,
coping_frequency,
confidence_frequency,
life_control_frequency,
irritation_control_frequency,
top_performance_frequency,
anger_frequency,
overwhelm_frequency,
no_interest_frequency,
hopeless_frequency,
sleep_issues_frequency,
low_energy_frequency,
appetite_issues_frequency,
self_worth_frequency,
concentration_issues_frequency,
slow_movement_frequency,
suicidal_thoughts_frequency
]])
 
 
import pandas as pd 
df_2 = pd.DataFrame(input_data)
df_2.columns = ['Age', 'Gender', 'Department', 'AcademicYear', 'CGPA', 'Scholarship',
 'NervousFreq', 'WorryFreq', 'RelaxationFreq', 'IrritationFreq',
 'OverthinkingFreq', 'RestlessnessFreq', 'FearFreq', 'UpsetFreq',
 'ControlFreq', 'StressFreq', 'CopingFreq', 'ConfidenceFreq',
 'LifeControlFreq', 'IrritationControlFreq', 'TopPerformanceFreq',
 'AngerFreq', 'OverwhelmFreq', 'NoInterestFreq', 'HopelessFreq',
 'SleepIssuesFreq', 'LowEnergyFreq', 'AppetiteIssuesFreq',
 'SelfWorthFreq', 'ConcentrationIssuesFreq', 'SlowMovementFreq',
 'SuicidalThoughtsFreq']

# Convert categorical data to numerical format if necessary
# This part depends on how your model was trained. Update accordingly.
 
# Make predictions
# Make predictions
if st.button("Prediksi"):
    # Lakukan prediksi
    anxiety_prediction = predict_model(model_anxiety, df_2)
    stress_prediction = predict_model(model_stress, df_2)
    depression_prediction = predict_model(model_depression, df_2)

    # # Tampilkan DataFrame prediksi secara lengkap
    # st.subheader("Output Prediksi Kecemasan")
    # st.write(anxiety_prediction)  # Tampilkan semua kolom untuk inspeksi

    # st.subheader("Output Prediksi Stres")
    # st.write(stress_prediction)  # Tampilkan semua kolom untuk inspeksi

    # st.subheader("Output Prediksi Depresi")
    # st.write(depression_prediction)  # Tampilkan semua kolom untuk inspeksi

    # Ambil label prediksi
    anxiety_label = anxiety_prediction['prediction_label'].iloc[0]  
    stress_label = stress_prediction['prediction_label'].iloc[0]    
    depression_label = depression_prediction['prediction_label'].iloc[0]  

    # Tampilkan hasil
    st.subheader("Prediksi")
    st.write(f"Tingkat Kecemasan: {anxiety_label}")
    st.write(f"Tingkat Stres: {stress_label}")
    st.write(f"Tingkat Depresi: {depression_label}")
