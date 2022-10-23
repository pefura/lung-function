# Import packages
import pandas as pd
import numpy as np
import streamlit as st


# App title
st.write("""
# Wellcome to Cameroonian's lung function  calculator App
By Pefura-Yone et al. 
""")

st.write(""" ### Reference of Article used
#### Spirometric reference equations for Cameroonians aged 4 to 89 years derived using lambda, mu, sigma (LMS) method
##### BMC Pulm Med. 2021 Nov 3;21(1):344. doi: 10.1186/s12890-021-01705-1 
""")

# Define user's input parameters

st.write(""" 
### Patients Input Parameters(please select patients features here)
""")
age = st.number_input('age, years', 4.0, 89.0)
gender = st.number_input('gender (Male=1, Female=2)',  1, 2)
height = st.number_input('height, cm', 103.0,196.0 )
FEV1_measured = st.number_input('FEV1_measured, L', 0.0, 15.0 )
FVC_measured = st.number_input('FVC_measured, L', 0.0, 15.0)
FEV1FVC_measured = st.number_input('FEV1FVC_measured, L', 0.0, 1.0)
FEF2575_measured = st.number_input('FEF2575_measured, L', 0.0, 15.0)

st.write(""" 
### Confirmed patient's data
""")

df = pd.DataFrame([round(age,2), gender, height, FEV1_measured, FVC_measured, FEV1FVC_measured,FEF2575_measured]).T.set_axis(['Age', 'Gender', 'Height', 'FEV1 measured,L', 'FVC measured,L', 'FEV1/FVC measured','FEF2575 measured,L/s'], axis='columns')
st.write(df)

# Import Splines lookup table
lookup= pd.read_csv("https://raw.githubusercontent.com/pefura/lung-function/main/lookup_py.csv", sep=';')

# Function to calculate spirometric parameters
def spirometric_parameters(age, gender, height, FEV1_measured=0, FVC_measured=0, FEV1FVC_measured=0,
                           FEF2575_measured=0):
    agebound = int(age * 4) / 4
    # For FEV1
    FEV1 = lookup.loc[lookup['f'] == 'FEV1']
    FEV1 = FEV1.loc[FEV1['gender'] == gender]

    a0_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['a0']
    a1_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['a1']
    a2_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['a2']

    Mspline_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['m0'] + (FEV1.loc[FEV1['agebound'] == agebound]['m1'] -
                                                                   FEV1.loc[FEV1['agebound'] == agebound]['m0']) * (
                               age - agebound) / (0.25)
    Sspline_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['s0'] + (FEV1.loc[FEV1['agebound'] == agebound]['s1'] -
                                                                   FEV1.loc[FEV1['agebound'] == agebound]['s0']) * (
                               age - agebound) / (0.25)
    Lspline_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['l0'] + (FEV1.loc[FEV1['agebound'] == agebound]['l1'] -
                                                                   FEV1.loc[FEV1['agebound'] == agebound]['l0']) * (
                               age - agebound) / (0.25)

    FEV1_predicted = np.exp(
        a0_FEV1 + (a1_FEV1 * np.log(height)) + (a2_FEV1 * np.log(age)) + Mspline_FEV1)  # compute mu
    FEV1_predicted = np.array(FEV1_predicted)[0]
    FEV1_predicted = round(FEV1_predicted, 3)

    p0_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['p0']
    p1_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['p1']
    FEV1_S = np.exp(p0_FEV1 + (p1_FEV1 * np.log(age)) + Sspline_FEV1)

    q0_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['q0']
    q1_FEV1 = FEV1.loc[FEV1['agebound'] == agebound]['q1']
    FEV1_L = q0_FEV1 + (q1_FEV1 * np.log(age)) + Lspline_FEV1

    FEV1_LLN = FEV1_predicted * ((-1.645 * FEV1_S * FEV1_L) + 1) ** (1 / FEV1_L)  # compute lower limit of normal
    FEV1_LLN = round(np.array(FEV1_LLN)[0], 3)

    FEV1_zscore = (((FEV1_measured / FEV1_predicted) ** FEV1_L) - 1) / (FEV1_L * FEV1_S)  # compute Z-score
    FEV1_zscore = round(np.array(FEV1_zscore)[0], 3)

    percent_predicted_FEV1 = (FEV1_measured / FEV1_predicted) * 100
    percent_predicted_FEV1 = round(percent_predicted_FEV1, 3)

    # For FVC

    FVC = lookup.loc[lookup['f'] == 'FVC']
    FVC = FVC.loc[FVC['gender'] == gender]

    a0_FVC = FVC.loc[FVC['agebound'] == agebound]['a0']
    a1_FVC = FVC.loc[FVC['agebound'] == agebound]['a1']
    a2_FVC = FVC.loc[FVC['agebound'] == agebound]['a2']

    Mspline_FVC = FVC.loc[FVC['agebound'] == agebound]['m0'] + (FVC.loc[FVC['agebound'] == agebound]['m1'] -
                                                                FVC.loc[FVC['agebound'] == agebound]['m0']) * (
                              age - agebound) / (0.25)
    Sspline_FVC = FVC.loc[FVC['agebound'] == agebound]['s0'] + (FVC.loc[FVC['agebound'] == agebound]['s1'] -
                                                                FVC.loc[FVC['agebound'] == agebound]['s0']) * (
                              age - agebound) / (0.25)
    Lspline_FVC = FVC.loc[FVC['agebound'] == agebound]['l0'] + (FVC.loc[FVC['agebound'] == agebound]['l1'] -
                                                                FVC.loc[FVC['agebound'] == agebound]['l0']) * (
                              age - agebound) / (0.25)

    FVC_predicted = np.exp(a0_FVC + (a1_FVC * np.log(height)) + (a2_FVC * np.log(age)) + Mspline_FVC)
    FVC_predicted = np.array(FVC_predicted)[0]
    FVC_predicted = round(FVC_predicted, 3)

    p0_FVC = FVC.loc[FVC['agebound'] == agebound]['p0']
    p1_FVC = FVC.loc[FVC['agebound'] == agebound]['p1']
    FVC_S = np.exp(p0_FVC + (p1_FVC * np.log(age)) + Sspline_FVC)

    q0_FVC = FVC.loc[FVC['agebound'] == agebound]['q0']
    q1_FVC = FVC.loc[FVC['agebound'] == agebound]['q1']
    FVC_L = q0_FVC + (q1_FVC * np.log(age)) + Lspline_FVC

    FVC_LLN = FVC_predicted * ((-1.645 * FVC_S * FVC_L) + 1) ** (1 / FVC_L)
    FVC_LLN = round(np.array(FVC_LLN)[0], 3)

    FVC_zscore = (((FVC_measured / FVC_predicted) ** FVC_L) - 1) / (FVC_L * FVC_S)
    FVC_zscore = round(np.array(FVC_zscore)[0], 3)

    percent_predicted_FVC = (FVC_measured / FVC_predicted) * 100
    percent_predicted_FVC = round(percent_predicted_FVC, 3)

    # For FEV1/FVC ratio

    FEV1FVC = lookup.loc[lookup['f'] == 'FEV1FVC']
    FEV1FVC = FEV1FVC.loc[FEV1FVC['gender'] == gender]

    a0_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['a0']
    a1_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['a1']
    a2_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['a2']

    Mspline_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['m0'] + (
                FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['m1'] -
                FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['m0']) * (age - agebound) / (0.25)
    Sspline_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['s0'] + (
                FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['s1'] -
                FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['s0']) * (age - agebound) / (0.25)
    Lspline_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['l0'] + (
                FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['l1'] -
                FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['l0']) * (age - agebound) / (0.25)

    FEV1FVC_predicted = np.exp(
        a0_FEV1FVC + (a1_FEV1FVC * np.log(height)) + (a2_FEV1FVC * np.log(age)) + Mspline_FEV1FVC)
    FEV1FVC_predicted = np.array(FEV1FVC_predicted)[0]
    FEV1FVC_predicted = round(FEV1FVC_predicted, 3)

    p0_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['p0']
    p1_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['p1']
    FEV1FVC_S = np.exp(p0_FEV1FVC + (p1_FEV1FVC * np.log(age)) + Sspline_FEV1FVC)

    q0_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['q0']
    q1_FEV1FVC = FEV1FVC.loc[FEV1FVC['agebound'] == agebound]['q1']
    FEV1FVC_L = q0_FEV1FVC + (q1_FEV1FVC * np.log(age)) + Lspline_FEV1FVC

    FEV1FVC_LLN = FEV1FVC_predicted * ((-1.645 * FEV1FVC_S * FEV1FVC_L) + 1) ** (1 / FEV1FVC_L)
    FEV1FVC_LLN = round(np.array(FEV1FVC_LLN)[0], 3)

    FEV1FVC_zscore = (((FEV1FVC_measured / FEV1FVC_predicted) ** FEV1FVC_L) - 1) / (FEV1FVC_L * FEV1FVC_S)
    FEV1FVC_zscore = round(np.array(FEV1FVC_zscore)[0], 3)

    percent_predicted_FEV1FVC = (FEV1FVC_measured / FEV1FVC_predicted) * 100
    percent_predicted_FEV1FVC = round(percent_predicted_FEV1FVC, 3)

    # For FEF25-75%

    FEF2575 = lookup.loc[lookup['f'] == 'FEF2575']
    FEF2575 = FEF2575.loc[FEF2575['gender'] == gender]

    a0_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['a0']
    a1_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['a1']
    a2_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['a2']

    Mspline_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['m0'] + (
                FEF2575.loc[FEF2575['agebound'] == agebound]['m1'] -
                FEF2575.loc[FEF2575['agebound'] == agebound]['m0']) * (age - agebound) / (0.25)
    Sspline_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['s0'] + (
                FEF2575.loc[FEF2575['agebound'] == agebound]['s1'] -
                FEF2575.loc[FEF2575['agebound'] == agebound]['s0']) * (age - agebound) / (0.25)
    Lspline_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['l0'] + (
                FEF2575.loc[FEF2575['agebound'] == agebound]['l1'] -
                FEF2575.loc[FEF2575['agebound'] == agebound]['l0']) * (age - agebound) / (0.25)

    FEF2575_predicted = np.exp(
        a0_FEF2575 + (a1_FEF2575 * np.log(height)) + (a2_FEF2575 * np.log(age)) + Mspline_FEF2575)
    FEF2575_predicted = np.array(FEF2575_predicted)[0]
    FEF2575_predicted = round(FEF2575_predicted, 3)

    p0_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['p0']
    p1_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['p1']
    FEF2575_S = np.exp(p0_FEF2575 + (p1_FEF2575 * np.log(age)) + Sspline_FEF2575)

    q0_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['q0']
    q1_FEF2575 = FEF2575.loc[FEF2575['agebound'] == agebound]['q1']
    FEF2575_L = q0_FEF2575 + (q1_FEF2575 * np.log(age)) + Lspline_FEF2575

    FEF2575_LLN = FEF2575_predicted * ((-1.645 * FEF2575_S * FEF2575_L) + 1) ** (1 / FEF2575_L)
    FEF2575_LLN = round(np.array(FEF2575_LLN)[0], 3)

    FEF2575_zscore = (((FEF2575_measured / FEF2575_predicted) ** FEF2575_L) - 1) / (FEF2575_L * FEF2575_S)
    FEF2575_zscore = round(np.array(FEF2575_zscore)[0], 3)

    percent_predicted_FEF2575 = (FEF2575_measured / FEF2575_predicted) * 100
    percent_predicted_FEF2575 = round(percent_predicted_FEF2575, 3)

    return FEV1_measured, FEV1_predicted, FEV1_LLN, FEV1_zscore, percent_predicted_FEV1, FVC_measured, FVC_predicted, FVC_LLN, FVC_zscore, percent_predicted_FVC, FEV1FVC_measured, FEV1FVC_predicted, FEV1FVC_LLN, FEV1FVC_zscore, percent_predicted_FEV1FVC, FEF2575_measured, FEF2575_predicted, FEF2575_LLN, FEF2575_zscore, percent_predicted_FEF2575


st.write(""" 
### Spirometric indices
""")

spiro = spirometric_parameters(age=age, gender= gender, height=height, FEV1_measured=FEV1_measured, FVC_measured=FVC_measured,FEV1FVC_measured= FEV1FVC_measured,
                           FEF2575_measured=FEF2575_measured)
spiro = np.array(spiro).reshape(4,5)
spiro_df = pd.DataFrame(spiro).set_axis(['Measured', 'Predicted', 'LLN', 'Z-score', '%Predicted'], axis='columns')
spiro_df = spiro_df.rename(index={0: "FEV1,L", 1:"FVC,L", 2:"FEV1/FVC", 3:"FEF25-75%,L/s"})
st.write(spiro_df)
st.write(""" 
#### FEV1 = forced expiratory volume in 1s ; FVC = forced vital capacity; FEV1/FVC = FEV1/FVC ratio; FEF25-75% = forced mid-expiratory flow; LLN = lower limit of normal
""")
