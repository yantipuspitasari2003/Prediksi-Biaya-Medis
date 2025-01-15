import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("Regression.csv")

# Preprocessing
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
X = data.drop('charges', axis=1)
y = data['charges']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

dec_tree = DecisionTreeRegressor(random_state=42)
dec_tree.fit(X_train, y_train)

# Streamlit app
st.title("Prediksi Biaya Medis")

# Sidebar input
st.sidebar.header("Masukkan Parameter")

def user_input():
    age = st.sidebar.slider('Usia', 18, 64, 30)
    bmi = st.sidebar.slider('BMI', 15.0, 53.1, 30.0)
    children = st.sidebar.slider('Jumlah Anak', 0, 5, 1)
    smoker = st.sidebar.selectbox('Merokok', ['Tidak', 'Ya'])
    gender = st.sidebar.selectbox('Jenis Kelamin', ['Perempuan', 'Laki-laki'])
    
    # Pilihan wilayah dengan nama lebih deskriptif
    region = st.sidebar.selectbox('Wilayah', ['Tenggara', 'Timur Laut', 'Barat Laut', 'Barat Daya'])

    # Terjemahan wilayah ke dalam format dummy variable
    region_northwest = 1 if region == 'Barat Laut' else 0
    region_southeast = 1 if region == 'Tenggara' else 0
    region_southwest = 1 if region == 'Barat Daya' else 0

    return {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if gender == 'Laki-laki' else 0,
        'smoker_yes': 1 if smoker == 'Ya' else 0,
        'region_northwest': region_northwest,
        'region_southeast': region_southeast,
        'region_southwest': region_southwest
    }

input_data = pd.DataFrame(user_input(), index=[0])
st.subheader("Input Pengguna")
st.write(input_data)

# Model selection and prediction
st.subheader("Hasil Prediksi")

lin_reg_prediction = lin_reg.predict(input_data)[0]
dec_tree_prediction = dec_tree.predict(input_data)[0]
combined_prediction = (lin_reg_prediction + dec_tree_prediction) / 2

model = st.selectbox("Pilih Model", ["Regresi Linier", "Pohon Keputusan", "Kombinasi (Rata-rata)"])
if model == "Regresi Linier":
    prediction = lin_reg_prediction
elif model == "Decision Tree":
    prediction = dec_tree_prediction
else:
    prediction = combined_prediction

st.write(f"**Biaya Medis yang Diprediksi: {prediction:,.2f}**")

# Display all predictions
st.subheader("Hasil Prediksi dari Semua Model")
st.write(f"Regresi Linier: {lin_reg_prediction:,.2f}")
st.write(f"Pohon Keputusan: {dec_tree_prediction:,.2f}")
st.write(f"Kombinasi (Rata-rata): {combined_prediction:,.2f}")

# Evaluation
st.subheader("Kinerja Model")
if model == "Regresi Linier":
    y_pred = lin_reg.predict(X_test)
elif model == "Pohon Keputusan":
    y_pred = dec_tree.predict(X_test)
else:
    y_pred_lin = lin_reg.predict(X_test)
    y_pred_tree = dec_tree.predict(X_test)
    y_pred = (y_pred_lin + y_pred_tree) / 2

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
st.write(f"MAE: {mae:,.2f}")
st.write(f"MSE: {mse:,.2f}")

# Visualizations
st.subheader("Visualisasi Data")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(data['age'], kde=True, bins=20, ax=ax[0], color='blue')
ax[0].set_title("Distribusi Usia")

sns.histplot(data['charges'], kde=True, bins=20, ax=ax[1], color='green')
ax[1].set_title("Distribusi Biaya Medis")

st.pyplot(fig)

# Heatmap
st.subheader("Analisis Korelasi")
correlation_matrix = data.corr()
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis', ax=ax_corr, cbar=True, square=True, linewidths=.5)
ax_corr.set_title("Matriks Korelasi", fontsize=16)
st.pyplot(fig_corr)

# Catplot
st.subheader("Distribusi Berdasarkan Wilayah")
fig_cat = sns.catplot(data=data, x='region_southeast', y='charges', hue='smoker_yes', kind='bar', height=6, aspect=1.5, palette='muted')
fig_cat.set_axis_labels("Wilayah Tenggara (1=Ya)", "Biaya Medis").set_titles("Distribusi Biaya Medis Berdasarkan Wilayah dan Kebiasaan Merokok")
st.pyplot(fig_cat)

# Health Recommendations
if prediction > 5000:
    st.warning("Biaya medis cukup tinggi. Pertimbangkan pola hidup sehat seperti mengurangi merokok dan menjaga berat badan.")
else:
    st.success("Biaya medis terprediksi rendah. Pertahankan gaya hidup sehat!")

st.markdown("---")
st.markdown("Aplikasi ini memprediksi biaya medis berdasarkan parameter pengguna menggunakan Regresi Linier, Decision Tree, dan Kombinasi.")
    