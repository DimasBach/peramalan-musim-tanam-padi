import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import pickle
import matplotlib.pyplot as plt


st.markdown(
    "<h2 style='text-align: center;'>Peramalan Curah Hujan Berbasis Jaringan Syaraf Tiruan untuk Optimalisasi Musim Tanam Padi</h2><br><br><br>", unsafe_allow_html=True
)

# Fungsi untuk membuat lagged dataset
def create_lagged_dataset(data, lag=1):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i:(i + lag), 0])
        y.append(data[i + lag, 0])
    return np.array(X), np.array(y)

# Load dataset
# @st.cache
# def load_data():
#     data = pd.read_excel('data.xlsx', parse_dates=['Tanggal'])
#     return data
data = pd.read_excel('data.xlsx', parse_dates=['Tanggal'])

# Sidebar
with st.sidebar:
    selected = option_menu("Main Menu", ['Dataset', 'Preprocessing', 'Modelling', 'Prediction'], default_index=3)

# Menu Dataset
if (selected == 'Dataset'):
    st.info("Data curah hujan harian diperoleh dari Badan Meteorologi, Klimatologi, dan Geofisika (BMKG). Kabupaten Bangkalan tidak memiliki stasiun pengamatan cuaca, sehingga data curah hujan yang diolah dari hasil pengamatan stasiun pengamatan cuaca terdekat, yaitu Stasiun Meteorologi Perak I Surabaya.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Curah Hujan Harian")
        # Load data
        st.write(data)
    with col2:
        st.subheader('Informasi Data')
        st.success(f"Jumlah total data: {len(data)}")
        st.success(f"Jumlah data bernilai 8888: {data['Curah Hujan (RR)'].eq(8888).sum()}")
        st.success(f"Jumlah data kosong: {data['Curah Hujan (RR)'].isna().sum()}")

# Menu Preprocessing
if (selected == 'Preprocessing'):
    st.title("Preprocessing Data")
    st.info("""
    Adapun tahapan - tahapan yang akan dilakukan pada persiapan data ini adalah :
    1. Data Imputation
    2. Data Tranformation
    3. Lag Feature
    4. Normalisasi Data
    """)
    tabs = st.tabs(["Data Imputation", "Data Transformation", "Normalisasi Data", "Lag Feature", "Dataset"])
    
    
    # Tab Data Imputation
    with tabs[0]:
        st.subheader("Data Imputation")
        st.info(f"Jumlah data kosong sebelum imputasi: {data['Curah Hujan (RR)'].isna().sum()}")
        st.write("Dataset Sebelum Imputasi:")
        st.write(data)
        
        # Impute missing values
        data['Curah Hujan (RR)'] = data['Curah Hujan (RR)'].fillna(0)
        st.write("Dataset Setelah Imputasi:")
        st.write(data)
    
    # Tab Data Transformation
    with tabs[1]:
        st.subheader("Data Transformation")
        st.info(f"Jumlah data bernilai 8888 sebelum transformasi: {data['Curah Hujan (RR)'].eq(8888).sum()}")
        st.write("Dataset Sebelum Transformasi:")
        st.write(data)
        
        # Transform 8888 to 0
        data['Curah Hujan (RR)'] = data['Curah Hujan (RR)'].replace(8888, 0)
        st.write("Dataset Setelah Transformasi:")
        st.write(data)
    
    # Tab Normalisasi Data
    with tabs[2]:
        st.subheader("Normalisasi Data")
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data[['Curah Hujan (RR)']])
        
        st.write("Dataset Sebelum Normalisasi:")
        st.write(data)
        st.write("Dataset Setelah Normalisasi:")
        st.write(pd.DataFrame(normalized_data, columns=['Curah Hujan (RR)']))
    
    # Tab Lag Feature
    with tabs[3]:
        st.subheader("Lag Feature")
        lag = 7
        normalized_data = scaler.fit_transform(data[['Curah Hujan (RR)']])
        X, y = create_lagged_dataset(normalized_data, lag=lag)
        
        st.write("Data Lag Features (X):")
        st.write(pd.DataFrame(X, columns=[f"Lag_{i+1}" for i in range(lag)]))
        st.write("Data Target (y):")
        st.write(pd.DataFrame(y, columns=["Target"]))
        
        # Save to Excel
        lagged_data = pd.DataFrame(X, columns=[f"Lag_{i+1}" for i in range(lag)])
        lagged_data['Target'] = y
        lagged_data.to_excel("lagged_data.xlsx", index=False)
    
    # Tab Dataset
    with tabs[4]:
        st.subheader("Dataset Setelah Preprocessing")
        st.write(lagged_data)

# Menu Modelling
if selected == 'Modelling':
    st.title("Model Terbaik")

    # Menampilkan hasil model terbaik
    learning_rate = 1.0
    n_hidden_layers = 5
    best_neurons = 10
    best_mape = 0.40747792867862026
    best_mse = 0.007769943129798958
    best_rmse = 0.08806215051092786

    st.write("### Model Terbaik yang Diperoleh:")
    st.write(f"Learning rate: {learning_rate}")
    st.write(f"Jumlah hidden layer: {n_hidden_layers}")
    st.write(f"Jumlah neuron per hidden layer: {best_neurons}")
    st.write(f"MAPE: {best_mape}")
    st.write(f"MSE: {best_mse}")
    st.write(f"RMSE: {best_rmse}")

    # Data hasil evaluasi
    neurons = list(range(1, 11))
    mape_values = [
        20423916945395.94, 37261079084355.19, 5369255493.604682, 
        2981415241.7340407, 14135998.266705928, 4189395111.0605016, 
        9950668.047627462, 6720192.984157499, 6260.474182724277, 
        0.40747792867862026
    ]
    mse_values = [
        0.0074503978531068925, 0.007468252973203838, 0.007769819769491974,
        0.007769872493729609, 0.007769942792408822, 0.007769852438575209,
        0.00776994291429094, 0.007769942970547447, 0.00776994312965061,
        0.007769943129798958
    ]
    rmse_values = [
        0.08606892663435783, 0.08630405501125032, 0.08806147051115798,
        0.08806172577118167, 0.08806214856099807, 0.08806163774568794,
        0.08806214929251997, 0.08806214955323761, 0.08806215051003574,
        0.08806215051092786
    ]

    # Membuat grafik
    fig_mape, ax_mape = plt.subplots(figsize=(6, 4))
    ax_mape.plot(neurons, mape_values, marker='o', label="MAPE")
    ax_mape.set_title("MAPE vs Jumlah Neuron")
    ax_mape.set_xlabel("Jumlah Neuron per Hidden Layer")
    ax_mape.set_ylabel("MAPE")
    ax_mape.grid()
    ax_mape.legend()
    st.pyplot(fig_mape)


    fig_mse, ax_mse = plt.subplots(figsize=(6, 4))
    ax_mse.plot(neurons, mse_values, marker='o', color='orange', label="MSE")
    ax_mse.set_title("MSE vs Jumlah Neuron")
    ax_mse.set_xlabel("Jumlah Neuron per Hidden Layer")
    ax_mse.set_ylabel("MSE")
    ax_mse.grid()
    ax_mse.legend()
    st.pyplot(fig_mse)


    fig_rmse, ax_rmse = plt.subplots(figsize=(6, 4))
    ax_rmse.plot(neurons, rmse_values, marker='o', color='green', label="RMSE")
    ax_rmse.set_title("RMSE vs Jumlah Neuron")
    ax_rmse.set_xlabel("Jumlah Neuron per Hidden Layer")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.grid()
    ax_rmse.legend()
    st.pyplot(fig_rmse)
    
# Menu Prediction
if (selected == 'Prediction'):
    st.title("Prediksi")
    tabs = st.tabs(["Jadwal Tanam Padi", "Prediksi Curah Hujan"])
    
    # Tab Jadwal Tanam Padi
    with tabs[0]:
        st.subheader("Jadwal Tanam Padi")
        
        #  2. Clean the data
        data['Curah Hujan (RR)'] = data['Curah Hujan (RR)'].replace(8888, 0)
        data['Curah Hujan (RR)'] = data['Curah Hujan (RR)'].fillna(0)

        # Normalize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[['Curah Hujan (RR)']])

        # 3. Create dataset for time series (lagged data)
        def create_dataset(data, lag=1):
            X, y = [], []
            for i in range(len(data) - lag):
                X.append(data[i:(i + lag), 0])
                y.append(data[i + lag, 0])
            return np.array(X), np.array(y)

        lag = 7  # Using 7-day lag based on your friend's code
        X, y = create_dataset(data_scaled, lag)

        # 4. Train-test split
        train_size = int(len(X) * 0.8)  # Using 80:20 split
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 5. Define and train the BPNN model with best parameters
        bpnn = MLPRegressor(hidden_layer_sizes=(10,), learning_rate_init=1.0, max_iter=100, activation='logistic', solver='adam', random_state=0)
        bpnn.fit(X_train, y_train)

        # Save the trained model to a file
        model_filename = 'bpnn_best_model.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(bpnn, file)

        # 6. Forecast 365 days ahead
        predictions = []
        current_input = data_scaled[-lag:].flatten()  # Take the last available data as starting input for predictions

        for _ in range(365):
            # Predict the next value
            pred = bpnn.predict(current_input.reshape(1, -1))
            predictions.append(pred[0])

            # Update input with the latest prediction
            current_input = np.roll(current_input, -1)
            current_input[-1] = pred  # Insert the latest prediction

        # 7. Reverse normalization
        predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # 8. Create DataFrame from predictions
        future_dates = pd.date_range(start=data['Tanggal'].max() + pd.Timedelta(days=1), periods=365)
        predicted_df = pd.DataFrame({
            'Tanggal': future_dates,
            'Curah Hujan (RR) Prediksi': predictions_rescaled.flatten()
        })

        # 9. Monthly summary for planting schedule
        predicted_df['Bulan'] = predicted_df['Tanggal'].dt.month

        # Get last month from the original data
        last_month = data['Tanggal'].max().month

        # Shift months to start from the last month
        predicted_df['Bulan'] = (predicted_df['Bulan'] + (last_month - 1)) % 12 + 1

        monthly_summary = predicted_df.groupby('Bulan')['Curah Hujan (RR) Prediksi'].sum().reset_index()

        # Print last month and monthly summary
        # print(f"Bulan terakhir pada data: {last_month}")
        st.write("Ringkasan Prediksi Bulanan:")
        st.write(monthly_summary)

        # Dapatkan bulan dari tanggal terakhir pada dataset asli
        last_month = data['Tanggal'].max().month

        # Menggeser bulan sehingga dimulai dari bulan terakhir + 1
        predicted_df['Bulan'] = (predicted_df['Bulan'] + last_month) % 12 + 1

        # Tampilkan bulan terakhir
        print(f"Bulan terakhir pada data: {last_month}")

        # 8. Mengelompokkan berdasarkan bulan untuk menentukan jadwal tanam padi
        monthly_summary = predicted_df.groupby('Bulan')['Curah Hujan (RR) Prediksi'].sum().reset_index()

        # 10. Visualisasi per bulan menggunakan grafik garis
        # Visualisasi per bulan menggunakan grafik garis
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_summary['Bulan'], monthly_summary['Curah Hujan (RR) Prediksi'], marker='o', color='skyblue', linestyle='-')
        ax.set_title('Total Prediksi Curah Hujan per Bulan')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Total Curah Hujan (mm)')

        # Menyesuaikan nama bulan sesuai dengan urutan dari bulan terakhir
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
        shifted_month_names = month_names[last_month % 12:] + month_names[:last_month % 12]
        ax.set_xticks(monthly_summary['Bulan'])
        ax.set_xticklabels(shifted_month_names, rotation=45)
        ax.grid(axis='y')
        st.pyplot(fig)

    # Tab Prediksi Curah Hujan
    with tabs[1]:
        st.subheader("Prediksi Curah Hujan")
        
        # Input jumlah hari untuk prediksi
        days_to_predict = st.number_input("Masukkan jumlah hari untuk diprediksi (contoh: 30):", min_value=1, max_value=365, value=30, step=1)
        
        if st.button("Prediksi"):
            # Load model terbaik
            with open("bpnn_best_model.pkl", "rb") as f:
                model = pickle.load(f)

            predictions = []

            # Mengambil data terakhir untuk dijadikan input awal (misal, 7 lag terakhir)
            current_input = data_scaled[-7:].flatten()  # Gunakan data terakhir yang sudah di-*scale*

            for _ in range(int(days_to_predict)):
                # Prediksi menggunakan model backpropagation
                pred = model.predict(current_input.reshape(1, -1))  # Pastikan model sudah dilatih sebelumnya
                predictions.append(pred[0])

                # Update input dengan memasukkan prediksi terbaru
                current_input = np.roll(current_input, -1)  # Geser input
                current_input[-1] = pred  # Masukkan prediksi ke input

            # Denormalisasi hasil prediksi
            predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

            # Membuat DataFrame hasil prediksi
            future_dates = pd.date_range(start=data['Tanggal'].max() + pd.Timedelta(days=1), periods=days_to_predict)
            predicted_df = pd.DataFrame({
                'Tanggal': future_dates,
                'CH Prediksi': predictions_rescaled.flatten()
            })

            # Menampilkan hasil
            st.subheader(f"Hasil Prediksi Curah Hujan untuk {days_to_predict} Hari")
            st.dataframe(predicted_df)

            # Menampilkan grafik prediksi
            st.line_chart(predicted_df.set_index('Tanggal')['CH Prediksi'])

            # Menampilkan hasil untuk hari terakhir
            last_prediction = predictions_rescaled[-1][0]
            if last_prediction > 0:
                st.success(f"Prediksi curah hujan pada hari ke-{days_to_predict}: {last_prediction:.2f} mm")
            else:
                st.error(f"Tidak turun hujan pada hari ke-{days_to_predict}")


