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
if (selected == 'Modelling'):
    st.title("Model Terbaik")
    
    # Memuat data lagged yang sudah disimpan
    lagged_data = pd.read_excel("lagged_data.xlsx")

    # Memisahkan fitur dan target
    X = lagged_data.drop(columns=["Target"])
    y = lagged_data["Target"]

    # Membagi data menjadi data latih (80%) dan data uji (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menentukan parameter pengujian
    n_neurons_options = range(1, 11)  # Jumlah neuron per hidden layer dari 1 hingga 10
    n_hidden_layers = 5  # Hidden layers terbaik dari pengujian sebelumnya
    learning_rate = 1.0  # Learning rate terbaik dari pengujian sebelumnya
    epochs = 100
    k_folds = 3

    # Menyimpan hasil evaluasi untuk setiap jumlah neuron per hidden layer
    results = {}
    best_mape = float('inf')  # Menyimpan MAPE terbaik
    best_mse = float('inf')  # Menyimpan MSE terbaik
    best_rmse = float('inf')  # Menyimpan RMSE terbaik
    best_model = None
    best_neurons = None

    # Cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for n_neurons in n_neurons_options:
        print(f"\nEvaluating for {n_neurons} neurons per layer")
        mape_scores, mse_scores, rmse_scores = [], [], []

        for train_index, val_index in kf.split(X_train):
            # Membagi data latih dan data validasi untuk fold ini
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

            # Definisikan model backpropagation
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(X_fold_train.shape[1],)))

            # Menambahkan 3 hidden layers dengan jumlah neuron yang bervariasi
            for _ in range(n_hidden_layers):
                model.add(tf.keras.layers.Dense(n_neurons, activation='sigmoid'))

            # Output layer
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            # Kompilasi model dengan optimizer dan learning rate yang tetap
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='mse', metrics=['mse'])

            # Melatih model
            model.fit(X_fold_train, y_fold_train, epochs=epochs, verbose=0)

            # Evaluasi pada data validasi
            y_pred_val = model.predict(X_fold_val)
            mape_scores.append(mean_absolute_percentage_error(y_fold_val, y_pred_val))
            mse_scores.append(mean_squared_error(y_fold_val, y_pred_val))
            rmse_scores.append(np.sqrt(mean_squared_error(y_fold_val, y_pred_val)))

        # Menyimpan hasil rata-rata evaluasi untuk setiap jumlah neuron
        avg_mape = np.mean(mape_scores)
        avg_mse = np.mean(mse_scores)
        avg_rmse = np.mean(rmse_scores)

        results[n_neurons] = {
            'MAPE': avg_mape,
            'MSE': avg_mse,
            'RMSE': avg_rmse
        }
        # print(f"Neurons per layer: {n_neurons} - MAPE: {avg_mape}, MSE: {avg_mse}, RMSE: {avg_rmse}")

        # Simpan model terbaik berdasarkan nilai MAPE
        if avg_mape < best_mape:
            best_mape = avg_mape
            best_mse = avg_mse
            best_rmse = avg_rmse
            best_model = model
            best_neurons = n_neurons

    # Menyimpan model terbaik dengan pickle
    with open("best_model_neurons.pkl", "wb") as f:
        pickle.dump(best_model, f)
        

    st.write(f"\nModel terbaik yang diperoleh :")
    st.write(f"Learning rate : {learning_rate}")
    st.write(f"Hidden layer : {n_hidden_layers}")
    st.write(f"Hidden Neuron : {best_neurons}")
    st.write(f"MAPE : {best_mape}")
    st.write(f"MSE : {best_mse}")
    st.write(f"RMSE : {best_rmse}")

    # Visualisasi hasil pengujian jumlah neuron dalam bentuk grafik
    neurons_list = list(results.keys())
    mape_values = [metrics['MAPE'] for metrics in results.values()]
    mse_values = [metrics['MSE'] for metrics in results.values()]
    rmse_values = [metrics['RMSE'] for metrics in results.values()]

    # Plot untuk MAPE
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(neurons_list, mape_values, label='MAPE', marker='o')
    ax.set_xlabel("Jumlah Neuron per Hidden Layer")
    ax.set_ylabel("MAPE")
    ax.set_title("Pengaruh Jumlah Neuron per Hidden Layer terhadap MAPE")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Plot untuk MSE
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(neurons_list, mse_values, label='MSE', marker='o', color='orange')
    ax.set_xlabel("Jumlah Neuron per Hidden Layer")
    ax.set_ylabel("MSE")
    ax.set_title("Pengaruh Jumlah Neuron per Hidden Layer terhadap MSE")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Plot untuk RMSE
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(neurons_list, rmse_values, label='RMSE', marker='o', color='green')
    ax.set_xlabel("Jumlah Neuron per Hidden Layer")
    ax.set_ylabel("RMSE")
    ax.set_title("Pengaruh Jumlah Neuron per Hidden Layer terhadap RMSE")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

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


