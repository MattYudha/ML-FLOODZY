import requests
import pandas as pd
from datetime import datetime, timedelta
import json

def get_petabencana_data(days_ago=7):
    """
    Mengambil data laporan banjir dari PetaBencana.id dari X hari terakhir.
    """
    print(f"Mengambil data arsip banjir dari PetaBencana.id untuk {days_ago} hari terakhir...")
    
    # Tentukan rentang waktu
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_ago)
    
    # Format waktu untuk URL API
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # URL API untuk arsip laporan
    # Kita hanya akan mengambil laporan 'flood' (banjir)
    url = f"https://data.petabencana.id/reports/archive?start={start_str}&end={end_str}&geoformat=geojson&disaster=flood"
    
    try:
        response = requests.get(url, timeout=30)
        # Cek jika request berhasil
        response.raise_for_status() 
        
        print(f"Berhasil mengambil data. Status: {response.status_code}")
        return response.json()
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error terjadi: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Error Koneksi: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Request timeout: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Error request lainnya: {req_err}")
    except json.JSONDecodeError:
        print("Gagal mem-parsing JSON. Mungkin respons API kosong atau tidak valid.")
        
    return None

def map_water_level(depth_cm):
    """
    Memetakan kedalaman air (numerik dalam cm) ke label kategorikal
    sesuai permintaan pengguna.
    """
    if depth_cm <= 0:
        return "no_flood" # Seharusnya tidak ada di data 'flood', tapi sebagai penjaga
    elif depth_cm <= 10:
        return "surface" # Genangan
    elif depth_cm <= 30:
        return "ankle-deep" # Semata kaki
    elif depth_cm <= 70:
        return "knee-deep" # Selutut
    elif depth_cm <= 150:
        return "waist-deep" # Sepinggang
    else:
        return "over-head" # Di atas kepala / > 150 cm

def process_data_to_csv(raw_data):
    """
    Memproses data GeoJSON mentah menjadi format CSV yang diinginkan.
    """
    if not raw_data or 'features' not in raw_data or not raw_data['features']:
        print("Tidak ada data 'features' yang ditemukan dalam respons. Mungkin tidak ada laporan banjir.")
        return

    processed_list = []
    
    for feature in raw_data['features']:
        try:
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            
            # Pastikan ini adalah laporan banjir dan memiliki koordinat
            if properties.get('report_type') == 'flood' and geometry.get('type') == 'Point':
                coordinates = geometry.get('coordinates')
                if coordinates and len(coordinates) == 2:
                    
                    # Ambil data kedalaman banjir (dalam cm)
                    # default ke 0 jika tidak ada 'flood_depth'
                    flood_depth = properties.get('flood_depth', 0)
                    
                    # Buat data terstruktur
                    data_point = {
                        'latitude': coordinates[1],
                        'longitude': coordinates[0],
                        'water_level': map_water_level(flood_depth),
                        'flood_event': 1, # Karena API ini hanya mengembalikan laporan banjir, event = 1
                        'observed_at': properties.get('created_at'),
                        'flood_depth_cm': flood_depth # Simpan data asli untuk referensi
                    }
                    processed_list.append(data_point)
                    
        except Exception as e:
            print(f"Gagal memproses satu fitur: {e}")
            
    if not processed_list:
        print("Tidak ada laporan banjir yang valid ditemukan dalam data.")
        return

    # Ini adalah baris yang error di skrip Anda (sebelumnya kosong)
    processed_data = pd.DataFrame(processed_list)
    
    # Pilih dan urutkan kolom sesuai permintaan
    final_columns = ['latitude', 'longitude', 'water_level', 'flood_event', 'flood_depth_cm', 'observed_at']
    # Filter agar hanya kolom yang ada di DataFrame
    final_columns_existing = [col for col in final_columns if col in processed_data.columns]
    final_df = processed_data[final_columns_existing]
    
    # Simpan ke CSV
    output_filename = 'flood_data_indonesia.csv'
    final_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    print(f"\nBerhasil! Dataset telah disimpan ke: {output_filename}")
    print(f"Total {len(final_df)} data laporan banjir diproses.")
    print("\nContoh data (5 baris pertama):")
    print(final_df.head())

# --- Fungsi Utama untuk Menjalankan Skrip ---
if __name__ == "__main__":
    # Cek dependensi
    try:
        import requests
        import pandas as pd
    except ImportError:
        print("Error: Modul 'requests' atau 'pandas' tidak ditemukan.")
        print("Silakan install dengan menjalankan:")
        print("pip install requests pandas")
        exit(1)
        
    # 1. Ambil data
    data = get_petabencana_data(days_ago=7) # Ambil data 7 hari terakhir
    
    # 2. Proses dan simpan data
    if data:
        process_data_to_csv(data)
