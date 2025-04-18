import network
import time
import urequests
from umqtt.simple import MQTTClient
from hcsr04 import HCSR04
import ujson

# MQTT Server Parameters
MQTT_CLIENT_ID = "gabrielleee"
MQTT_BROKER    = "broker.emqx.io"
MQTT_USER      = ""
MQTT_PASSWORD  = ""
MQTT_TOPIC     = "ivies/ultrasonic/sensor"

# Ubidots API Parameters (STEM Version)
UBIDOTS_TOKEN = "BBUS-KrPBrDs26eWo012B3nYp4OHnIJthqd"
UBIDOTS_URL = "https://industrial.api.ubidots.com/api/v1.6/devices/esp32-sic6/"

# Wifi connection
print("Connecting to WiFi", end="")
sta_if = network.WLAN(network.STA_IF)
sta_if.active(True)
sta_if.connect("herucakra", "87654321")  # Ganti jika pakai ESP32 asli

while not sta_if.isconnected():
    print(".", end="")
    time.sleep(0.5)

print("\nConnected to WiFi!")

# MQTT Connection
print("Connecting to MQTT server... ", end="")
client = MQTTClient(MQTT_CLIENT_ID, MQTT_BROKER, user=MQTT_USER, password=MQTT_PASSWORD, keepalive=60)
client.connect()
print("Connected to MQTT broker!")

# Inisialisasi sensor ultrasonik pertama
sensor1 = HCSR04(trigger_pin=5, echo_pin=18, echo_timeout_us=10000)
# Inisialisasi sensor ultrasonik kedua
sensor2 = HCSR04(trigger_pin=17, echo_pin=16, echo_timeout_us=10000)

# Tinggi tangki (misalnya 12 cm)
TANK_HEIGHT = 30  # cm

def calculate_trash_height(distance):
    """
    Menghitung ketinggian sampah berdasarkan jarak sensor ke permukaan sampah.
    """
    # Ketinggian sampah = tinggi tangki - jarak sensor ke permukaan sampah
    trash_height = TANK_HEIGHT - distance
    # Pastikan ketinggian sampah tidak lebih kecil dari 0 cm
    trash_height = max(trash_height, 0)
    return trash_height

def calculate_trash_percentage(trash_height):
    """
    Menghitung ketinggian sampah dalam bentuk persentase.
    """
    # Ketinggian sampah dalam persen
    trash_percentage = (trash_height / TANK_HEIGHT) * 100
    # Pastikan persentase berada antara 0% dan 100%
    trash_percentage = max(0, min(100, trash_percentage))
    return trash_percentage

def send_to_ubidots(distance1, trash_height1, trash_percentage1, distance2, trash_height2, trash_percentage2):
    payload = {
        "Distance1": {"value": distance1},  # Mengirimkan jarak sensor 1
        "Trash_Height1": {"value": trash_height1},  # Mengirimkan ketinggian sampah sensor 1 (cm)
        "Trash_Percentage1": {"value": trash_percentage1},  # Mengirimkan ketinggian sampah sensor 1 (persen)
        "Distance2": {"value": distance2},  # Mengirimkan jarak sensor 2
        "Trash_Height2": {"value": trash_height2},  # Mengirimkan ketinggian sampah sensor 2 (cm)
        "Trash_Percentage2": {"value": trash_percentage2}  # Mengirimkan ketinggian sampah sensor 2 (persen)
    }
    headers = {
        'X-Auth-Token': UBIDOTS_TOKEN,
        'Content-Type': 'application/json'  # Tambahkan Content-Type
    }

    print(f"Sending to Ubidots: {payload}")  # Debug payload
    try:
        response = urequests.post(UBIDOTS_URL, json=payload, headers=headers)
        print(f"Data sent to Ubidots: {response.status_code}")
        print(f"Ubidots Response: {response.status_code}, {response.text}")  # Debug response
        response.close()
    except Exception as e:
        print(f"Failed to send data to Ubidots: {e}")

while True:
    try:
        print("Getting distance from sensor 1...")
        distance1 = sensor1.distance_cm()
        print("Sensor 1 distance:", distance1, "cm")

        print("Getting distance from sensor 2...")
        distance2 = sensor2.distance_cm()
        print("Sensor 2 distance:", distance2, "cm")

        # Hitung ketinggian sampah untuk sensor 1
        trash_height1 = calculate_trash_height(distance1)
        # Hitung ketinggian sampah dalam persen untuk sensor 1
        trash_percentage1 = calculate_trash_percentage(trash_height1)

        # Hitung ketinggian sampah untuk sensor 2
        trash_height2 = calculate_trash_height(distance2)
        # Hitung ketinggian sampah dalam persen untuk sensor 2
        trash_percentage2 = calculate_trash_percentage(trash_height2)

        print(f"Sensor 1 - Trash height: {trash_height1} cm, Trash percentage: {trash_percentage1}%")
        print(f"Sensor 2 - Trash height: {trash_height2} cm, Trash percentage: {trash_percentage2}%")

        # Kirim data ke Ubidots
        print("Publishing to Ubidots...")
        send_to_ubidots(distance1, trash_height1, trash_percentage1, distance2, trash_height2, trash_percentage2)
        print("Data terkirim ke Ubidots!")

        # Kirim data ke MQTT
        print("Publishing to MQTT...")
        data = {
            "sensor1": distance1,
            "sensor2": distance2
        }
        client.publish(MQTT_TOPIC, ujson.dumps(data))
        print("Data terkirim ke MQTT:", data)

    except Exception as e:
        print(f"Error in loop: {e}")

    time.sleep(1)
