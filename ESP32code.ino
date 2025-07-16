#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <DHT_U.h>
#include <Wire.h>
#include <BH1750.h>
#include <ArduinoJson.h> // For JSON creation
#include <OneWire.h>     // For DS18B20
#include <DallasTemperature.h> // For DS18B20

// --- WiFi Credentials ---
const char* ssid = "PaulResidency";         // <<< CHANGE THIS
const char* password = "IAMFREENOW"; // <<< CHANGE THIS

// --- Backend Server Configuration ---
const char* serverUrl = "https://smart-agritech-streamlit-dashbard.onrender.com/api/sensor_data"; // <<< CHANGE THIS (e.g., http://192.168.1.100:5000/api/sensor_data or your deployed URL)

// --- Sensor Pin Definitions ---
#define SOIL_MOISTURE_PIN 34 // ESP32 ADC1_CH6
#define DHT_PIN 27           // DHT22 Data pin (connect to GPIO27)
#define DHT_TYPE DHT22       // <<< THIS LINE WAS LIKELY MISSING OR MISPLACED
#define RAIN_SENSOR_PIN 35   // YL-83 Digital Out (DO) pin (connect to GPIO35)
#define ONE_WIRE_BUS 4       // DS18B20 Data pin (connect to GPIO4, or any other digital pin)

// --- Sensor Objects ---
DHT_Unified dht(DHT_PIN, DHT_TYPE);
BH1750 bh1750;

// Setup a oneWire instance to communicate with any OneWire devices (not just DS18B20)
OneWire oneWire(ONE_WIRE_BUS);

// Pass our oneWire reference to Dallas Temperature sensor
DallasTemperature sensors(&oneWire);

// --- Calibration for Soil Moisture Sensor ---
// These values might need adjustment based on your specific sensor and soil type.
// Read sensor in air and in water to find min/max analog values.
const int AIR_VALUE = 3200;  // Analog reading when sensor is in dry air (approx)
const int WATER_VALUE = 1200; // Analog reading when sensor is in water (approx)

void setup() {
  Serial.begin(115200);

  // Initialize DHT sensor.
  dht.begin();
  sensor_t sensor;
  dht.temperature().getSensor(&sensor);
  Serial.print("Temperature Sensor (DHT): "); Serial.println(sensor.name);
  dht.humidity().getSensor(&sensor);
  Serial.print("Humidity Sensor (DHT): "); Serial.println(sensor.name);

  // Initialize BH1750
  Wire.begin();
  if (bh1750.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
    Serial.println("BH1750 sensor initialized");
  } else {
    Serial.println("Error initializing BH1750");
  }

  // Pin for rain sensor
  pinMode(RAIN_SENSOR_PIN, INPUT);

  // Initialize DS18B20
  sensors.begin();
  Serial.print("DS18B20 Devices found: ");
  Serial.println(sensors.getDeviceCount());
  if (sensors.getDeviceCount() == 0) {
    Serial.println("No DS18B20 devices found on OneWire bus!");
  }

  // Connect to WiFi
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500);
    Serial.print(".");
    retries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to WiFi. Please check credentials and try again.");
    Serial.println("Restarting ESP32 in 5 seconds...");
    delay(5000);
    ESP.restart(); // Restart if WiFi connection fails
  }
}

void loop() {
  // --- Read Soil Moisture ---
  int soilMoistureAnalog = analogRead(SOIL_MOISTURE_PIN);
  // Convert analog value to percentage
  float soilMoisturePercent = map(soilMoistureAnalog, AIR_VALUE, WATER_VALUE, 0, 100);
  soilMoisturePercent = constrain(soilMoisturePercent, 0, 100); // Ensure it's between 0 and 100%
  Serial.print("Soil Moisture (Analog): "); Serial.print(soilMoistureAnalog);
  Serial.print(" Soil Moisture (%): "); Serial.println(soilMoisturePercent);

  // --- Read Temperature (DS18B20) ---
  sensors.requestTemperatures(); // Send the command to get temperatures
  float temperature = sensors.getTempCByIndex(0); // Get temperature from the first sensor found
  if (temperature == DEVICE_DISCONNECTED_C) {
    Serial.println("Error reading DS18B20 temperature!");
    temperature = -999.0; // Indicate error
  } else {
    Serial.print("Temperature (DS18B20): "); Serial.print(temperature); Serial.println(" *C");
  }

  // --- Read Humidity (DHT22) ---
  sensors_event_t event;
  dht.humidity().getEvent(&event);
  float humidity = event.relative_humidity;
  if (isnan(humidity)) {
    Serial.println("Error reading DHT22 humidity!");
    humidity = -999.0; // Indicate error
  } else {
    Serial.print("Humidity (DHT22): "); Serial.print(humidity); Serial.println(" %");
  }

  // --- Read Light Intensity (BH1750) ---
  float lux = bh1750.readLightLevel();
  if (lux < 0) { // BH1750 returns -1 if no measurement
    Serial.println("Error reading BH1750 light sensor!");
    lux = 0.0; // Default to 0 on error
  } else {
    Serial.print("Light: "); Serial.print(lux); Serial.println(" lux");
  }

  // --- Read Rain Sensor (YL-83) ---
  // The YL-83 has a digital output (DO) that goes HIGH when it detects rain (default sensitivity).
  // Some versions might be active LOW, check your module.
  int rainDigital = digitalRead(RAIN_SENSOR_PIN);
  float rainfall = 0.0;
  if (rainDigital == LOW) { // Assuming LOW indicates rain, adjust if your module is HIGH
    rainfall = 1.0; // Simple indicator: 1.0 for rain, 0.0 for no rain
    Serial.println("Rain detected!");
  } else {
    Serial.println("No rain detected.");
  }

  // --- Send Data to Flask Backend ---
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    // Create JSON payload
    StaticJsonDocument<256> doc; // Adjust size if needed
    doc["temperature"] = temperature; // Now from DS18B20
    doc["humidity"] = humidity;       // Still from DHT22
    doc["soil_moisture"] = soilMoisturePercent;
    doc["light_intensity"] = lux;
    doc["rainfall"] = rainfall; // Sending rainfall data

    String httpRequestData;
    serializeJson(doc, httpRequestData);

    Serial.print("Sending JSON: ");
    Serial.println(httpRequestData);

    int httpResponseCode = http.POST(httpRequestData);

    if (httpResponseCode > 0) {
      Serial.print("HTTP Response code: ");
      Serial.println(httpResponseCode);
      String response = http.getString();
      Serial.println(response);
    } else {
      Serial.print("Error code: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  } else {
    Serial.println("WiFi Disconnected. Reconnecting...");
    WiFi.reconnect();
  }

  delay(30000); // Send data every 30 seconds
}
