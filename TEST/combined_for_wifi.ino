#include <WiFi.h>
#include <DHT.h>
#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);
const int DO_PIN 27
const int sensorPin = 34; 
int sensorValue = 0; 
int moisturePercent = 0;
int dryValue = 3500;
int wetValue = 1500;
const char* ssid = "catastrophr";
const char* password = "catastrophr98";

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("ESP32 STARTED");
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  dht.begin();
  pinMode(DO_PIN, INPUT);
  analogSetPinAttenuation(sensorPin, ADC_11db);

}

void loop() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read from DHT22!");
  } else {
    Serial.print("Temperature: ");
    Serial.print(temperature);
    Serial.println(" °C");

    Serial.print("Humidity: ");
    Serial.print(humidity);
    Serial.println(" %");

    Serial.println("--------------------");
  }

  delay(2000);

   sensorValue = analogRead(sensorPin);
  moisturePercent = map(sensorValue, dryValue, wetValue, 0, 100);
  moisturePercent = constrain(moisturePercent, 0, 100);
  int digitalValue = digitalRead(DO_PIN);
  
  if (digitalValue == 1) {
     Serial.print("wet");
  }
  else {serial.print("dry");}

  Serial.print("Raw ADC Value: ");
  Serial.print(sensorValue);
  Serial.print(" | Soil Moisture: ");
  Serial.print(moisturePercent);
  Serial.println("%");
  
  delay(1000);


}