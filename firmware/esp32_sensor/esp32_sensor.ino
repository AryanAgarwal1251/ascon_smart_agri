// ESP32 sensor node -- the sensor contract of ascon_smart_agri (Phase 8, plan §2).
//
// Publishes one JSON reading every INTERVAL_MS to  farm/<FARM>/sensor/<DEVICE>  and listens on
// farm/<FARM>/scenario  so a demonstration can put this node into an "attack" scenario. The
// scenario changes NOTHING about what this node sends; it tells the Pi's provenance adapter
// which held-out record pool to pair this device's readings with (plan §8 option A). The Pi
// runtime cannot tell this sketch apart from scripts/run_virtual_sensor.py -- by design.
//
// Wokwi: open this folder's diagram.json (ESP32 + DHT22 on GPIO 15 + potentiometer on GPIO 34
// as the soil-moisture probe). Wokwi's free public gateway reaches the internet but not your
// LAN, so the default broker is a public one; on the hardware set BROKER to the Pi's address.
// Libraries (Arduino Library Manager / Wokwi libraries.txt): PubSubClient, DHT sensor library,
// ArduinoJson.
//
// Demo credentials only. Nothing here is a production sensor node: no TLS to the broker
// (the farm network is the trust boundary; the Pi -> cloud hop is where TLS lives), no
// provisioning, no OTA.

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "DHT.h"

// ---- configuration -------------------------------------------------------------------
const char* WIFI_SSID = "Wokwi-GUEST";        // hardware: the farm Wi-Fi
const char* WIFI_PASS = "";
const char* BROKER    = "test.mosquitto.org";  // hardware: the Pi's IP (farm-local broker)
const int   BROKER_PORT = 1883;
const char* FARM      = "farm1";
const char* DEVICE    = "soil01";
const unsigned long INTERVAL_MS = 2000;

#define DHT_PIN 15
#define DHT_TYPE DHT22
#define MOISTURE_PIN 34

// ---- state ---------------------------------------------------------------------------
WiFiClient wifi;
PubSubClient mqtt(wifi);
DHT dht(DHT_PIN, DHT_TYPE);
unsigned long seq = 0;              // per-device replay counter carried in every reading
String scenario = "benign";         // informational only; shown on the serial console
char sensorTopic[64], scenarioTopic[64];

void onScenario(char* topic, byte* payload, unsigned int length) {
  String text;
  for (unsigned int i = 0; i < length; i++) text += (char)payload[i];
  // Either "attack"/"benign" for the whole farm, or {"deviceId":"soil01","mode":"attack"}.
  if (text.startsWith("{")) {
    StaticJsonDocument<128> doc;
    if (deserializeJson(doc, text)) return;
    if (doc["deviceId"].isNull() || String((const char*)doc["deviceId"]) == DEVICE)
      scenario = String((const char*)doc["mode"]);
  } else {
    scenario = text;
  }
  Serial.printf("[%s] scenario -> %s (readings unchanged; the Pi selects the record pool)\n",
                DEVICE, scenario.c_str());
}

void connectWifi() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(250); Serial.print("."); }
  Serial.printf("\nWi-Fi up: %s\n", WiFi.localIP().toString().c_str());
}

void connectMqtt() {
  while (!mqtt.connected()) {
    String clientId = String("esp32-") + FARM + "-" + DEVICE;
    if (mqtt.connect(clientId.c_str())) {
      mqtt.subscribe(scenarioTopic);
      Serial.printf("MQTT up: %s, subscribed %s\n", BROKER, scenarioTopic);
    } else {
      delay(1000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  snprintf(sensorTopic, sizeof sensorTopic, "farm/%s/sensor/%s", FARM, DEVICE);
  snprintf(scenarioTopic, sizeof scenarioTopic, "farm/%s/scenario", FARM);
  dht.begin();
  connectWifi();
  mqtt.setServer(BROKER, BROKER_PORT);
  mqtt.setCallback(onScenario);
}

void loop() {
  if (!mqtt.connected()) connectMqtt();
  mqtt.loop();

  static unsigned long last = 0;
  if (millis() - last < INTERVAL_MS) return;
  last = millis();

  float temperature = dht.readTemperature();
  if (isnan(temperature)) temperature = 25.0;                 // DHT read glitch: hold a value
  float moisture = analogRead(MOISTURE_PIN) * 100.0 / 4095.0;  // 0..100 %

  StaticJsonDocument<192> doc;
  doc["deviceId"] = DEVICE;
  doc["temperature"] = temperature;
  doc["soilMoisture"] = moisture;
  doc["seq"] = seq++;
  char body[192];
  serializeJson(doc, body, sizeof body);
  mqtt.publish(sensorTopic, body);
  Serial.printf("%s %s [%s]\n", sensorTopic, body, scenario.c_str());
}
