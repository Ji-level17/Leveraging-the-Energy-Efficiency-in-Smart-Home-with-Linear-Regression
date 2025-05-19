#include "WiFi.h"
#include <DHT.h>
#include <ThingSpeak.h>
#include <BH1750.h>

#define CHANNEL_ID
#define CHANNEL_API_KEY ""

BH1750 lightMeter;

WiFiClient client;
int counter = 0;

// DHT my_sensor(5,DHT22);

// 替换为你的WiFi信息
#define WIFI_NETWORK ""
#define WIFI_PASSWORD ""
#define WIFI_TIMEOUT_MS 20000

void connectToWiFi(){
  Serial.print("connecting to WiFi");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_NETWORK,WIFI_PASSWORD);

  unsigned long startAttemptTime = millis();

  while(WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < WIFI_TIMEOUT_MS){
    Serial.print(".");
    delay(100);
  }

  if(WiFi.status() != WL_CONNECTED){
    Serial.println("Failed!");
    // take action
  }else{
    Serial.print("Connected!");
    Serial.println(WiFi.localIP());
  }
}


// DHT22配置
#define DHTPIN 4      // GPIO4连接DHT22数据线
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  Wire.begin(18, 23); // SDA=GPIO18, SCL=GPIO23
  lightMeter.begin();
  dht.begin();
  connectToWiFi();
  ThingSpeak.begin(client);
}

void loop() {
  float lux = lightMeter.readLightLevel();
  // 读取温湿度数据
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature(); // 默认单位为摄氏度

  //检查数据是否有效
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("DHT22 failed");
    return;
  }

  ThingSpeak.setField(4, humidity);
  ThingSpeak.setField(5, lux);
  ThingSpeak.setField(6, temperature);
  ThingSpeak.writeFields(CHANNEL_ID, CHANNEL_API_KEY);

  delay(15000); // ThingSpeak限制15秒/次更新
  // Serial.println(counter);
  // delay(500);
}