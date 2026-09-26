#include <M5CoreS3.h>
#include <math.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>

// USB-only motion acquisition. No classifier, microphone, or network.
struct Packet {
  char line[220];
  int length;
  uint32_t at, sequence;
  bool valid;
  float ax, ay, az, gx, gy, gz;
};
QueueHandle_t packets = nullptr;
bool taskReady = false, haveSample = false;
Packet latest = {};
uint32_t lastDisplay = 0;
portMUX_TYPE dropMux = portMUX_INITIALIZER_UNLOCKED;
uint32_t transportDrops = 0;
uint32_t dropCount(bool increment = false) {
  portENTER_CRITICAL(&dropMux);
  if (increment) ++transportDrops;
  const uint32_t count = transportDrops;
  portEXIT_CRITICAL(&dropMux);
  return count;
}
void sampleMotion(void*);

void setup() {
  auto config = M5.config();
  config.internal_imu = true;
  config.internal_mic = false;
  config.internal_spk = false;
  CoreS3.begin(config);
  Serial.begin(115200);
  CoreS3.Display.setRotation(1);
  CoreS3.Display.fillScreen(BLACK);
  CoreS3.Display.setTextColor(CYAN, BLACK);
  CoreS3.Display.setTextSize(2);
  CoreS3.Display.setCursor(8, 8);
  CoreS3.Display.println("TinyML RECORDER v0.2");
  CoreS3.Display.setTextSize(1);
  CoreS3.Display.setCursor(8, 40);
  packets = xQueueCreate(32, sizeof(Packet));
  if (packets) {
    // Higher priority than Arduino loop; acquisition preempts screen drawing.
    taskReady = xTaskCreatePinnedToCore(sampleMotion, "motion50Hz", 4096,
                                       nullptr, 2, nullptr, 1) == pdPASS;
    if (!taskReady) { vQueueDelete(packets); packets = nullptr; }
  }
  if (!taskReady) CoreS3.Display.print("ERROR: recorder task failed. Restart.");
}

void sampleMotion(void*) {
  uint32_t sequence = 0, accelAt = 0, gyroAt = 0;
  bool haveAccel = false, haveGyro = false, valid = false;
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
  TickType_t anchor = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(20);
  for (;;) {
    const uint32_t now = micros();
    ++sequence;
    const auto changed = CoreS3.Imu.update();
    const auto data = CoreS3.Imu.getImuData();
    if (changed & m5::IMU_Class::sensor_mask_accel) {
      ax = data.accel.x; ay = data.accel.y; az = data.accel.z;
      accelAt = now; haveAccel = true;
    }
    if (changed & m5::IMU_Class::sensor_mask_gyro) {
      gx = data.gyro.x; gy = data.gyro.y; gz = data.gyro.z;
      gyroAt = now; haveGyro = true;
    }
    // Both streams must have updated for this row: never label cached data fresh.
    valid = CoreS3.Imu.isEnabled() &&
      (changed & m5::IMU_Class::sensor_mask_accel) &&
      (changed & m5::IMU_Class::sensor_mask_gyro) &&
      isfinite(ax) && isfinite(ay) && isfinite(az) &&
      isfinite(gx) && isfinite(gy) && isfinite(gz);
    Packet packet = {};
    const int n = snprintf(packet.line, sizeof(packet.line),
      "M5IMU1,%lu,%lu,%d,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%ld,%ld,%lu\n",
      (unsigned long)sequence, (unsigned long)now, valid,
      ax, ay, az, gx, gy, gz,
      haveAccel ? (long)(now - accelAt) : -1L,
      haveGyro ? (long)(now - gyroAt) : -1L, (unsigned long)dropCount());
    packet.length = n;
    packet.at = now; packet.sequence = sequence; packet.valid = valid;
    packet.ax = ax; packet.ay = ay; packet.az = az;
    packet.gx = gx; packet.gy = gy; packet.gz = gz;
    // No USB or display access in this task; queue overflow is an explicit gap.
    if (n <= 0 || n >= int(sizeof(packet.line)) ||
        xQueueSend(packets, &packet, 0) != pdTRUE) dropCount(true);
    // Skip missed deadlines instead of manufacturing rapid catch-up samples.
    if (xTaskGetTickCount() - anchor >= period) anchor = xTaskGetTickCount();
    vTaskDelayUntil(&anchor, period);
  }
}

void loop() {
  if (!taskReady) { delay(100); return; }
  // Touch/buttons are unused. Avoid CoreS3.update() so only the sampler
  // accesses the internal sensor I2C bus. Queue copies avoid shared-state races.
  Packet packet;
  for (int i = 0; i < 32 && xQueueReceive(packets, &packet, 0) == pdTRUE; ++i) {
    latest = packet;
    haveSample = true;
    if (Serial) {
      // USB may wait here, but cannot block acquisition. The bounded queue
      // makes any prolonged stall visible in subsequent sequence numbers.
      if (Serial.write(reinterpret_cast<const uint8_t*>(packet.line), packet.length)
          != size_t(packet.length)) dropCount(true);
    } else dropCount(true);
  }
  if (millis() - lastDisplay >= 500) {
    lastDisplay = millis();
    CoreS3.Display.fillRect(0, 40, 320, 160, BLACK);
    CoreS3.Display.setCursor(8, 44);
    const bool healthy = haveSample && latest.valid && micros() - latest.at < 250000;
    CoreS3.Display.setTextColor(healthy ? GREEN : YELLOW, BLACK);
    CoreS3.Display.printf("IMU: %s\n", healthy ? "OK" : "WAIT / MISSING");
    CoreS3.Display.setTextColor(WHITE, BLACK);
    CoreS3.Display.printf("\nAccel g: %.2f %.2f %.2f\n", latest.ax, latest.ay, latest.az);
    CoreS3.Display.printf("Gyro deg/s: %.1f %.1f %.1f\n", latest.gx, latest.gy, latest.gz);
    CoreS3.Display.printf("\nSequence: %lu  Stream skips: %lu\n", (unsigned long)latest.sequence, (unsigned long)dropCount());
    CoreS3.Display.println("\n50 Hz task / buffered USB\nStart labeled capture on your laptop.\nNot a trained model yet.");
  }
  delay(1);
}
