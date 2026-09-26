#include <M5CoreS3.h>
#include <math.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include "motion_tree.h"

constexpr uint32_t SAMPLE_PERIOD_MS = 20;
constexpr size_t WINDOW = 50;  // 1 second at 50 Hz.
struct Sample { float values[6]; };
QueueHandle_t samples;
volatile uint32_t streamDrops = 0;
Sample history[WINDOW];
size_t historyCount = 0, historyHead = 0;
const char* currentLabel = "WARMING UP";
uint32_t windows = 0, lastDisplay = 0, lastInferenceUs = 0;
uint32_t samplesSincePrediction = 0;

void sampleMotion(void*) {
  TickType_t anchor = xTaskGetTickCount();
  for (;;) {
    const auto changed = CoreS3.Imu.update();
    const auto data = CoreS3.Imu.getImuData();
    if ((changed & m5::IMU_Class::sensor_mask_accel) &&
        (changed & m5::IMU_Class::sensor_mask_gyro)) {
      Sample sample = {{data.accel.x, data.accel.y, data.accel.z,
                        data.gyro.x, data.gyro.y, data.gyro.z}};
      if (xQueueSend(samples, &sample, 0) != pdTRUE) ++streamDrops;
    } else {
      ++streamDrops;
    }
    vTaskDelayUntil(&anchor, pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
  }
}

void addSample(const Sample& sample) {
  history[historyHead] = sample;
  historyHead = (historyHead + 1) % WINDOW;
  if (historyCount < WINDOW) ++historyCount;
  if (historyCount < WINDOW) return;
  // Training used 50-sample windows with a 25-sample step.
  if (++samplesSincePrediction < WINDOW / 2) return;
  samplesSincePrediction = 0;

  float values[WINDOW][8];
  // Reconstruct chronological order from the circular buffer.
  for (size_t i = 0; i < WINDOW; ++i) {
    const Sample& s = history[(historyHead + i) % WINDOW];
    for (size_t axis = 0; axis < 6; ++axis) values[i][axis] = s.values[axis];
    values[i][6] = sqrtf(s.values[0]*s.values[0] + s.values[1]*s.values[1] + s.values[2]*s.values[2]);
    values[i][7] = sqrtf(s.values[3]*s.values[3] + s.values[4]*s.values[4] + s.values[5]*s.values[5]);
  }
  float features[48];
  size_t offset = 0;
  for (size_t stat = 0; stat < 6; ++stat) {
    for (size_t axis = 0; axis < 8; ++axis) {
      float sum = 0, sumSquares = 0, minValue = values[0][axis], maxValue = values[0][axis];
      float differenceSum = 0;
      for (size_t i = 0; i < WINDOW; ++i) {
        const float v = values[i][axis];
        sum += v; sumSquares += v * v;
        if (v < minValue) minValue = v;
        if (v > maxValue) maxValue = v;
        if (i > 0) differenceSum += fabsf(v - values[i-1][axis]);
      }
      const float mean = sum / WINDOW;
      if (stat == 0) features[offset++] = mean;
      else if (stat == 1) {
        float variance = 0;
        for (size_t i = 0; i < WINDOW; ++i) {
          const float d = values[i][axis] - mean; variance += d * d;
        }
        features[offset++] = sqrtf(variance / WINDOW);
      } else if (stat == 2) features[offset++] = minValue;
      else if (stat == 3) features[offset++] = maxValue;
      else if (stat == 4) features[offset++] = sqrtf(sumSquares / WINDOW);
      else features[offset++] = differenceSum / (WINDOW - 1);
    }
  }
  const uint32_t started = micros();
  currentLabel = tinyml_motion::label(tinyml_motion::predict(features));
  lastInferenceUs = micros() - started;
  ++windows;
  if (Serial) Serial.printf("# PREDICTION,%lu,%s,%lu,%lu\n",
    (unsigned long)windows, currentLabel, (unsigned long)lastInferenceUs,
    (unsigned long)streamDrops);
}

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
  CoreS3.Display.println("TINYML MOTION v0.1");
  samples = xQueueCreate(64, sizeof(Sample));
  if (!samples || xTaskCreatePinnedToCore(sampleMotion, "motion50Hz", 4096,
      nullptr, 2, nullptr, 1) != pdPASS) {
    CoreS3.Display.setTextColor(RED, BLACK);
    CoreS3.Display.setCursor(8, 48);
    CoreS3.Display.println("CLASSIFIER START FAILED");
    while (true) delay(1000);
  }
  Serial.println("# TinyML classifier ready; warming up one second");
}

void loop() {
  Sample sample;
  for (int i = 0; i < 64 && xQueueReceive(samples, &sample, 0) == pdTRUE; ++i) addSample(sample);
  if (millis() - lastDisplay >= 500) {
    lastDisplay = millis();
    CoreS3.Display.fillRect(0, 40, 320, 180, BLACK);
    CoreS3.Display.setCursor(8, 48);
    CoreS3.Display.setTextColor(historyCount < WINDOW ? YELLOW : GREEN, BLACK);
    CoreS3.Display.setTextSize(2);
    CoreS3.Display.printf("%s\n", currentLabel);
    CoreS3.Display.setTextSize(1);
    CoreS3.Display.setTextColor(WHITE, BLACK);
    CoreS3.Display.printf("\nWindow: %u / %u samples\n", (unsigned)historyCount, (unsigned)WINDOW);
    CoreS3.Display.printf("Inference: %lu us\n", (unsigned long)lastInferenceUs);
    CoreS3.Display.printf("Predictions: %lu (2 Hz)\n", (unsigned long)windows);
    CoreS3.Display.printf("Stream drops: %lu\n", (unsigned long)streamDrops);
    CoreS3.Display.println("\nMove slowly to rotate.\nMove quickly to shake.");
  }
  delay(1);
}
