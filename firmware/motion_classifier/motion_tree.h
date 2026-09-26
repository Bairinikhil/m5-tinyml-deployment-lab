#pragma once
// Generated from the clean motion dataset. Features must match the desktop pipeline.
// 48 floats: 8 signals x mean, std, minimum, maximum, RMS, mean absolute difference.
namespace tinyml_motion {
inline int predict(const float* features) {
  if (features[20] <= -0.732450008f) {
    if (features[14] <= 0.0970725417f) return 0;
    return 1;
  }
  if (features[46] <= 0.00221362279f) return 2;
  return 0;
}
inline const char* label(int id) {
  switch (id) {
    case 0: return "ROTATING";
    case 1: return "SHAKING";
    case 2: return "STILL";
    default: return "UNKNOWN";
  }
}
}
