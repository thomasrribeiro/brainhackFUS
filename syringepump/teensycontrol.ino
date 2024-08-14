#define PULSE_PIN 3
#define DIR_PIN 2

const int STEPS_PER_REVOLUTION = 6400;
const float LEAD_SCREW_PITCH_MM = 1.0;

const float SYRINGE_DIAMETER_MM = 9.0;
const float TUBE_DIAMETER_MM = 1.0; // Change this to match your tube

float FLUID_VELOCITY_MM_PER_SEC = 200; // Desired fluid velocity in mm/s

float steps_per_second;
float plunger_velocity_mm_per_sec;

const unsigned long DIRECTION_SWITCH_INTERVAL = 5000; // 5 seconds in milliseconds
unsigned long lastDirectionSwitchTime = 0;
bool isForward = true;

void setup() {
  pinMode(PULSE_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  float area_ratio = pow(SYRINGE_DIAMETER_MM / TUBE_DIAMETER_MM, 2);
  
  plunger_velocity_mm_per_sec = FLUID_VELOCITY_MM_PER_SEC / area_ratio;
  
  steps_per_second = plunger_velocity_mm_per_sec * (STEPS_PER_REVOLUTION / LEAD_SCREW_PITCH_MM);
  
  digitalWrite(DIR_PIN, HIGH); // Initial direction
}

void loop() {
  unsigned long currentTime = millis();
  
  // Check if it's time to switch direction
  if (currentTime - lastDirectionSwitchTime >= DIRECTION_SWITCH_INTERVAL) {
    isForward = !isForward; // Toggle direction
    digitalWrite(DIR_PIN, isForward ? HIGH : LOW);
    lastDirectionSwitchTime = currentTime;
  }

  long step_delay_micros = (1000000 / steps_per_second) / 2; // Divide by 2 for high and low pulse
  
  digitalWrite(PULSE_PIN, HIGH);
  delayMicroseconds(step_delay_micros);
  digitalWrite(PULSE_PIN, LOW);
  delayMicroseconds(step_delay_micros);
}
