#define PULSE_PIN 3
#define DIR_PIN 2

bool FORWARD;

void setup() {
  pinMode(PULSE_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  FORWARD=true;
}

void loop() {
  if(FORWARD){
    digitalWrite(DIR_PIN, HIGH);
    FORWARD=false;
  }
  else{
    digitalWrite(DIR_PIN, LOW);
    FORWARD=true;
  }

  delay(1000);

  int steps = 10000;
  for(int i=0; i<steps; i++){
    digitalWrite(PULSE_PIN, HIGH);
    delayMicroseconds(2000);
    digitalWrite(PULSE_PIN, LOW);
    delayMicroseconds(2000);
  }
}


