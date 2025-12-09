#include <Servo.h>

Servo blueServo, redServo, greenServo, blackServo;

// PINS (3, 5, 9, 6)
const int BLUE_PIN = 3;
const int RED_PIN = 5;
const int GREEN_PIN = 6;
const int BLACK_PIN = 9;

void setup() {
  Serial.begin(9600);
  blueServo.attach(BLUE_PIN);
  redServo.attach(RED_PIN);
  greenServo.attach(GREEN_PIN);
  blackServo.attach(BLACK_PIN);

  blueServo.write(0);
  redServo.write(0);
  greenServo.write(0);
  blackServo.write(0);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'B') {
       blueServo.write(90);       
       delay(500);               // Shorter delay (0.5s) to help speed
       blueServo.write(0);        
       delay(500);               
    }
    else if (command == 'R') {
       redServo.write(90);
       delay(500);
       redServo.write(0);
       delay(500);
    }
    else if (command == 'G') {
       greenServo.write(90);
       delay(500);
       greenServo.write(0);
       delay(500);
    }
    else if (command == 'K') {
       blackServo.write(90);
       delay(500);
       blackServo.write(0);
       delay(500);
    }

    // --- CRITICAL FIX: DELETE THE QUEUE ---
    // This empties the buffer so it doesn't remember old commands
    while(Serial.available() > 0) {
      char trash = Serial.read();
    }
    // --------------------------------------
  }
}