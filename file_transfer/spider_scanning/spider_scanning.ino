/*  These are the pins on the arduino. Hook up TOP_LEFT to row 25 on the left side of the breadboard, oriented so that
    the top of the breadboard is row 1) TOP_RIGHT to 25 on the right side, BOTTOM_LEFT to 26 on the left side, and 
    BOTTOM_RIGHT to 26 on the right side. 
    The two pins of the laser go into a separate breadboard.
    Use power cable to power breadboard (long one with bjts) with 9V. Red to + rail (either one) and black to the blue - rail (either one).
    Plug the arduino into computer.
*/

#include <SoftwareSerial.h>
//SoftwareSerial mySerial(2,3);//Define software serial, 3 is TX, 2 is RX
char buff[4]={0x80,0x06,0x03,0x77};
unsigned char data[11]={0};
bool forward = false;

#define TOP_LEFT                   11
#define TOP_RIGHT              5
#define BOTTOM_LEFT               6
#define BOTTOM_RIGHT            9


#define OFF 0
#define ON_FORWARD 255
#define ON_BACKWARD 255



/*  spent stationary before switching direction. When running this overnight, bump this up so that there are 
fewer scans. This also gives the transistors more time to cool down, which will prevent overheating and extend circuit lifespan */
#define TIME_BETWEEN 2500  // 2.5 sec

byte shutdownCommand[] = {0xFA, 0x04, 0x02, 0x00};

byte continuousScanning[] = {0x80, 0x06, 0x03, 0x77};

byte singleRead[] = {0x80, 0x06, 0x02, 0x78};


void setup() {
  
  Serial.begin(115200); // begin serial communication
  //mySerial.begin(9600);

  pinMode(TOP_LEFT, OUTPUT);
  pinMode(TOP_RIGHT, OUTPUT);
  pinMode(BOTTOM_LEFT, OUTPUT);
  pinMode(BOTTOM_RIGHT, OUTPUT);
  // pinMode(LASER, OUTPUT);
  analogWrite(TOP_LEFT, OFF);
  analogWrite(TOP_RIGHT, OFF);
  analogWrite(BOTTOM_LEFT, OFF);
  analogWrite(BOTTOM_RIGHT, OFF);
  // digitalWrite(LASER, HIGH);
  delay(2000); // time when the laser is on but the camera is not moving, setup time and buffer to upload new code before circuit starts running
  //mySerial.write(continuousScanning, sizeof(continuousScanning));
  delay(1000);
}

void go_backward() {
  analogWrite(TOP_RIGHT, OFF);
  analogWrite(BOTTOM_LEFT, OFF);
  delay(20);
  analogWrite(TOP_LEFT, ON_BACKWARD);
  analogWrite(BOTTOM_RIGHT, ON_BACKWARD);
}

void go_forward(int speed) {
  analogWrite(TOP_LEFT, OFF);
  analogWrite(BOTTOM_RIGHT, OFF);
  delay(20);
  analogWrite(TOP_RIGHT, speed);
  analogWrite(BOTTOM_LEFT, speed);
}

void stop_delay() {
  analogWrite(TOP_LEFT, OFF);
  analogWrite(TOP_RIGHT, OFF);
  analogWrite(BOTTOM_LEFT, OFF);
  analogWrite(BOTTOM_RIGHT, OFF);
}

bool stopped = true;
// int loop_counter = 0;
void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();  // Read once and use the value multiple s.
    if (cmd == '1' && !forward) {
      go_forward(ON_FORWARD);
      forward = true;
      stopped = false;
    }
    if (cmd == '2' && forward) {
      go_backward();
      forward = false;
      stopped = false;
    }

    if (cmd == '3' && !stopped) {
      stop_delay();
      stopped = true;
    }

    if (cmd == '4' && !forward) {
      go_forward(240);
      forward = true;
      stopped = false;
    }

  }
}

  

  


