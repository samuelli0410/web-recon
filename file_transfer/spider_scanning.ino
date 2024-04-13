/*  These are the pins on the arduino. Hook up TOP_LEFT to row 25 on the left side of the breadboard, oriented so that
    the top of the breadboard is row 1) TOP_RIGHT to 25 on the right side, BOTTOM_LEFT to 26 on the left side, and 
    BOTTOM_RIGHT to 26 on the right side. 
    The two pins of the laser go into breadboard ground and arduino pin 11 (NOT 9V breadboard +).
    Check laser polarity (red wire to + on arduino pin 11, black to gnd).
    Use power cable to power breadboard with 9V.
    Plug the arduino into computer.
*/
#define TOP_LEFT                   3
#define TOP_RIGHT              5
#define BOTTOM_LEFT               6
#define BOTTOM_RIGHT            9
#define LASER                   11

#define OFF 0
#define ON_FORWARD 255 
#define ON_BACKWARD 255


#define TIME 20000 // time per direction, 20000ms at 255 power
#define RATIO 0.91 // which direction is stronger. <1 means forward is stronger, and will make the forward direction less time
                  // Currently this is empirically determined but eventually we should add sensors


/* time spent stationary before switching direction. When running this overnight, bump this up so that there are 
fewer scans. This also gives the transistors more time to cool down, which will prevent overheating and extend circuit lifespan */
#define TIME_BETWEEN 2500  // 2.5 sec

void setup() {
  pinMode(TOP_LEFT, OUTPUT);
  pinMode(TOP_RIGHT, OUTPUT);
  pinMode(BOTTOM_LEFT, OUTPUT);
  pinMode(BOTTOM_RIGHT, OUTPUT);
  pinMode(LASER, OUTPUT);
  analogWrite(TOP_LEFT, OFF);
  analogWrite(TOP_RIGHT, OFF);
  analogWrite(BOTTOM_LEFT, OFF);
  analogWrite(BOTTOM_RIGHT, OFF);
  digitalWrite(LASER, HIGH);
  delay(2000); // time when the laser is on but the camera is not moving, setup time and buffer to upload new code before circuit starts running
}

void go_backward() {
  analogWrite(TOP_LEFT, ON_BACKWARD);
  analogWrite(BOTTOM_RIGHT, ON_BACKWARD);
  delay(TIME * RATIO);
  analogWrite(TOP_LEFT, OFF);
  analogWrite(BOTTOM_RIGHT, OFF);
  delay(TIME_BETWEEN);
}

void go_forward() {
  analogWrite(TOP_RIGHT, ON_FORWARD);
  analogWrite(BOTTOM_LEFT, ON_FORWARD);
  delay(TIME);
  analogWrite(TOP_RIGHT, OFF);
  analogWrite(BOTTOM_LEFT, OFF);
  delay(TIME_BETWEEN);
}
void loop() {
  // feel free to add anything else that the python integration should do and/or control signals
  go_forward();
  go_backward();

  

  

}
