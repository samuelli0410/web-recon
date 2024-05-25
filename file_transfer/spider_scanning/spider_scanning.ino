/*  These are the pins on the arduino. Hook up TOP_LEFT to row 25 on the left side of the breadboard, oriented so that
    the top of the breadboard is row 1) TOP_RIGHT to 25 on the right side, BOTTOM_LEFT to 26 on the left side, and 
    BOTTOM_RIGHT to 26 on the right side. 
    The two pins of the laser go into breadboard ground and arduino pin 11 (NOT 9V breadboard +).
    Check laser polarity (red wire to + on arduino pin 11, black to gnd).
    Use power cable to power breadboard with 9V.
    Plug the arduino into computer.
*/

#include <SoftwareSerial.h>
SoftwareSerial mySerial(2,3);//Define software serial, 3 is TX, 2 is RX
char buff[4]={0x80,0x06,0x03,0x77};
unsigned char data[11]={0};
bool forward = false;

#define TOP_LEFT                   11
#define TOP_RIGHT              5
#define BOTTOM_LEFT               6
#define BOTTOM_RIGHT            9


#define OFF 0
#define ON_FORWARD 230
#define ON_BACKWARD 230


#define TIME 15000 // time per direction, 20000ms at 255 power
#define RATIO 0.91 // which direction is stronger. <1 means forward is stronger, and will make the forward direction less time
                  // Currently this is empirically determined but eventually we should add sensors


/* time spent stationary before switching direction. When running this overnight, bump this up so that there are 
fewer scans. This also gives the transistors more time to cool down, which will prevent overheating and extend circuit lifespan */
#define TIME_BETWEEN 2500  // 2.5 sec




void setup() {
  
  Serial.begin(115200); // begin serial communication
  mySerial.begin(9600);

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
}

void go_backward() {
  analogWrite(TOP_RIGHT, OFF);
  analogWrite(BOTTOM_LEFT, OFF);
  delay(20);
  analogWrite(TOP_LEFT, ON_BACKWARD);
  analogWrite(BOTTOM_RIGHT, ON_BACKWARD);
}

void go_forward() {
  analogWrite(TOP_LEFT, OFF);
  analogWrite(BOTTOM_RIGHT, OFF);
  delay(20);
  analogWrite(TOP_RIGHT, ON_FORWARD);
  analogWrite(BOTTOM_LEFT, ON_FORWARD);
}

void stop_delay() {
  analogWrite(TOP_LEFT, OFF);
  analogWrite(TOP_RIGHT, OFF);
  analogWrite(BOTTOM_LEFT, OFF);
  analogWrite(BOTTOM_RIGHT, OFF);
}

bool stopped = true;

void loop() {
  mySerial.print(buff);

  while (1) {
    if (Serial.available() > 0) {
    char cmd = Serial.read();  // Read once and use the value multiple times.
    if (cmd == '1' && !forward) {
      go_forward();
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
    
  }
  
  
  if(mySerial.available() >= 0)//Determine whether there is data to read on the serial
  {
    delay(50);
    for(int i=0;i<11;i++)
    {
      data[i]=mySerial.read();
    }
    unsigned char Check=0;
    for(int i=0;i<10;i++)
    {
      Check=Check+data[i];
    }
    Check=~Check+1;
    if(data[10]==Check)
    {
      if(data[3]=='E'&&data[4]=='R'&&data[5]=='R')
      {
        Serial.println("Out of range");
      }
      else
      {
        float distance=0;
        distance=(data[3]-0x30)*100+(data[4]-0x30)*10+(data[5]-0x30)*1+(data[7]-0x30)*0.1+(data[8]-0x30)*0.01+(data[9]-0x30)*0.001;
        Serial.println(distance, 3);
      }
    }
    else
    {
      //Serial.println("Invalid Data!");
    }
  }
  delay(20);
  }
  // feel free to add anything else that the python integration should do and/or control signals
  
}

  

  


