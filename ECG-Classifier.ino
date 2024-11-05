#include <LSM6DS3.h>
#include <Wire.h>

#include <Adafruit_TinyUSB.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
//#include <tensorflow/lite/version.h>

#include "c_CNNECG.h"
#define DEBUG 1

#define NUM_READINGS 140
float readings[NUM_READINGS];
int readingIndex = 0;


// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 2 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "Class 1",
  "Class 2",
  "Class 3",
  "Class 4",
  "Class 5"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))
void setup() {
// initialize the serial communication:
Serial.begin(9600);
pinMode(2, INPUT); // Setup for leads off detection LO +
pinMode(3, INPUT); // Setup for leads off detection LO -
 
   // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(c_CNNECG);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}
 
void loop() {
 if (readingIndex < NUM_READINGS) {
if((digitalRead(2) == 1)||(digitalRead(3) == 1)){
Serial.println('!');
}
else{
// send the value of analog input 0:
Serial.println(analogRead(A0));
 int number = analogRead(A0);
  number = number - 512;
  float normalized = number / 512.0; // Convert to float for accurate division
  readings[readingIndex]=normalized;
}
readingIndex++;
 }
  if (readingIndex == NUM_READINGS) {
     tflInputTensor->data.f=readings;
     TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
  }
  for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 6);
        }
        Serial.println();
//Wait for a bit to keep serial data from saturating
delay(200);
}
}