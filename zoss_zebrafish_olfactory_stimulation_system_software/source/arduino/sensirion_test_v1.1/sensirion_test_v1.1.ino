#include <Wire.h> // Arduino library for I2C

const char* END_TRANSMIT_ERRORS[6] = {
  "0: success",
  "1: data too long to fit in transmit buffer",
  "2: received NACK on transmit of address",
  "3: received NACK on transmit of data",
  "4: other error.",
  "5: timeout"};

bool verbose = false;

// -----------------------------------------------------------------------------
// TCA multiplexer (TCA9548A) setup
// -----------------------------------------------------------------------------
//  number of tca i2c multiplexer devices connected to the i2c line
constexpr unsigned NUM_TCA_DEVICES = 2;

// number of ports on each i2c multiplexer device. This code is for the
// TCA9548A, which has 8. This code assumes that each of the devices are
// identical aside from i2c port address.
constexpr unsigned PORTS_PER_TCA = 8;

// computes the maximum number of sensors based on the above
constexpr unsigned MAX_NUM_PORTS = (NUM_TCA_DEVICES * PORTS_PER_TCA);

// i2c port address for the i2c multiplexer (with port unmodified)
// code assumes that the next i2c multiplexer will be on port
// 0x71, then 0x72 and so on. Addresses are modified with physical jumpers.
const int BASE_TCA_ADDR = 0x70;

// binary array for keeping track of the detected senssors
bool active_ports[MAX_NUM_PORTS];
int current_tca_addr = -1;

void deselect_all_tca_ports(bool &comm_check) {
  int tca_addr;
  int device_acknowledge;
  for (int i = 0; i < NUM_TCA_DEVICES; i++) {
    tca_addr = i + BASE_TCA_ADDR;
    Wire.beginTransmission(tca_addr);
    Wire.write(0x00);
    device_acknowledge = Wire.endTransmission();

    if (device_acknowledge != 0) {
      comm_check = true;
    }
  }
}

int safe_select_tca_port(uint8_t port_index, bool &comm_check) {
  int tca_addr;
  int tca_port_index;

  if (port_index > (MAX_NUM_PORTS - 1)) {
    if (verbose) {
      Serial.print("Cannot select port with index ");
      Serial.print(port_index);
      Serial.println("; value is outside of the range of multiplexed ports.");
    }
    return;
  }

  // compute the tca i2c address by integer dividing the port index
  // by the number of ports per TCA multiplexer
  // port_index =  1 -> tca_addr = ( 1 / 8) + 0x70 = 0 + 0x70 = 0x70
  // port_index = 23 -> tca_addr = (23 / 8) + 0x70 = 2 + 0x70 = 0x72
  tca_addr = (port_index / PORTS_PER_TCA) + BASE_TCA_ADDR;
  
  if (current_tca_addr != tca_addr) {
    deselect_all_tca_ports(comm_check);
  }
  current_tca_addr = tca_addr;

  // compute the port index (on the tca multiplexer) by using
  // the remainder of the port index divided by the number of ports per
  // TCA multiplexer
  // port_index =  1 -> tca_port_index =  1 % 8 = 1
  // port_index = 23 -> tca_port_index = 23 % 8 = 7
  tca_port_index = port_index % PORTS_PER_TCA;

  Wire.beginTransmission(tca_addr);
  Wire.write(((uint8_t) 1) << tca_port_index);
  return Wire.endTransmission();
}

// -----------------------------------------------------------------------------
// Sensirion CRC-8 Checksum Implementation
// -----------------------------------------------------------------------------

byte crc8_table[256];
const byte CRC_POLYNOMIAL = 0x31;
const byte CRC_INIT = 0xFF;
const byte LEADING_BIT = ((byte) 1) << 7;

void compute_crc8_lookup_table() {
  // see http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html
  byte x;
  for (int i = 0; i < 256; i++) {
    x = ((byte) i);
    for (int i = 0; i < 8; i++) {
      if ((x & LEADING_BIT) != 0) {
        x = ((byte) ((x << 1) ^ CRC_POLYNOMIAL));
      } else {
        x <<= 1;
      }
    }
    crc8_table[i] = x;
  }
}

byte crc8(uint16_t x) {
  // see http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html
  byte crc = CRC_INIT;
  for (int i = 1; i >= 0; i--) {
    crc ^= ((byte) (x >> (i * 8)));
    crc = crc8_table[crc];
  }
  return crc;
}

// -----------------------------------------------------------------------------
// Sensirion Sensor-Spcific Setup
// -----------------------------------------------------------------------------
// Generall call I2C address (used for resetting)
const int GENERAL_CALL_ADDRESS = 0x00; 
const int SENSOR_ADDRESS = 0x08; // Address for SLF3x Liquid Flow Sensors

const unsigned long SENSOR_RESET_WAIT = 25; // milliseconds to wait after reset
// milliseconds to wait for readings after beginning continuous measurement mode
const unsigned long SENSOR_WARMUP_WAIT = 50; 

// 1300F (+/- 65 mL/min) sensor part number
const uint32_t S1300F_PN = 0x07030201;
// 0600F (+/- 3.25 mL/min) sensor part number
const uint32_t S0600F_PN = 0x07030301;
// in min/mL; scale factor for flow rate measurement
const float S1300F_SCALE_FACTOR_FLOW = 500.0;
// in min/mL (10 (ul/min)^-1); scale factor for flow rate measurement
const float S0600F_SCALE_FACTOR_FLOW = 10000.0;

// number of sensor types supported/annotated ("none" sensor counts as one)
constexpr unsigned NUM_SENSOR_TYPES = 3;
// indexes for refering to the different sensor types
const uint8_t SENSOR_NONE = 0;
const uint8_t SENSOR_S1300F = 1;
const uint8_t SENSOR_S0600F = 2;
const uint32_t SENSIRION_PART_NUMBERS[NUM_SENSOR_TYPES] =
  {0, S1300F_PN, S0600F_PN};
const float FLOW_SCALE_FACTORS[NUM_SENSOR_TYPES] =
  {0.0, S1300F_SCALE_FACTOR_FLOW, S0600F_SCALE_FACTOR_FLOW};
const char* SENSOR_PART_NAMES[NUM_SENSOR_TYPES] = {
  "(no or unknown sensor)",
  "SLF3S-1300F",
  "SLF3S-0600F"};

const float SCALE_FACTOR_TEMP = 200.0; // scale factor for the temp measurement
const char UNIT_FLOW[] = "mL/min"; //physical unit of the flow rate measurement
const char UNIT_TEMP[] = "C"; // physical unit of the temperature measurement

int sensirion_soft_reset() {
  // send 0x06 to the general call address to reset the sensor
  // see datasheet section "Soft Reset" for more information
  Wire.beginTransmission(GENERAL_CALL_ADDRESS);
  Wire.write(0x06); // reset command code
  return Wire.endTransmission();
}

uint32_t sensirion_read_product_number(bool &comm_check) {
  uint32_t product_number;
  unsigned expected_num_bytes;
  int device_acknowledge;

  // send command code 0x367C followed by 0xE102 to get the sensor's product
  // number and serial number see datasheet section "Read Product Indentifier
  // and Serial Number" for more information.
  Wire.beginTransmission(SENSOR_ADDRESS);
  Wire.write(0x36);
  Wire.write(0x7C);
  device_acknowledge = Wire.endTransmission(false);

  if (device_acknowledge != 0) {
    comm_check = true;
    return 0;
  }

  Wire.beginTransmission(SENSOR_ADDRESS);
  Wire.write(0xE1);
  Wire.write(0x02);
  device_acknowledge = Wire.endTransmission(false);

  if (device_acknowledge != 0) {
    comm_check = true;
    return 0;
  }

  expected_num_bytes = 6;
  Wire.requestFrom(SENSOR_ADDRESS, expected_num_bytes);

  if (Wire.available() < expected_num_bytes) {
    comm_check = true;
    return 0;
  }

  product_number = ((uint32_t) sensirion_read_int16(comm_check)) << 16;
  product_number |= ((uint32_t) sensirion_read_int16(comm_check));
  return product_number;
}

int16_t sensirion_read_int16(bool &comm_check) {
  int16_t output_value;
  byte ms_byte;
  byte ls_byte;
  byte crc;

  ms_byte = Wire.read(); // read most significant byte
  ls_byte = Wire.read(); // read least significant byte
  crc = Wire.read(); // read crc cheksum byte

  output_value = ((int16_t) ms_byte) << 8;
  output_value |= ((int16_t) ls_byte);

  comm_check = comm_check || (crc8((uint16_t) output_value) != crc);

  return output_value;
}

void sensirion_read_measurement_flags(
    bool &air_in_line_flag,
    bool &high_flow_flag,
    bool &exp_smooth_flag,
    bool &comm_check) {

  uint16_t raw_value;
  byte ms_byte;
  byte ls_byte;
  byte crc;

  static const uint16_t bit_0 = 1;
  static const uint16_t bit_1 = ((uint16_t) 1) << 1;
  static const uint16_t bit_5 = ((uint16_t) 1) << 5;

  ms_byte = Wire.read(); // read most significant byte
  ls_byte = Wire.read(); // read least significant byte
  crc = Wire.read(); // read crc cheksum byte

  raw_value = ((uint16_t) ms_byte) << 8;
  raw_value |= ((uint16_t) ls_byte);

  comm_check = comm_check || (crc8(raw_value) != crc);

  air_in_line_flag = (raw_value & bit_0) > 0;
  high_flow_flag = (raw_value & bit_1) > 0;
  exp_smooth_flag = (raw_value & bit_5) > 0;
}

int sensirion_begin_continuous_measure() {
  // Send 0x3608 to switch to continuous measurement mode (H20 calibration).
  // Check datasheet section "Start Continuout Measurement" for more 
  // information.
  Wire.beginTransmission(SENSOR_ADDRESS);
  Wire.write(0x36);
  Wire.write(0x08);
  return Wire.endTransmission();
}

uint8_t sensors[MAX_NUM_PORTS];

// -----------------------------------------------------------------------------
// Arduino setup routine, just runs once:
// -----------------------------------------------------------------------------

const unsigned long BAUD_RATE = 460800;

void setup() {
  int device_acknowledge;
  // `comm_check` is false for OK I2C communication, and true for failed communication
  bool comm_check = false;
  uint32_t sensor_product_number;

  Serial.begin(BAUD_RATE); // initialize serial communication
  if (verbose) {
    Serial.println("Beginning sensor detection and identification.");
  }

  // initialize crc lookup table
  if (verbose) {
    Serial.println("Initializing crc8 lookup table.");
  }
  compute_crc8_lookup_table();

  Wire.begin(); // join i2c bus (address optional for master)

  // Try to reset the sensor via each of the TCA(s)'s
  // ports to check for sensor presence.
  for(int i = 0; i < MAX_NUM_PORTS; i++) {    
    comm_check = false;
    device_acknowledge = safe_select_tca_port(i,  comm_check);

    if ((device_acknowledge != 0) || comm_check) {
      if (verbose) {
        Serial.print("Error occured while attempting communication with TCA "
          "device # ");
        Serial.print(i / PORTS_PER_TCA);
        Serial.println(".");
      }
      active_ports[i] = false;
      sensors[i] = SENSOR_NONE;
      continue;
    }

    // Soft reset the sensor to check
    if (verbose) {
      Serial.print("Attempting soft-reset to check for sensor on TCA port # ");
      Serial.print(i);
      Serial.println(".");
    }
    device_acknowledge = sensirion_soft_reset();
    delay(SENSOR_RESET_WAIT + 10); // wait for sensor reset to complete

    if (device_acknowledge != 0) {
      if (verbose) {
        Serial.print("No device detected on TCA port # ");
        Serial.print(i);
        Serial.println(".");
      }
      active_ports[i] = false;
      sensors[i] = SENSOR_NONE;
      continue;
    }

    if (verbose) {
      Serial.print("Device detected on TCA port # ");
      Serial.print(i);
      Serial.println(".");
    }

    comm_check = false;
    sensor_product_number = sensirion_read_product_number(comm_check);

    if (comm_check) {
      if (verbose) {
        Serial.println("I2C communication error, could not read product number "
          "from device on port # ");
        Serial.print(i);
        Serial.println(".");
      }
      active_ports[i] = false;
      sensors[i] = SENSOR_NONE;
      continue;
    }

    if (verbose) {
      Serial.print("Device product number was detected to be ");
      Serial.print("0x");
      Serial.print(sensor_product_number, HEX);
      Serial.println(".");
    }
    active_ports[i] = true;

    switch (sensor_product_number) {
      case S1300F_PN:
        sensors[i] = SENSOR_S1300F;
        break;
      case S0600F_PN:
        sensors[i] = SENSOR_S0600F;
        break;
      default:
        sensors[i] = SENSOR_NONE;
        break;
    }

    if (verbose) {
      Serial.print("Sensor on port # ");
      Serial.print(i);
      Serial.print(" is identified to be ");
      Serial.print(SENSOR_PART_NAMES[sensors[i]]);
      Serial.println(".");
    }
  }

  for (int i = 0; i < MAX_NUM_PORTS; i++) {
    if ((!active_ports[i]) || (sensors[i] == SENSOR_NONE)) {
      continue; // dismiss if this port/sensor is not active.
    }

    comm_check = false;

    device_acknowledge = safe_select_tca_port(i, comm_check);

    if ((device_acknowledge != 0) || comm_check) {
      if (verbose) {
        Serial.print("Error occured while attempting communication with TCA "
          "device # ");
        Serial.print(i / PORTS_PER_TCA);
        Serial.println(".");
        Serial.print("Failed to write measurement mode to sensor on TCA port # ");
        Serial.print(i);
        Serial.println(".");
      }
      continue;
    }

    device_acknowledge = sensirion_begin_continuous_measure();

    if (device_acknowledge != 0) {
      if (verbose) {
        Serial.print("Communication error (");
        Serial.print(END_TRANSMIT_ERRORS[device_acknowledge]);
        Serial.print(") occured during write measurement mode "
          "command to sensor on TCA port # ");
        Serial.print(i);
        Serial.println(".");
      }
      continue;
    }

    if (verbose) {
      Serial.print("Continuous measurement mode on TCA port # ");
      Serial.print(i);
      Serial.println(" initiated.");
    }
  }

  delay(SENSOR_WARMUP_WAIT + 10);
}

// -----------------------------------------------------------------------------
// Communication setup
// -----------------------------------------------------------------------------
constexpr unsigned NUM_SENSORS = 12;
// value for indicating port is invalid/not used
const unsigned NOT_USED_SENSOR = NUM_SENSORS;

const unsigned SENSOR_PORT_MAP[MAX_NUM_PORTS] = {
  0,   1,  2,  3,
  4,   5,  6,  7,
  8,   9, 10, 11,
  NOT_USED_SENSOR,
  NOT_USED_SENSOR,
  NOT_USED_SENSOR,
  NOT_USED_SENSOR};

void float_to_str(float x, char buffer[]) {
  // char sign_str[2];
  // char num_buffer[14];  
  const unsigned NUM_DECIMALS = 4;
  dtostrf(x, NUM_DECIMALS + 5, NUM_DECIMALS, buffer);
}

void print_array(float arr[], char label[4]) { 
  char num_str[15];

  Serial.print("<");
  Serial.print(label);
  Serial.print(":");
  for (int i = 0; i < NUM_SENSORS; ++i) {
    float_to_str(arr[i], num_str);
    Serial.print(num_str);
    if (i < (NUM_SENSORS - 1)) {
      Serial.print(",");
    }
  }
  Serial.print(">");
}

void print_array(bool arr[], char label[4]) {
  char num_str[2];

  Serial.print("<");
  Serial.print(label);
  Serial.print(":");
  for (int i = 0; i < NUM_SENSORS; ++i) {
    Serial.print(arr[i] ? "1" : "0");
    if (i < (NUM_SENSORS - 1)) {
      Serial.print(",");
    }
  }
  Serial.print(">");
}

// -----------------------------------------------------------------------------
// The Arduino `loop` routune will run continuously
// -----------------------------------------------------------------------------

void loop() {
  int16_t raw_flow_value;
  int16_t raw_temp_value;
  float flow_value;
  float temp_value;
  bool air_in_line_flag;
  bool high_flow_flag;
  bool exp_smooth_flag;
  bool flow_values_failed;
  bool temp_value_failed;
  bool flag_values_failed;
  bool comm_check;
  int device_acknowledge;
  
  // in continuous measurement mode the first 3 bytes are the flow reading
  // w/ checksum, the second 3 bytes are the temp reading w/ checksum,
  // and the third set of 3 bytes are the signaling flags and checksum.
  const unsigned long expected_num_bytes = 9;
  
  unsigned sensor_index;
  const unsigned long FAIL_DELAY = 1;

  float flow_values[NUM_SENSORS];
  float temp_values[NUM_SENSORS];
  bool air_in_line_flags[NUM_SENSORS];
  bool high_flow_flags[NUM_SENSORS];

  bool flow_value_failed_arr[NUM_SENSORS];
  bool temp_value_failed_arr[NUM_SENSORS];
  bool flag_value_failed_arr[NUM_SENSORS];

  // char serial_buffer[460];

  for (int i = 0; i < MAX_NUM_PORTS; i++) {
    sensor_index = SENSOR_PORT_MAP[i];
    
    if (sensor_index == NOT_USED_SENSOR) {
      continue; // dismiss if this port/sensor is not active.
    }

    // default values; will be overwritten if reads from sensor are sucessful
    flow_values[sensor_index] = 0;
    temp_values[sensor_index] = 0;
    air_in_line_flags[sensor_index] = false;
    high_flow_flags[sensor_index] = false;
    flow_value_failed_arr[sensor_index] = true;
    temp_value_failed_arr[sensor_index] = true;
    flag_value_failed_arr[sensor_index] = true;

    if ((!active_ports[i]) || (sensors[i] == SENSOR_NONE)) {
      delay(FAIL_DELAY);
      continue;
    }

    comm_check = false;
    device_acknowledge = safe_select_tca_port(i, comm_check);

    if ((device_acknowledge != 0) || comm_check) {
      if (verbose) {
        Serial.print("Error occured while attempting communication with TCA "
          "device # ");
        Serial.print(i / PORTS_PER_TCA);
        Serial.println(".");
        Serial.print("Failed to take reading from sensor on TCA port # ");
        Serial.print(i);
        Serial.println(".");
      }
      delay(FAIL_DELAY);
      continue;
    }

    Wire.requestFrom(SENSOR_ADDRESS, expected_num_bytes);
    if (Wire.available() < expected_num_bytes) {
      if (verbose) {
        Serial.print("Received too few bytes from sensor on TCA port # ");
        Serial.print(i);
        Serial.println(".");
        Serial.print("Failed to take reading from sensor on TCA port # ");
        Serial.print(i);
        Serial.println(".");
      }
      delay(FAIL_DELAY);
      continue;
    }

    comm_check = false;
    raw_flow_value = sensirion_read_int16(comm_check);

    if (comm_check) {
      if (verbose) {
        Serial.print("I2C communication error corrupted flow reading from "
          "sensor on TCA port # ");
        Serial.print(i);
        Serial.println(".");
      }
      flow_value = 0;
      flow_values_failed = true;
    } else {
      flow_value = ((float) raw_flow_value) / FLOW_SCALE_FACTORS[sensors[i]];
      flow_values_failed = false;
    }

    comm_check = false;
    raw_temp_value = sensirion_read_int16(comm_check);

    if (comm_check) {
      if (verbose) {
        Serial.print("I2C communication error corrupted reading temperature from "
          "sensor on TCA port # ");
        Serial.print(i);
        Serial.println(".");
      }
      temp_value = 0;
      temp_value_failed = true;
    } else {
      temp_value = ((float) raw_temp_value) / SCALE_FACTOR_TEMP;
      temp_value_failed = false;
    }

    comm_check = false;
    sensirion_read_measurement_flags(
      air_in_line_flag, high_flow_flag, exp_smooth_flag, comm_check);

    if (comm_check) {
      if (verbose) {
        Serial.print("I2C communication error corrupted reading flags from "
          "sensor on TCA port # ");
        Serial.print(i);
        Serial.println(".");
      }
      air_in_line_flag = false;
      high_flow_flag = false;
      exp_smooth_flag = false;
      flag_values_failed = true;
    } else {
      flag_values_failed = false;
    }

    if (verbose) {
      Serial.print("SENSOR-");
      Serial.print(i + 1);
      Serial.print(" (");
      Serial.print(SENSOR_PART_NAMES[sensors[i]]);
      Serial.print(") : ");
      
      Serial.print("flow_rate: ");
      if (!flow_values_failed) {
        Serial.print(flow_value, 3);
      } else {
        Serial.print("????");
      }
      Serial.print(" ");
      Serial.print(UNIT_FLOW);

      Serial.print(", temp: ");
      if (!temp_value_failed) {
        Serial.print(temp_value, 1);
      } else {
        Serial.print("????");
      }
      Serial.print(" ");
      Serial.print(UNIT_TEMP);

      if (!flag_values_failed) {
        Serial.print(", air_in_line: ");
        Serial.print(air_in_line_flag ? "YES" : "NO");
        Serial.print(", high_flow: ");
        Serial.print(high_flow_flag ? "YES" : "NO");
        Serial.println(", exp_smooth_active: ");
        Serial.println(exp_smooth_flag ? "YES" : "NO");
      } else {
        Serial.print(", air_in_line: ");
        Serial.print("?");
        Serial.print(", high_flow: "); 
        Serial.print("?");
        Serial.println(", exp_smooth_active: ");
        Serial.println("?");
      }
    }

    flow_values[sensor_index] = flow_value;
    temp_values[sensor_index] = temp_value;
    air_in_line_flags[sensor_index] = air_in_line_flag;
    high_flow_flags[sensor_index] = high_flow_flag;
    flow_value_failed_arr[sensor_index] = flow_values_failed;
    temp_value_failed_arr[sensor_index] = temp_value_failed;
    flag_value_failed_arr[sensor_index] = flag_values_failed;
  }

  /*
  strcpy(serial_buffer, "");  
  strcat_array(flow_values, "flow", serial_buffer);
  strcat_array(flow_value_failed_arr, "fwok", serial_buffer);
  strcat_array(temp_values, "temp", serial_buffer);
  strcat_array(temp_value_failed_arr, "tpok", serial_buffer);
  strcat_array(high_flow_flags, "hffg", serial_buffer);
  strcat_array(air_in_line_flags, "alfg", serial_buffer);
  strcat_array(flag_value_failed_arr, "fgok", serial_buffer);
  Serial.println(serial_buffer);
  */
  
  print_array(flow_values, "flow");
  print_array(flow_value_failed_arr, "fwok");
  print_array(temp_values, "temp");
  print_array(temp_value_failed_arr, "tpok");
  print_array(high_flow_flags, "hffg");
  print_array(air_in_line_flags, "alfg");
  print_array(flag_value_failed_arr, "fgok");
  Serial.println();
}
