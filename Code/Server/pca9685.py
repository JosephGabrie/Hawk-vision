#!/usr/bin/python

import glob
import math
import os
import sys
import time

if sys.platform.startswith("linux"):
    for _match in glob.glob("/usr/lib/python3*/dist-packages"):
        if _match not in sys.path:
            sys.path.append(_match)

try:
    import smbus
except ImportError:
    try:
        import smbus2 as smbus
    except ImportError:
        smbus = None


class _MockSMBus:
    def __init__(self, bus: int = 1):
        self._regs = {}

    def write_byte_data(self, addr: int, reg: int, val: int) -> None:
        self._regs[(addr, reg)] = val & 0xFF

    def read_byte_data(self, addr: int, reg: int) -> int:
        return self._regs.get((addr, reg), 0)

    def write_byte(self, addr: int, val: int) -> None:
        self._regs[(addr, 0)] = val & 0xFF

    def read_byte(self, addr: int) -> int:
        return self._regs.get((addr, 0), 128)

    def close(self) -> None:
        pass


# ============================================================================
# Raspi PCA9685 16-Channel PWM Servo Driver
# ============================================================================

class PCA9685:
    # Registers/etc.
    __SUBADR1            = 0x02
    __SUBADR2            = 0x03
    __SUBADR3            = 0x04
    __MODE1              = 0x00
    __PRESCALE           = 0xFE
    __LED0_ON_L          = 0x06
    __LED0_ON_H          = 0x07
    __LED0_OFF_L         = 0x08
    __LED0_OFF_H         = 0x09
    __ALLLED_ON_L        = 0xFA
    __ALLLED_ON_H        = 0xFB
    __ALLLED_OFF_L       = 0xFC
    __ALLLED_OFF_H       = 0xFD

    def __init__(self, address: int = 0x40, debug: bool = False):
        if smbus is not None:
            try:
                self.bus = smbus.SMBus(1)
            except Exception:
                self.bus = _MockSMBus(1)
        else:
            self.bus = _MockSMBus(1)
        self.address = address
        self.debug = debug
        self.write(self.__MODE1, 0x00)
    
    def write(self, reg: int, value: int) -> None:
        """Writes an 8-bit value to the specified register/address."""
        self.bus.write_byte_data(self.address, reg, value)
      
    def read(self, reg: int) -> int:
        """Read an unsigned byte from the I2C device."""
        result = self.bus.read_byte_data(self.address, reg)
        return result
    
    def set_pwm_freq(self, freq: float) -> None:
        """Sets the PWM frequency."""
        prescaleval = 25000000.0    # 25MHz
        prescaleval /= 4096.0       # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        prescale = math.floor(prescaleval + 0.5)

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10        # sleep
        self.write(self.__MODE1, newmode)        # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)


    def set_pwm(self, channel: int, on: int, off: int) -> None:
        """Sets a single PWM channel."""
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)
    def set_motor_pwm(self, channel: int, duty: int) -> None:
        """Sets the PWM duty cycle for a motor."""
        self.set_pwm(channel, 0, duty)

    def set_servo_pulse(self, channel: int, pulse: float) -> None:
        """Sets the Servo Pulse, The PWM frequency must be 50HZ."""
        pulse = pulse * 4096 / 20000        # PWM frequency is 50HZ, the period is 20000us
        self.set_pwm(channel, 0, int(pulse))

    def close(self) -> None:
        """Close the I2C bus."""
        self.bus.close()


if __name__=='__main__':
    pass
    
      
