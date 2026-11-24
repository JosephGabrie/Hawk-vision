import time
try:
    from gpiozero import OutputDevice
except ImportError:
    OutputDevice = None

class Buzzer:
    def __init__(self):
        """Initialize the Buzzer class."""
        self.PIN = 17                            # Set the GPIO pin for the buzzer
        self.buzzer_pin = None
        if OutputDevice is not None:
            try:
                self.buzzer_pin = OutputDevice(self.PIN) # Initialize the buzzer pin
            except Exception:
                self.buzzer_pin = None

    def set_state(self, state: bool) -> None:
        """Set the state of the buzzer."""
        if self.buzzer_pin is not None:
            self.buzzer_pin.on() if state else self.buzzer_pin.off() # Turn on or off the buzzer based on the state

    def close(self) -> None:
        """Close the buzzer pin."""
        if self.buzzer_pin is not None:
            self.buzzer_pin.close()           # Close the buzzer pin to release the GPIO resource

if __name__ == '__main__':
    print('Program is starting ... ')     # Print a message indicating the start of the program
    buzzer = Buzzer()                     # Create an instance of the Buzzer class
    try:
        for _ in range(3):
            buzzer.set_state(True)        # Turn on the buzzer
            time.sleep(0.1)               # Wait for 0.1 second
            buzzer.set_state(False)       # Turn off the buzzer
            time.sleep(0.1)               # Wait for 0.1 second
    finally:
        buzzer.close()                    # Ensure the buzzer pin is closed when the program is interrupted