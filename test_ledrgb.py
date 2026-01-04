import os
import re
import json
import shutil
import subprocess
from time import sleep
from pathlib import Path

from pidog.rgb_strip import RGBStrip  


def main():
    strip = RGBStrip()
    print("Turn LED BLUE (breath) while walking...")
    strip.set_mode(style="breath", color="blue", bps=1.2, brightness=0.8)
    strip.show()


if __name__ == "__main__":
    main()