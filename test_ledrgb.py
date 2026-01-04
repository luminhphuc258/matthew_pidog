from pidog.rgb_strip import RGBStrip  


def main():
    strip = RGBStrip()
    print("Turn LED BLUE (breath) while walking...")
    strip.set_mode(style="breath", color="blue", bps=1.2, brightness=0.8)
    strip.show()