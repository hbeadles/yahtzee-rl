import random
import math
import numpy


UPPER_CATEGORY_NAMES = ["aces", "twos", "threes", "fours", "fives", "sixes"]
LOWER_CATEGORY_NAMES = ["three_of_a_kind", "four_of_a_kind", "yahtzee", "small_straight", "large_straight", "full_house", "chance"]
CATEGORY_NAMES = UPPER_CATEGORY_NAMES + LOWER_CATEGORY_NAMES

def dice_roll(num_dice=5):
    return numpy.ceil(numpy.random.uniform(0, 1, num_dice) * 6)


def main() -> None:
    print("Hello from yahtzee-rl!")
