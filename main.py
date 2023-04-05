from PIL import Image
import numpy as np

im = Image.open("240sx.jpg")

load = im.load()

pixels = []

pixels_matrix = []
for i in range(im.height):
    rows = []
    for j in range(im.width):
        coordinate = x,y = j,i
        each_pixel = im.getpixel(coordinate)
        rows.append(each_pixel)
    pixels_matrix.append(rows)

for i in range(len(pixels_matrix)):
    for j in range(len(pixels_matrix[i])):
        each_pixel = pixels_matrix[i][j]
        r,g,b = each_pixel
        brightness = int((r + g + b)/3)
        pixels_matrix[i][j] = brightness


ascii_symbols = "`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

for i in range(len(pixels_matrix)):
    for j in range(len(pixels_matrix[i])):
        each_pixel = pixels_matrix[i][j]
        print(ascii_symbols[int(each_pixel/4)], end="")
    print()