from PIL import Image

im = Image.open("240sx.jpg")

pixels_data = im.getdata()

pixels_matrix = list(pixels_data)

def rbg_to_brightness(pixel_matrix:list):
    for height in range(len(pixel_matrix)):
        for width in range(len(pixel_matrix[height])):
            pixel_rgb = pixel_matrix[height][width]
            red = pixel_rgb[0]
            green = pixel_rgb[1]
            blue = pixel_rgb[2]
            pixel_brightness = (max(red, green, blue) + min(red, green, bl)) / 2
            pixel_matrix[height][width] = pixel_brightness
            
            
pixels_matrix = rbg_to_brightness(pixels_matrix)

print(pixels_matrix)