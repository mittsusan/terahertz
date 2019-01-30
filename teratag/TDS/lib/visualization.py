from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
# カラーコード方式で可視化(厚み、横の数、縦の数)
def colorcode(best_pred, x, y):
    im = Image.new('RGB', (100 + x * 100, 100 + y * 100), (128, 128, 128))
    draw = ImageDraw.Draw(im)
    position = []
    for position_number in range(x):
        position.append(50)
    for i, j in enumerate(best_pred):
        if j == 1:
            color_R = 255
            color_G = 0
            color_B = 0
        elif j == 2:
            color_R = 0
            color_G = 255
            color_B = 0
        elif j == 3:
            color_R = 0
            color_G = 0
            color_B = 255

        if i < x:
            draw.rectangle((position[0], 30, position[0] + 100, 130), fill=(color_R, color_G, color_B))
            position[0] += 110
        elif i < x * 2:
            draw.rectangle((position[1], 140, position[1] + 100, 240), fill=(color_R, color_G, color_B))
            position[1] += 110
        elif i < x * 3:
            draw.rectangle((position[2], 250, position[2] + 100, 350), fill=(color_R, color_G, color_B))
            position[2] += 110

    plt.imshow(np.array(im))
    plt.show()