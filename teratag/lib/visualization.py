from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
# カラーコード方式で可視化(厚み、横の数、縦の数)
def concentration_colorcode(best_pred, x, y, concentration_color_type):
    plt.style.use('default')
    im = Image.new('RGB', (100 + x * 100, 100 + y * 120), (128, 128, 128))
    draw = ImageDraw.Draw(im)
    position_x = []
    position_y = []
    first_position_y = 30
    for i in range(y+2):
        position_x.append(50)

    for i in range(y+2):
        position_y.append(first_position_y)
        first_position_y += 110

    for i, j in enumerate(best_pred):

        for k in range(y):
            if int(i) < int(x) * int(k+1):
                draw.rectangle((position_x[k], position_y[k], position_x[k] + 100, position_y[k+1] - 10),
                                fill=(concentration_color_type[j-1][0], concentration_color_type[j-1][1], concentration_color_type[j-1][2]))
                position_x[k] += 110

                break
    plt.imshow(np.array(im))
    plt.show()


# カラーコード方式で可視化(厚み、横の数、縦の数)
def colorcode(best_pred, x, y):
    plt.style.use('default')
    im = Image.new('RGB', (100 + x * 100, 100 + y * 100), (128, 128, 128))
    draw = ImageDraw.Draw(im)
    position_x = []
    position_y = []
    first_position_y = 30
    color_R = 50
    color_G = 125
    color_B = 75
    for i in range(y+2):
        position_x.append(50)

    for i in range(y+2):
        position_y.append(first_position_y)
        first_position_y += 110
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
        elif j == 4:
            color_R = 255
            color_G = 255
            color_B = 0
        elif j == 5:
            color_R = 255
            color_G = 0
            color_B = 255
        elif j == 6:
            color_R = 0
            color_G = 255
            color_B = 255
        elif j == 7:
            color_R = 255
            color_G = 255
            color_B = 255
        #print(i)
        #print(j)


        for k in range(y):
            if int(i) < int(x) * int(k+1):
                draw.rectangle((position_x[k], position_y[k], position_x[k] + 100, position_y[k+1] - 10),
                               fill=(color_R, color_G, color_B))
                position_x[k] += 110
                break


    plt.imshow(np.array(im))
    plt.show()



