from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
# カラーコード方式で可視化(厚み、横の数、縦の数)
def colorcode(best_pred, x, y):
    plt.style.use('default')
    im = Image.new('RGB', (100 + x * 100, 100 + y * 100), (128, 128, 128))
    draw = ImageDraw.Draw(im)
    #position = []
    position_x = []
    position_y = []
    first_position_y = 30
    color_R = 50
    color_G = 125
    color_B = 75
    j_save = 0
    for i in range(y+2):
        position_x.append(50)

    for i in range(y+2):
        position_y.append(first_position_y)
        first_position_y += 110
    for i, j in enumerate(best_pred):
        '''
        if j != j_save:
            color_R = color_G*2
            color_G = color_B*2
            color_B = color_R*2


        if color_R >= 255:
            color_R = 10

        if color_G >= 255:
            color_G = 10

        if color_R >= 255:
            color_G = 10
        '''

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
        print(i)
        print(j)


        for k in range(y):
            print('loopstart')
            print(k)
            print('kkk{}'.format(k))
            print('x{}'.format(x))
            print('i:{}'.format(i))
            if int(i) < int(x) * int(k+1):
                print('描けた')
                print(position_x)
                print(position_y)
                print('k{}'.format(k))
                print(j)
                draw.rectangle((position_x[k], position_y[k], position_x[k] + 100, position_y[k+1] - 10),
                               fill=(color_R, color_G, color_B))
                position_x[k] += 110
                break
            print('描けなかった')


        '''
        if i < x:
            draw.rectangle((position_x[0], position_y[0], position_x[0] + 100, position_y[1]-10), fill=(color_R, color_G, color_B))
            position_x[0] += 110
        elif i < x * 2:
            draw.rectangle((position_x[1], position_y[1], position_x[1] + 100, position_y[2]-10), fill=(color_R, color_G, color_B))
            position_x[1] += 110
        elif i < x * 3:
            draw.rectangle((position_x[2], position_y[2], position_x[2] + 100, position_y[3]-10), fill=(color_R, color_G, color_B))
            position_x[2] += 110
        elif i < x * 4:
            draw.rectangle((position_x[3], position_y[3], position_x[3] + 100, position_y[4]-10), fill=(color_R, color_G, color_B))
            position_x[3] += 110
        
        '''

        #j_save = j
    plt.imshow(np.array(im))
    plt.show()