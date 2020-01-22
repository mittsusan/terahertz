from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
# カラーコード方式で可視化(厚み、横の数、縦の数)
def concentration_colorcode(best_pred, x, y):
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
    r = 0
    g = 0
    b = 0
    j_save = 0

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
            color_R = int(255*0.8)
            color_G = int(255*0.2)
            color_B = 0
        elif j == 5:
            color_R = 0
            color_G = int(255*0.8)
            color_B = int(255*0.8)
        elif j == 6:
            color_R = int(255*0.2)
            color_G = 0
            color_B = int(255*0.8)
        elif j == 7:
            color_R = int(255*0.6)
            color_G = int(255*0.4)
            color_B = 0
        elif j == 8:
            color_R = 0
            color_G = int(255*0.6)
            color_B = int(255*0.4)
        elif j == 9:
            color_R = int(255*0.4)
            color_G = 0
            color_B = int(255*0.6)
        elif j == 10:
            color_R = int(255*0.4)
            color_G = int(255*0.6)
            color_B = 0
        elif j == 11:
            color_R = 0
            color_G = int(255*0.4)
            color_B = int(255*0.6)
        elif j == 12:
            color_R = int(255*0.6)
            color_G = 0
            color_B = int(255*0.4)
        elif j == 13:
            color_R = int(255*0.2)
            color_G = int(255*0.8)
            color_B = 0
        elif j == 14:
            color_R = 0
            color_G = int(255*0.2)
            color_B = int(255*0.8)
        elif j == 15:
            color_R = int(255*0.8)
            color_G = 0
            color_B = int(255*0.2)
        # print(i)
        # print(j)
        for k in range(y):
            if int(i) < int(x) * int(k+1):
                draw.rectangle((position_x[k], position_y[k], position_x[k] + 100, position_y[k+1] - 10),
                                fill=(color_R, color_G, color_B))
                position_x[k] += 110

                break
    plt.imshow(np.array(im))
    plt.show()
    #j_save = j

    '''
    for m in range(1,last_type+1):
        if j == m:
            
                def Base_10_to_n(X,n):
                    if (int(X/n)):
                        return Base_10_to_n(int(X/n), n)+str(X%n)
                    return str(X%n)
                print(Base_10_to_n(m,3))

                
                if m % 3 == 1:
                    r = (m-1)/3
                    r = int(r)
                if m % 3 == 2:
                    g = (m - 2)/3
                    g = int(g)
                if m % 3 == 0:
                    b = m / 3
                    b = int(b)
                color_R = 0 + 127*r
                color_G = 0 + 127*g
                color_B = 0 + 127*b
            print(color_R,color_G,color_B)
            print()
            '''
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

        #j_save = j
    plt.imshow(np.array(im))
    plt.show()



