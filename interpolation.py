import cv2
import numpy as np
import math

Radius = 30
ixs, iys = -1, -1
ixt, iyt = -1, -1
start = True
interpolation_picker = 1
direction = 0


def callback_interpolation_near(x):
    global interpolation_picker
    if x == 1:
        interpolation_picker = 1


def callback_interpolation_bili(x):
    global interpolation_picker
    if x == 1:
        interpolation_picker = 2


def callback_interpolation_cub(x):
    global interpolation_picker
    if x == 1:
        interpolation_picker = 3


def callback_radius(x):
    global Radius
    Radius = x


def nearest_neighbor(x, y, m):
    # return m[i, j]
    x_max, y_max = m.shape[0] - 1, m.shape[1] - 1
    if np.floor(x) == x and np.floor(y) == y:
        x, y = int(x), int(y)
        return m[x, y]
    if np.abs(np.floor(x) - x) < np.abs(np.ceil(x) - x):
        x = int(np.floor(x))
    else:
        x = int(np.ceil(x))
    if np.abs(np.floor(y) - y) < np.abs(np.ceil(y) - y):
        y = int(np.floor(y))
    else:
        y = int(np.ceil(y))
    if x > x_max:
        x = x_max
    if y > y_max:
        y = y_max
    return m[x, y]


def bilinear(i, j, m):
    i = np.asarray(i)
    j = np.asarray(j)
    i0 = np.floor(i).astype(int)
    i1 = i0 + 1
    j0 = np.floor(j).astype(int)
    j1 = j0 + 1
    i0 = np.clip(i0, 0, m.shape[1] - 1)
    i1 = np.clip(i1, 0, m.shape[1] - 1)
    j0 = np.clip(j0, 0, m.shape[0] - 1)
    j1 = np.clip(j1, 0, m.shape[0] - 1)
    return ((i1 - i) * (j1 - j)) * m[j0, i0] + ((i1 - i) * (j - j0)) * m[j1, i0] + ((i - i0) * (j1 - j)) * m[j0, i1] \
           + ((i - i0) * (j - j0)) * m[j1, i1]


def make_equation(x, y, m):
    eq = []
    for i in range(4):
        for j in range(4):
            eq.append(math.pow(x, i) * math.pow(y, j))
    return eq, m[x, y]


def bicubic(x, y, m):
    b = []
    a = []
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = \
        (x0, x0 - 1, x0 + 1, x0, x0, x0 - 1, x0 + 1, x0 - 1, x0 + 1, x0 - 1, x0 + 1, x0 + 2, x0 + 2, x0 + 2, x0 + 2)
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15 = \
        (y0 - 1, y0, y0, y0 + 1, y0 + 2, y0 - 1, y0 - 1, y0 + 1, y0 + 1, y0 + 2, y0 + 2, y0 + 2, y0 + 1, y0, y0 - 1)
    xis = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15]
    yis = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15]
    for x, y in zip(xis, yis):
        ai, bi = make_equation(x, y, m)
        a.append(ai)
        b.append(bi)

    ais = np.linalg.solve(a, b)
    part, _ = make_equation(x, y, m)
    res = 0
    for a, xi in zip(ais, part):
        res = res + a * xi
    return res


def interpolation(x, y, m):
    global interpolation_picker
    image_big = cv2.copyMakeBorder(m, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    if interpolation_picker == 1:
        return nearest_neighbor(x + 2, y + 2, image_big)
    elif interpolation_picker == 2:
        return bilinear(x + 2, y + 2, image_big)
    elif interpolation_picker == 3:
        return bicubic(x + 2, y + 2, image_big)
    else:
        raise Exception('non existing interpolation')


def distortion():
    global ixs, iys, ixt, iyt, Radius, cols, rows, direction
    sx = -1
    sy = -1
    ex = -1
    ey = -1
    sx1 = -1
    sy1 = -1
    sx2 = -1
    sy2 = -1
    ax1 = -1
    ay1 = -1
    ax2 = -1
    ay2 = -1
    ex1 = -1
    ey1 = -1
    ex2 = -1
    ey2 = -1
    slope = (iyt - iys) / (ixt - ixs)
    anti_slope = -(1 / slope)
    img_og = img.copy()
    angle = abs(math.atan(slope))
    anti_angle = abs(math.atan(anti_slope))
    if ixs < ixt:
        if iys < iyt:
            direction = 1
        if iys > iyt:
            direction = 2
    elif ixs > ixt:
        if iys < iyt:
            direction = 4
        if iys > iyt:
            direction = 3
    if direction == 1:
        sx = ixs - (Radius * math.cos(angle))
        sy = iys - (Radius * math.sin(angle))
        ex = ixt + (Radius * math.cos(angle))
        ey = iyt + (Radius * math.sin(angle))
        ax1 = ixs - (Radius * math.cos(anti_angle))
        ay1 = iys + (Radius * math.sin(anti_angle))
        ax2 = ixs + (Radius * math.cos(anti_angle))
        ay2 = iys - (Radius * math.sin(anti_angle))
        sx1 = sx - (Radius * math.cos(anti_angle))
        sy1 = sy + (Radius * math.sin(anti_angle))
        sx2 = sx + (Radius * math.cos(anti_angle))
        sy2 = sy - (Radius * math.sin(anti_angle))
        ex1 = ex - (Radius * math.cos(anti_angle))
        ey1 = ey + (Radius * math.sin(anti_angle))
        ex2 = ex + (Radius * math.cos(anti_angle))
        ey2 = ey - (Radius * math.sin(anti_angle))
    elif direction == 2:
        sx = ixs - (Radius * math.cos(angle))
        sy = iys + (Radius * math.sin(angle))
        ex = ixt + (Radius * math.cos(angle))
        ey = iyt - (Radius * math.sin(angle))
        ax1 = ixs + (Radius * math.cos(anti_angle))
        ay1 = iys + (Radius * math.sin(anti_angle))
        ax2 = ixs - (Radius * math.cos(anti_angle))
        ay2 = iys - (Radius * math.sin(anti_angle))
        sx1 = sx + (Radius * math.cos(anti_angle))
        sy1 = sy + (Radius * math.sin(anti_angle))
        sx2 = sx - (Radius * math.cos(anti_angle))
        sy2 = sy - (Radius * math.sin(anti_angle))
        ex1 = ex + (Radius * math.cos(anti_angle))
        ey1 = ey + (Radius * math.sin(anti_angle))
        ex2 = ex - (Radius * math.cos(anti_angle))
        ey2 = ey - (Radius * math.sin(anti_angle))
    elif direction == 3:
        sx = ixs + (Radius * math.cos(angle))
        sy = iys + (Radius * math.sin(angle))
        ex = ixt - (Radius * math.cos(angle))
        ey = iyt - (Radius * math.sin(angle))
        ax1 = ixs - (Radius * math.cos(anti_angle))
        ay1 = iys + (Radius * math.sin(anti_angle))
        ax2 = ixs + (Radius * math.cos(anti_angle))
        ay2 = iys - (Radius * math.sin(anti_angle))
        sx1 = sx - (Radius * math.cos(anti_angle))
        sy1 = sy + (Radius * math.sin(anti_angle))
        sx2 = sx + (Radius * math.cos(anti_angle))
        sy2 = sy - (Radius * math.sin(anti_angle))
        ex1 = ex - (Radius * math.cos(anti_angle))
        ey1 = ey + (Radius * math.sin(anti_angle))
        ex2 = ex + (Radius * math.cos(anti_angle))
        ey2 = ey - (Radius * math.sin(anti_angle))
    elif direction == 4:
        sx = ixs + (Radius * math.cos(angle))
        sy = iys - (Radius * math.sin(angle))
        ex = ixt - (Radius * math.cos(angle))
        ey = iyt + (Radius * math.sin(angle))
        ax1 = ixs + (Radius * math.cos(anti_angle))
        ay1 = iys + (Radius * math.sin(anti_angle))
        ax2 = ixs - (Radius * math.cos(anti_angle))
        ay2 = iys - (Radius * math.sin(anti_angle))
        sx1 = sx + (Radius * math.cos(anti_angle))
        sy1 = sy + (Radius * math.sin(anti_angle))
        sx2 = sx - (Radius * math.cos(anti_angle))
        sy2 = sy - (Radius * math.sin(anti_angle))
        ex1 = ex + (Radius * math.cos(anti_angle))
        ey1 = ey + (Radius * math.sin(anti_angle))
        ex2 = ex - (Radius * math.cos(anti_angle))
        ey2 = ey - (Radius * math.sin(anti_angle))
    sx = int(sx)
    sy = int(sy)
    ex = int(ex)
    ey = int(ey)
    sx1 = int(sx1)
    sy1 = int(sy1)
    sx2 = int(sx2)
    sy2 = int(sy2)
    ax1 = int(ax1)
    ay1 = int(ay1)
    ax2 = int(ax2)
    ay2 = int(ay2)
    ex1 = int(ex1)
    ey1 = int(ey1)
    ex2 = int(ex2)
    ey2 = int(ey2)

    def f1(x):
        return slope * (x - ax1) + ay1

    def f2(x):
        return slope * (x - ax2) + ay2

    def g3(x): return anti_slope * (x - ixs) + iys

    def g4(x): return anti_slope * (x - ixt) + iyt
    if ex1 > sx1:

        def g1(x):
            return anti_slope * (x - sx1) + sy1

        def g2(x):
            return anti_slope * (x - ex1) + ey1
    else:

        def g1(x):
            return anti_slope * (x - ex1) + ey1

        def g2(x):
            return anti_slope * (x - sx1) + sy1

    def parabola(x, y):
        return (y * math.sqrt(math.pow(ixs - ixt, 2) + math.pow(iys - iyt, 2))) / Radius * \
               ((math.pow(Radius, 2) - math.pow(x, 2)) / math.pow(Radius, 2))

    def parabola2(x, y, dis): return y / math.sqrt(math.pow(ixs - ex, 2) + math.pow(iys - ey, 2)) *\
                                (Radius - (math.pow(x, 2) / Radius)) +\
                                math.sqrt(math.pow(ixs - ixt, 2) + math.pow(iys - iyt, 2)) *\
                                (dis /
                                 math.sqrt(math.pow(ixs - ex, 2) + math.pow(iyt - ey, 2)))

    def parabola_inverse(x, y): return y * Radius /\
                                       math.sqrt(math.pow(ixt - sx, 2) + math.pow(iyt - sy, 2)) -\
                                        y * math.pow(x, 2) / (Radius *
                                        math.sqrt(math.pow(ixt - sx, 2) + math.pow(iyt - sy, 2)))

    def parabola2_inverse(x, y): return y * math.sqrt(math.pow(ixs - ex, 2) + math.pow(iys - ey, 2)) / Radius -\
                                        y * math.pow(x, 2) * math.sqrt(math.pow(ixs - ex, 2) + math.pow(iys - ey, 2)) /\
                                        math.pow(Radius, 3)

    # Transformation
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            if f2(i) <= j <= f1(i) and ((g1(i) <= j <= g2(i) and (direction == 1 or direction == 3)) or
                                        (g1(i) >= j >= g2(i) and (direction == 2 or direction == 4))):
                if (g3(i) >= j and (direction == 1 or direction == 4)) or \
                        (g3(i) <= j and (direction == 3 or direction == 2)):
                    d = math.sqrt(math.pow(ixs - i, 2) + math.pow(iys - j, 2))
                    try:
                        new_angle = math.atan(((j - iys) / (i - ixs)))
                    except ZeroDivisionError:
                        new_angle = 1.5708
                    new_angle = abs(new_angle - angle)
                    d_tag = abs(d * math.cos(new_angle))
                    jump_distance = parabola(math.sqrt(math.pow(d, 2) -
                                                       math.pow(d_tag, 2)), d_tag)
                    # print(jump_distance)
                    if direction == 1:
                        img[int(j + jump_distance * math.sin(angle)), int(i + jump_distance * math.cos(angle))] = \
                            img_og[j, i]
                    if direction == 2:
                        img[int(j - jump_distance * math.sin(angle)), int(i + jump_distance * math.cos(angle))] = \
                            img_og[j, i]
                    if direction == 3:
                        img[int(j - jump_distance * math.sin(angle)), int(i - jump_distance * math.cos(angle))] = \
                            img_og[j, i]
                    if direction == 4:
                        img[int(j + jump_distance * math.sin(angle)), int(i - jump_distance * math.cos(angle))] = \
                            img_og[j, i]
                else:
                    d = math.sqrt(math.pow(ixs - i, 2) + math.pow(iys - j, 2))
                    try:
                        new_angle = math.atan(((j - iys) / (i - ixs)))
                    except ZeroDivisionError:
                        new_angle = 1.5708
                    new_angle = abs(new_angle - angle)
                    d_tag = abs(d * math.cos(new_angle))
                    jump_distance = parabola2(math.sqrt(math.pow(d, 2) - math.pow(d_tag, 2)), d_tag,
                                              math.sqrt(math.pow(ixs - ixt, 2) + math.pow(iys - iyt, 2)) - d_tag)
                    if direction == 1:
                        img[int(j + jump_distance * math.sin(angle)), int(i + jump_distance * math.cos(angle))] = \
                            img_og[j, i]
                    if direction == 2:
                        img[int(j - jump_distance * math.sin(angle)), int(i + jump_distance * math.cos(angle))] = \
                            img_og[j, i]
                    if direction == 3:
                        img[int(j - jump_distance * math.sin(angle)), int(i - jump_distance * math.cos(angle))] = \
                            img_og[j, i]
                    if direction == 4:
                        img[int(j + jump_distance * math.sin(angle)), int(i - jump_distance * math.cos(angle))] = img_og[j, i]

    # Interpolation
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            if f2(i) <= j <= f1(i) and ((g1(i) <= j <= g2(i) and (direction == 1 or direction == 3)) or
                                      (g1(i) >= j >= g2(i) and (direction == 2 or direction == 4))):
                if (g4(i) >= j and (direction == 1 or direction == 4)) or \
                        (g4(i) <= j and (direction == 3 or direction == 2)):
                    d = math.sqrt(math.pow(sx - i, 2) + math.pow(sy - j, 2))
                    try:
                        new_angle = math.atan(((j - sy) / (i - sx)))
                    except ZeroDivisionError:
                        new_angle = 1.5708
                    new_angle = abs(new_angle - angle)
                    d_tag = abs(d * math.cos(new_angle))
                    jump_distance = parabola_inverse(math.sqrt(math.pow(d, 2) - math.pow(d_tag, 2)), d_tag)
                    if direction == 1:
                        img[j, i] = interpolation(sx + jump_distance * math.cos(angle), sy +
                                                  jump_distance * math.sin(angle), img_og)
                    if direction == 2:
                        img[j, i] = interpolation(sx - jump_distance * math.cos(angle), sy +
                                                  jump_distance * math.sin(angle), img_og)
                    if direction == 3:
                        img[j, i] = interpolation(sx - jump_distance * math.cos(angle), sy -
                                                  jump_distance * math.sin(angle), img_og)
                    if direction == 4:
                        img[j, i] = interpolation(sx + jump_distance * math.cos(angle), sy -
                                                  jump_distance * math.sin(angle), img_og)
                else:
                    d = math.sqrt(math.pow(ixs - i, 2) + math.pow(iys - j, 2))
                    try:
                        new_angle = math.atan(((j - iys) / (i - ixs)))
                    except ZeroDivisionError:
                        new_angle = 1.5708
                    new_angle = abs(new_angle - angle)
                    d_tag = abs(d * math.cos(new_angle))
                    jump_distance = parabola2_inverse(math.sqrt(math.pow(d, 2) - math.pow(d_tag, 2)), d_tag)
                    if direction == 1:
                        img[j, i] = interpolation(ixs + jump_distance * math.cos(angle), iys +
                                                  jump_distance * math.sin(angle), img_og)
                    if direction == 2:
                        img[j, i] = interpolation(sx - jump_distance * math.cos(angle), sy +
                                                  jump_distance * math.sin(angle), img_og)
                    if direction == 3:
                        img[j, i] = interpolation(sx - jump_distance * math.cos(angle), sy -
                                                  jump_distance * math.sin(angle), img_og)
                    if direction == 4:
                        img[j, i] = interpolation(sx + jump_distance * math.cos(angle), sy -
                                                  jump_distance * math.sin(angle), img_og)

    # m_rotation = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 2)
    # tmp_img = cv2.warpAffine(img, m_rotation, (cols, rows))
    # m_translation = np.float32([[1, 0, 5], [0, 1, 5]])
    # tmp_img = cv2.warpAffine(tmp_img, m_translation, (cols, rows))
    # print(ixs, iys, ixt, iyt, Radius)
    # scale = 20
    # print(angle)
    # for i in range(iys - Radius, iys + Radius):
    #     for j in range(ixs - Radius, ixs + Radius):
    #         if math.sqrt(math.pow(ixs - j, 2) + math.pow(iys - i, 2)) <= Radius:
    #             img[i, j] = tmp_img[i, j]
    # for i in range(iyt - Radius, iyt + Radius):
    #     for j in range(ixt - Radius, ixt + Radius):
    #         if math.sqrt(math.pow(ixt - j, 2) + math.pow(iyt - i, 2)) <= Radius:
    #             img[i, j] = tmp_img[i, j]
    # a = np.squeeze(np.asarray(m_rotation))
    # a = np.vstack([a, [0, 0, 1]])
    # m_rotation = np.asmatrix(a)
    # a = np.squeeze(np.asarray(m_translation))
    # a = np.vstack([a, [0, 0, 1]])
    # m_translation = np.asmatrix(a)
    # print(m_rotation)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if math.sqrt(math.pow(ixt - i, 2) + math.pow(iyt - j, 2)) <= Radius:
    #             img[i, j] = interpolation(i, j, img_og, np.linalg.inv(m_rotation))
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if math.sqrt(math.pow(ixt - i, 2) + math.pow(iyt - j, 2)) <= Radius:
    #             img[i, j] = interpolation(i, j, img_og, np.linalg.inv(m_translation))
    # cv2.imshow('image', img)
    # cv2.waitKey(0)


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ixs, iys, ixt, iyt, start
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if start:
            ixs, iys = x, y
            start = False
        else:
            ixt, iyt = x, y
            distortion()

        cv2.circle(img, (x, y), Radius, (255, 0, 0), 1)
        cv2.imshow('image', img)


img = cv2.imread('im.jpg', 0)
rows, cols = img.shape
cv2.namedWindow('image')
cv2.createTrackbar('Radius', 'image', 30, 50, callback_radius)
cv2.createTrackbar('nearest', 'image', 0, 1, callback_interpolation_near)
cv2.createTrackbar('bilinear', 'image', 0, 1, callback_interpolation_bili)
cv2.createTrackbar('cubic', 'image', 0, 1, callback_interpolation_cub)

cv2.imshow('image', img)
cv2.setMouseCallback('image', draw_circle)
cv2.waitKey(0)
cv2.destroyAllWindows()
