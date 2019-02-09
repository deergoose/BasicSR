import cv2

scale = 4

for i in range(25):
    im = cv2.imread('/Users/zhoufang/Desktop/lu/results/{}_hr.png'.format(i))
    im = cv2.resize(im, None, fx=1./scale, fy=1./scale, interpolation=cv2.INTER_AREA)
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('/Users/zhoufang/Desktop/lu/results/{}_lr.png'.format(i), im)
