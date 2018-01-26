import freenect
import cv2

def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    #array = cv2.cvtColor(array,  cv2.COLOR_BGR2GRAY)
    return array

def set_path(namePic):
    setPath = '/home/uawsscu/PycharmProjects/Project3/image/' + namePic + '.jpg'
    return setPath

def capNeg(pathPic):
    i=0
    while 1:

        setPath = '/home/uawsscu/PycharmProjects/project3/tutor/neg/' + pathPic + str(i)+'.jpg'
        print "ok"
        frame = get_video()


        params = list()
        resized_image = cv2.resize(frame, (100, 100))
        #crop_img = frame[250:350, 290:390]  # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        cv2.imwrite(setPath, resized_image, params)
        i+=1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif i > 20:
            break

        cv2.imshow('RGB image', frame)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    cv2.destroyAllWindows()

def capPos(pathPic):
    i=0
    while 1:

        setPath = '/home/uawsscu/PycharmProjects/project3/tutor/pos/' + pathPic + str(i)+'.jpg'
        print "ok"
        frame = get_video()


        params = list()

        crop_img = frame[250:350, 290:390]  # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        #resized_image = cv2.resize(crop_img, (50, 50))
        #cv2.imwrite(setPath, resized_image, params)
        cv2.imwrite(setPath, crop_img, params)
        i+=1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif i > 10:
            break

        cv2.imshow('RGB image', frame)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    cv2.destroyAllWindows()

capPos('p')
#capNeg('n')