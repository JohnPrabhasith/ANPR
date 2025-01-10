import cv2
from matplotlib import pyplot as plt
import numpy as np
import easyocr

class RunProcess():
    def __init__(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray, cmap='gray')
        plt.show()

        # Noise reduction and edge detection
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
        plt.imshow(edged, cmap='gray')
        plt.show()

        # Find contours
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:  # Looking for quadrilateral
                location = approx
                break

        if location is None:
            print("No contour with 4 sides found.")
            return

        # Masking and cropping
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        plt.show()

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]

        plt.imshow(cropped_image, cmap='gray')
        plt.show()

        # OCR using EasyOCR
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        if not result:
            print("No text found.")
            return

        text = result[0][-2]
        print("Detected Text:", text)

        # Annotating the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(location[0][0][0], location[0][0][1] - 10), 
                          fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.show()

if __name__ == "__main__":
    RunProcess('./Images/image1.jpg')
