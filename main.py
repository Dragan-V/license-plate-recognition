import cv2
from matplotlib import pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = "car1.png"

image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.show()

canny_edge = cv2.Canny(gray_image, 170, 200)
plt.imshow(cv2.cvtColor(canny_edge, cv2.COLOR_BGR2RGB))
plt.show()

contours, new = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

contour_with_license_plate = None
license_plate = None
x = None
y = None
w = None
h = None

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    if len(approx) == 4: 
        contour_with_license_plate = approx
        x, y, w, h = cv2.boundingRect(contour)
        license_plate = gray_image[y:y + h, x:x + w]
        break
    


if license_plate is None or license_plate.size == 0:
    print("Failed to find license plate")
    exit()

plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
plt.show()

text = pytesseract.image_to_string(license_plate, config='--psm 8')
text = ''.join(e for e in text if e.isalnum()) 

if text:
    print("License Plate:", text)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 225), 3)
else:
    print("No text detected")



with open("text.txt", "w") as file:
    if text:
        file.write(f"License Plate: {text}\n")
    else:
        file.write("No text detected")
        print("No text detected")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
