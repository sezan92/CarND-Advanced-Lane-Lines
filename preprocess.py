import cv2

import util


class PerspectiveTransformer:
    def __init__(self):
        self.pts = []

    def set_config(self, img):
        self.img = img
        cv2.imshow("original", self.img)
        cv2.setMouseCallback("original", self._click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.pts

    def transform(self, img):
        pass

    def _click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pts) < 4:
                self.pts.append((x, y))
                self.img = cv2.circle(
                    self.img, (x, y), radius=5, color=(0, 0, 255), thickness=-1
                )
                cv2.imshow("original", self.img)
            else:
                print("got all points!")
                print(self.pts)
