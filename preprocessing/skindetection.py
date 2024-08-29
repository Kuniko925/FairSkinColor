import dlib
from imutils import face_utils
import cv2
import matplotlib.pyplot as plt
import numpy as np


class SkinExtraction:
    def __init__(self, filepath):
        self.filepath = filepath
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"

    # Landmark部分を赤で点表示する
    def create_landmark_image(self):
        face_detector = dlib.get_frontal_face_detector()
        face_predictor = dlib.shape_predictor(self.predictor_path)
        img = cv2.imread(self.filepath)
        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(img_gry, 1)
        
        for face in faces:
            
            landmark = face_predictor(img_gry, face)
            landmark = face_utils.shape_to_np(landmark)
        
            for (i, (x, y)) in enumerate(landmark):
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def remove_facial_parts(self):
        face_detector = dlib.get_frontal_face_detector()
        face_predictor = dlib.shape_predictor(self.predictor_path)
        img = cv2.imread(self.filepath)
        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(img_gry, 1)
        
        for face in faces:
            landmark = face_predictor(img_gry, face)
            landmark = face_utils.shape_to_np(landmark)
            
            # Define groups of landmarks for different facial features
            landmarks = {
                "mouth": landmark[48:68],
                "right_eye": landmark[36:42],
                "left_eye": landmark[42:48],
                "right_eyebrow": landmark[17:22],
                "left_eyebrow": landmark[22:27],
                "nose": landmark[27:36],
                "jaw": landmark[0:17]
            }
        
            # Create a mask for blacking out specific facial features
            mask = np.zeros_like(img)
        
            # Fill the regions with black for "right_eye", and "left_eye"
            for feature in ["right_eye", "left_eye"]:
                points = landmarks[feature]
                cv2.fillPoly(mask, [points], (255, 255, 255))
        
            # Apply the mask to the image
            img[mask == 255] = 0
        
            # Create a convex hull for the points
            points = np.concatenate((landmarks["right_eyebrow"], landmarks["left_eyebrow"], landmarks["jaw"]))
            hull = cv2.convexHull(points)
            
            # Create a mask for the convex hull
            hull_mask = np.zeros_like(img)
            cv2.fillConvexPoly(hull_mask, hull, (255, 255, 255))
        
            # Invert the mask to black out the area outside the convex hull
            mask_inv = cv2.bitwise_not(hull_mask)
            img[mask_inv == 255] = 0
        
            # Draw the outer mouth landmarks and fill the inside with black
            outer_mouth_points = np.array([
                landmarks["mouth"][0], landmarks["mouth"][1], landmarks["mouth"][2],
                landmarks["mouth"][3], landmarks["mouth"][4], landmarks["mouth"][5],
                landmarks["mouth"][6], landmarks["mouth"][7], landmarks["mouth"][8],
                landmarks["mouth"][9], landmarks["mouth"][10], landmarks["mouth"][11]
            ])
        
            # Fill the inside of the mouth with black
            cv2.fillPoly(img, [outer_mouth_points], color=(0, 0, 0))
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_rgb

    def extract_skin_pixels(self):
        
        face_detector = dlib.get_frontal_face_detector()
        face_predictor = dlib.shape_predictor(self.predictor_path)
        img = cv2.imread(self.filepath)
        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(img_gry, 1)
        
        for face in faces:
            landmark = face_predictor(img_gry, face)
            landmark = face_utils.shape_to_np(landmark)
            
            # Define groups of landmarks for different facial features
            landmarks = {
                "mouth": landmark[48:68],
                "right_eye": landmark[36:42],
                "left_eye": landmark[42:48],
                "right_eyebrow": landmark[17:22],
                "left_eyebrow": landmark[22:27],
                "nose": landmark[27:36],
                "jaw": landmark[0:17]
            }
        
            # Create a mask for blacking out specific facial features
            mask = np.zeros_like(img)

            # Fill the regions with black for "right_eye", "left_eye", and "mouth"
            for feature in ["right_eye", "left_eye"]:
                points = landmarks[feature]
                cv2.fillPoly(mask, [points], (255, 255, 255))
        
            # Apply the mask to the image
            img[mask == 255] = 0
        
            # Create a convex hull for the points
            points = np.concatenate((landmarks["right_eyebrow"], landmarks["left_eyebrow"], landmarks["jaw"]))
            hull = cv2.convexHull(points)
            
            # Create a mask for the convex hull
            hull_mask = np.zeros_like(img)
            cv2.fillConvexPoly(hull_mask, hull, (255, 255, 255))
        
            # Invert the mask to black out the area outside the convex hull
            mask_inv = cv2.bitwise_not(hull_mask)
            img[mask_inv == 255] = 0
        
            # Draw the outer mouth landmarks and fill the inside with black
            outer_mouth_points = np.array([
                landmarks["mouth"][0], landmarks["mouth"][1], landmarks["mouth"][2],
                landmarks["mouth"][3], landmarks["mouth"][4], landmarks["mouth"][5],
                landmarks["mouth"][6], landmarks["mouth"][7], landmarks["mouth"][8],
                landmarks["mouth"][9], landmarks["mouth"][10], landmarks["mouth"][11]
            ])
        
            # Fill the inside of the mouth with black
            cv2.fillPoly(img, [outer_mouth_points], color=(0, 0, 0))
        
            # Determine the top of the eyes (approximated by the mean of the upper eye landmarks)
            eye_top = int(np.mean([landmarks["right_eye"][1][1], landmarks["left_eye"][1][1]]))
        
            # Create a mask to black out the area from above the eyes to the top
            mask_above_eyes = np.zeros_like(img)
            mask_above_eyes[:eye_top, :] = 255  # Fill from top of image to eye_top with white
        
            # Apply the mask to black out the area from above the eyes to the top
            img[mask_above_eyes == 255] = 0
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_rgb

    def display_process_images(self):
        
        landmark_dot_image = self.create_landmark_image()
        mask_image = self.remove_facial_parts()
        skin_pixels_image = self.extract_skin_pixels()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(landmark_dot_image)
        axes[0].axis("off")
        axes[0].set_title("Landmark")
        axes[1].imshow(mask_image)
        axes[1].axis("off")
        axes[1].set_title("Removed Facial Parts")
        axes[2].imshow(skin_pixels_image)
        axes[2].axis("off")
        axes[2].set_title("Extracted Skin Pixels")
        
        plt.tight_layout()
        plt.show()


class SkinColorDetector:
    def __init__(self, RGB):
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(self.predictor_path)
        self.RGB = RGB
    def check_frontal_face(self):
        img = self.RGB
        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(img_gry, 1)
        if len(faces) > 0:
            return True
        else:
            return False
    def extract_skin_pixels(self):
        img = self.RGB
        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(img_gry, 1)
        
        for face in faces:
            landmark = self.face_predictor(img_gry, face)
            landmark = face_utils.shape_to_np(landmark)
            
            # Define groups of landmarks for different facial features
            landmarks = {
                "mouth": landmark[48:68],
                "right_eye": landmark[36:42],
                "left_eye": landmark[42:48],
                "right_eyebrow": landmark[17:22],
                "left_eyebrow": landmark[22:27],
                "nose": landmark[27:36],
                "jaw": landmark[0:17]
            }
        
            # Create a mask for blacking out specific facial features
            mask = np.zeros_like(img)

            # Fill the regions with black for "right_eye", "left_eye", and "mouth"
            for feature in ["right_eye", "left_eye"]:
                points = landmarks[feature]
                cv2.fillPoly(mask, [points], (255, 255, 255))
        
            # Apply the mask to the image
            img[mask == 255] = 0
        
            # Create a convex hull for the points
            points = np.concatenate((landmarks["right_eyebrow"], landmarks["left_eyebrow"], landmarks["jaw"]))
            hull = cv2.convexHull(points)
            
            # Create a mask for the convex hull
            hull_mask = np.zeros_like(img)
            cv2.fillConvexPoly(hull_mask, hull, (255, 255, 255))
        
            # Invert the mask to black out the area outside the convex hull
            mask_inv = cv2.bitwise_not(hull_mask)
            img[mask_inv == 255] = 0
        
            # Draw the outer mouth landmarks and fill the inside with black
            outer_mouth_points = np.array([
                landmarks["mouth"][0], landmarks["mouth"][1], landmarks["mouth"][2],
                landmarks["mouth"][3], landmarks["mouth"][4], landmarks["mouth"][5],
                landmarks["mouth"][6], landmarks["mouth"][7], landmarks["mouth"][8],
                landmarks["mouth"][9], landmarks["mouth"][10], landmarks["mouth"][11]
            ])
        
            # Fill the inside of the mouth with black
            cv2.fillPoly(img, [outer_mouth_points], color=(0, 0, 0))
        
            # Determine the top of the eyes (approximated by the mean of the upper eye landmarks)
            eye_top = int(np.mean([landmarks["right_eye"][1][1], landmarks["left_eye"][1][1]]))
        
            # Create a mask to black out the area from above the eyes to the top
            mask_above_eyes = np.zeros_like(img)
            mask_above_eyes[:eye_top, :] = 255  # Fill from top of image to eye_top with white
        
            # Apply the mask to black out the area from above the eyes to the top
            img[mask_above_eyes == 255] = 0
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_rgb