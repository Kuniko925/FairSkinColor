import numpy as np
import pandas as pd
import os
import PIL
from PIL import Image
import cv2
from skindetection import SkinColorDetector
from sklearn.model_selection import train_test_split

def check_non_frontal_face_image(dataset):

        non_frontal_index = []
        
        for _, row in dataset.iterrows():
            
            img = cv2.imread(row["filepath"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detector = SkinColorDetector(img)
    
            if detector.check_frontal_face() == False:
                #print(_)
                non_frontal_index.append(_)
        
        print(f"Number of non frontal image: {len(non_frontal_index)}")
    
        return non_frontal_index

def split_three_dataset(df):
    column = "labels"
    df_train, df_dummy = train_test_split(df, train_size=0.6, shuffle=True, random_state=42, stratify=df[column])
    df_valid, df_test = train_test_split(df_dummy, train_size=0.5, shuffle=True, random_state=42, stratify=df_dummy[column])
    
    print('Final sizes - train:', len(df_train), 'validation:', len(df_valid), 'test:', len(df_test))
    print("---train-------------")
    print(df_train.groupby("labels").size())
    print("---valid-------------")
    print(df_valid.groupby("labels").size())
    print("---test-------------")
    print(df_test.groupby("labels").size())
    
    return df_train, df_valid, df_test

class CelebA:
    def __init__(self, root, df):
        self.root = root
        self.df = df
        self.groupsize = ""
    def unify_dataset(self):
        
        # Rename columns names
        self.df.loc[self.df["Smiling"] == 1, "labels"] = "1"
        self.df.loc[self.df["Smiling"] == -1, "labels"] = "0"
        self.df.loc[self.df["Pale_Skin"] == 1, "skin tone"] = "1"
        self.df.loc[self.df["Pale_Skin"] == -1, "skin tone"] = "2"
        
        self.df["filepath"] = self.root + "data/" + self.df["image_id"]
        self.df["filename"] = self.df["image_id"]
        self.df["image_id"] = self.df["image_id"].str.replace(".jpg", "")

    def check_balance(self):
        
        self.groupsize = self.df.groupby(["labels", "skin tone"]).size()
        print(self.groupsize)
        print(self.groupsize.min())

    def balance_dataset(self):
        
        tmp1 = self.df[(self.df["labels"] == "0") & (self.df["skin tone"] == "1")].sample(n=self.groupsize.min(), random_state=42)
        tmp2 = self.df[(self.df["labels"] == "0") & (self.df["skin tone"] == "2")].sample(n=self.groupsize.min(), random_state=42)
        tmp3 = self.df[(self.df["labels"] == "1") & (self.df["skin tone"] == "1")].sample(n=self.groupsize.min(), random_state=42)
        tmp4 = self.df[(self.df["labels"] == "1") & (self.df["skin tone"] == "2")].sample(n=self.groupsize.min(), random_state=42)
        df_balanced = pd.concat([tmp1, tmp2, tmp3, tmp4])

        return df_balanced

    def split_dataset(self, df_balanced):
        return split_three_dataset(df_balanced)
        
    def check_non_frontal_face(self, dataset):
        return check_non_frontal_face_image(dataset)

    def create_masked_image(self, dataset, save_directory):
        
        dataset["masked filepath"] = save_directory + dataset["filename"]
        
        for i, d in dataset.iterrows():
            filepath = d["filepath"]
            maskpath = d["masked filepath"]
            
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detector = SkinColorDetector(image)
            skin_image = detector.extract_skin_pixels()
            cv2.imwrite(maskpath, skin_image)

            print(i)

        return dataset

    # Remove all bakc photos
    def check_no_skin_indetified(self, dataset):

        blacks = []
        for i, d in dataset.iterrows():

            maskpath = d["masked filepath"]
            image = cv2.imread(maskpath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_all_black = np.all(image == [0, 0, 0])

            if is_all_black:
                blacks.append(i)
                print(f"All black: {maskpath}")
                
        return blacks


class UTKFace:
    def __init__(self, root):
        
        self.root = root
        self.df = ""
        self.groupsize = ""

    def create_dataframe(self):
        
        folder_path = self.root + "data"
        file_names = os.listdir(folder_path)
        file_names = [f for f in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, f)) and f.count("_") == 3]
        
        df = pd.DataFrame()
        df["filename"] = file_names
        df["age"] = df["filename"].str.split("_").str.get(0)
        df["gender"] = df["filename"].str.split("_").str.get(1)
        df["race"] = df["filename"].str.split("_").str.get(2)
        df["labels"] = df["gender"].astype("str")
        df["filepath"] = self.root + "data/" + df["filename"]
        df["image_id"] = df["filename"].str.split(".chip").str.get(0) + ".chip"
        df["skin tone"] = df["race"]
        
        # To remove others because it can not be regarded as the group has the same attribute
        index = df[df["race"] == "4"].index
        df.drop(index=index, inplace=True)

        self.df = df

    def check_balance(self):

        self.groupsize = self.df.groupby(["labels", "skin tone"]).size()
        print(self.groupsize)
        print(self.groupsize.min())

    def balance_dataset(self):
        
        tmp1 = self.df[(self.df["labels"] == "0") & (self.df["skin tone"] == "0")].sample(n=self.groupsize.min(), random_state=42)
        tmp2 = self.df[(self.df["labels"] == "0") & (self.df["skin tone"] == "1")].sample(n=self.groupsize.min(), random_state=42)
        tmp3 = self.df[(self.df["labels"] == "0") & (self.df["skin tone"] == "2")].sample(n=self.groupsize.min(), random_state=42)
        tmp4 = self.df[(self.df["labels"] == "0") & (self.df["skin tone"] == "3")].sample(n=self.groupsize.min(), random_state=42)
        tmp5 = self.df[(self.df["labels"] == "1") & (self.df["skin tone"] == "0")].sample(n=self.groupsize.min(), random_state=42)
        tmp6 = self.df[(self.df["labels"] == "1") & (self.df["skin tone"] == "1")].sample(n=self.groupsize.min(), random_state=42)
        tmp7 = self.df[(self.df["labels"] == "1") & (self.df["skin tone"] == "2")].sample(n=self.groupsize.min(), random_state=42)
        tmp8 = self.df[(self.df["labels"] == "1") & (self.df["skin tone"] == "3")].sample(n=self.groupsize.min(), random_state=42)
        df_balanced = pd.concat([tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8])
        
        return df_balanced

    def split_dataset(self, df_balanced):
        return split_three_dataset(df_balanced)

    def check_non_frontal_face(self, dataset):
        return check_non_frontal_face_image(dataset)

    def create_masked_image(self, dataset, save_directory):
        
        dataset["masked filepath"] = save_directory + dataset["filename"]
        
        for i, d in dataset.iterrows():
            filepath = d["filepath"]
            maskpath = d["masked filepath"]
            
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detector = SkinColorDetector(image)
            skin_image = detector.extract_skin_pixels()
            cv2.imwrite(maskpath, skin_image)

            #print(i)

        return dataset


        

class HAM10000:
    def __init__(self, root, df):
        self.root = root
        self.df = df
        self.groupsize = ""

    def update_dataset(self):
        index = self.df[self.df["dx"].isin(["bkl", "df", "bcc", "vasc", "akiec"])].index
        self.df.drop(index=index, inplace=True)
        self.df["filename"] = self.df["image_id"] + ".jpg"
        self.df["filepath"] = self.root + "data/" + self.df["filename"]
        self.df.loc[self.df["dx"] == "nv", "labels"] = "0"
        self.df.loc[self.df["dx"] == "mel", "labels"] = "1"
        return self.df

    def check_segmentation(self, seg_directory):

        self.df["seg filepath"] = seg_directory + self.df["image_id"] + "_segmentation.png"
        
        no_segmentation = []
        for i, d in self.df.iterrows():
            segpath = d["seg filepath"]
            if os.path.exists(segpath) == False:
                no_segmentation.append(d["image_id"])
                print(segpath)
        
        if len(no_segmentation) != 0:
            index = self.df[self.df["image_id"].isin(no_segmentation)].index
            self.df.drop(index=index, inplace=True)
        
    def create_masked_image(self, save_directory):

        self.df["masked filepath"] = save_directory + self.df["filename"]
        
        for i, d in self.df.iterrows():
            filepath = d["filepath"]
            maskpath = d["seg filepath"]
        
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        
            if image.shape[:2] != mask.shape[:2]:
                print(filepath)
            else:
                skin = (mask == 0).astype(np.uint8) # 0:black, 1: white -> To change black: 1 means active
                masked_image = cv2.bitwise_and(image, image, mask=skin)
            
                new_filepath = d["masked filepath"]
                masked_image = Image.fromarray(masked_image)
                masked_image.save(new_filepath)

        print("Completed to make masked files")

        return self.df

    def override_dataset(self, df):
        self.df = df
        
    def check_balance(self):

        self.groupsize = self.df.groupby(["labels", "skin tone"]).size()
        print(self.groupsize)
        print(self.groupsize.min())

    def balance_dataset(self):
        
        tmp1 = self.df[(self.df["labels"] == "0") & (self.df["skin tone"] == "1")].sample(n=self.groupsize.min(), random_state=42)
        tmp2 = self.df[(self.df["labels"] == "1") & (self.df["skin tone"] == "1")].sample(n=self.groupsize.min(), random_state=42)
        df_balanced = pd.concat([tmp1, tmp2])

        return df_balanced