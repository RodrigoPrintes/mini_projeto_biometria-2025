import zipfile
import os
import random
import shutil
import subprocess

class SplitDataset:

      @staticmethod
      def download_dataset():

          cDown = "gdown --id 1X2fzTkqwj8svFXhtY0wz6M0wyBdCo4Rd -O  dataset_train_iris_v2.zip"
          subprocess.run(cDown, shell=True)

      @staticmethod
      def download_model():
          #https://drive.google.com/drive/folders/1POQyRfbIVx-IVIoqRDRCAq7QYOQaH4lg?usp=drive_link

          cDown = "gdown --id 1POQyRfbIVx-IVIoqRDRCAq7QYOQaH4lg -O  model.h5"
          subprocess.run(cDown, shell=True)

      @staticmethod
      def unzip_dataset():

         c_unzip = "unzip  dataset_train_iris_v2.zip"
         subprocess.run(c_unzip, shell=True)


      @staticmethod
      def copy_files(root_dir):
          for folder_name, subfolders, filenames in os.walk(root_dir):
              for filename in filenames:
                  if filename.endswith('.bmp'):
                      parent_folder_path = os.path.join(folder_name, os.pardir)
                      file_path = os.path.join(folder_name, filename)
                      new_file_path = os.path.join(parent_folder_path, filename)

                      if os.path.exists(new_file_path):
                          pass
                      else:
                          shutil.copy(file_path, new_file_path)
                          print(f" {filename} copy to {parent_folder_path}")


      @staticmethod
      def refactorDatabase(data_dir):

          folder_path = data_dir

          IDs = [id for id in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, id))]

          data_dict = {}
          for id in IDs:
              files_1 = os.listdir(os.path.join(folder_path, id, '1'))
              files_2 = os.listdir(os.path.join(folder_path, id, '2'))
              combined_files = files_1 + files_2
              data_dict[id] = combined_files



          for id in data_dict.keys():
              data_dict[id] = [f"{data_dir}/{id}/{name_bmp}" for name_bmp in data_dict[id]]


          return data_dict