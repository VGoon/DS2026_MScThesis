CKA.py implementation credited to https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment 

for google collab
!pip install -r https://raw.githubusercontent.com/username/repo/main/requirements.txt

runtime restart after dependency install
import os
os.kill(os.getpid(), 9)


mounting the google drive to access data
from google.colab import drive
drive.mount('/content/drive')

appears in -> /content/drive/MyDrive/

data_path = "/content/drive/MyDrive/your_large_folder"

import os
os.listdir(data_path)


----


from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/imagenet/imagenet_subset.tar /content/

!tar -xf imagenet_subset.tar -C /content/ImageNetSmall/


!git clone https://github.com/your-username/your-repo.git

import os
os.chdir("your-repo")