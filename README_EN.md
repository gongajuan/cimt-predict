
# Carotid Intima-Media Thickness Prediction Project

## Introduction
This project uses binocular fundus images and applies deep learning technology to predict the thickening of the carotid intima-media. Eight different deep learning models have been designed in the project, aiming to explore which model performs best with this type of medical image data.

## Model List
The following are the deep learning models designed in this project:
1. Standard ResNet
2. Parallel ResNet
3. Flatten ResNet
4. Standard ResNext
5. Parallel ResNeXt
6. Flatten ResNeXt
7. Parallel ResNeXt & Age
8. Flatten ResNext & Age

## System Requirements
- **GPU**: At least 2 GPUs
- **CPU**: 16 cores or more
- **Python Version**: Python 3.8 or higher recommended
- **Dependencies**: See `requirements.txt` file

## Installation Guide
1. Clone the repository to your local system:
   ```
   git clone [repository link]
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Training the Models
Each model uses a different `Train.py` method for training. Run the following command to train a specific model:
```
python Train.py --model [model name]
```

## Testing the Models
Use the `test.py` script to test the performance of the models. Run the following command for testing:
```
python test.py --model [model name]
```

### Result


![image-20231223221429967](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231223221429967.png)

### ![image-20231223221411578](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231223221411578.png).

![image-20231223221504003](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231223221504003.png)