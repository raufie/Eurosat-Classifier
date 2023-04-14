# How to use
## Step1: Clone the repository or download it

## Step 2: Install the requirements

`pip install requirements.txt`
Make sure that you are running python 3.7+
I am running python 3.8 right now

## Step 3: Give an image

In the folder you downloaded the repository, replace the test.jpg with the satellite image you want to test

## Step 4: Run the script "classify.py"

just type this in the terminal

`python classify.py`

## Step 5: Use a custom file with a different name

`python classify.py -file path`

For example, there are other images in the folder we can use them too

`python classify.py -file permanentcrop.jpg`

### Conclusion

It was easy to solve this problem from scratch, but I wasn't able to get good accuracy on fine tuning pre trained networks in torch vision, they didn't go above 83%. 

This one scored 97% on the test net and 96% on the validation set... 

Visit the kaggle notebook here

[Kaggle Notebook](https://www.kaggle.com/code/raufie/eurosat-classification)