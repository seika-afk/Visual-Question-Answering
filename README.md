# Visual Question Answering (VQA) Project

This project implements a simple **Visual Question Answering (VQA)** pipeline using images and text questions to predict answers.

## Project Overview

- **Dataset**: Contains images and corresponding questions/answers.
- **Goal**: Train a model that can answer questions about images (multi-choice answers).
- **Framework**: TensorFlow / Keras.
- **Output**: Predicted answer for a given image-question pair.

## Folder Structure (From Google collab)/after training

data/
  ├── train/
  │   ├── images/
  │   └── questions.json
  ├── test/
  │   ├── images/
  │   └── questions.json
  └── answers.txt
files/
  └── model.h5

## Dependencies

pip install numpy opencv-python tensorflow scikit-learn

## Usage

1. **Load Dataset**
```
train_Q, train_A, train_I = get_data(dataset_path, train=True)  
test_Q, test_A, test_I = get_data(dataset_path, train=False)  
mcq_answers_choices = get_answers_labels(dataset_path)
```
2. **Split Data**
```
from sklearn.model_selection import train_test_split  
trainQ, valQ, trainA, valA, trainI, valI = train_test_split(  
    train_Q, train_A, train_I, test_size=0.2, random_state=42  
)
```
3. **Build Model**
```
model = build_model(image_shape=(64,64,3), vocab_size=len(tokenizer.word_index)+1, num_answers=len(mcq_answers_choices))  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
4. **Prepare Dataset Pipeline**
```
ds = DatasetPipe(tokenizer, mcq_answers_choices, 64, 64)  
train_ds = ds.tf_dataset(trainQ, trainA, trainI, batch_size=32)  
val_ds = ds.tf_dataset(valQ, valA, valI, batch_size=32)
```
5. **Train Model**
```
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
```
**Callbacks**:

- ModelCheckpoint – save best model.
- ReduceLROnPlateau – reduce learning rate on plateau.
- EarlyStopping – stop if no improvement.
- CSVLogger – log training metrics.

## Key Components

1. **DatasetPipe**: Handles tokenization, image preprocessing, and TF dataset creation.  
2. **Model**: CNN for images + dense layers for question embedding.  
3. **Training**: Uses categorical crossentropy for multi-class answer prediction.

## Notes

- Images are resized to (64,64,3) and normalized.  
- Questions are tokenized into a vector of size equal to the vocabulary.  
- Answers are one-hot encoded according to answers.txt.
