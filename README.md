# Arabic Speech Recognition Model

This repository contains code for an Arabic Speech Recognition model developed for the Arabic Egyptian ASR competition.

## Project Overview

This project implements an end-to-end speech recognition system for Arabic, specifically tailored for the Egyptian dialect. The system combines an acoustic model and a language model to transcribe speech into text.

## Key Features

- Acoustic model using a CNN-LSTM architecture
- Separate LSTM-based language model to improve prediction accuracy
- Custom CTC (Connectionist Temporal Classification) loss function
- Data augmentation techniques for improved robustness
- Integration with a pronunciation lexicon for post-processing

## Model Architecture

### Acoustic Model
- Input: MFCC features (13 coefficients)
- Architecture:
  - Conv1D layer (32 filters, kernel size 3)
  - BatchNormalization
  - Conv1D layer (64 filters, kernel size 3)
  - BatchNormalization
  - Bidirectional LSTM (128 units)
  - Bidirectional LSTM (128 units)
  - TimeDistributed Dense layer
  - Softmax activation

### Language Model
- Input: Character sequences
- Architecture:
  - Embedding layer
  - LSTM layer (128 units)
  - TimeDistributed Dense layer with softmax activation

## Data Preprocessing

- MFCC feature extraction (13 coefficients) using Librosa
- Padding/truncation of audio features to fixed length
- Character-level tokenization for transcripts

## Training Process

- Custom data generator with on-the-fly augmentation (time stretching and noise addition)
- #Transition to Conformer-CTC loss optimization
  Connectionist Temporal Classification (CTC) is a way to get around not knowing the alignment between the input and the output.
   CTC works by summing over the probability of all possible alignments between the two. We need to understand what these alignments are 
   in order to understand how the loss function is ultimately calculated.
   ![alt text]([/image.jpg](https://distill.pub/2017/ctc/assets/ctc_cost.svg)?raw=true)
- Early stopping and learning rate reduction strategies

## Inference

- CTC beam search decoding
- Language model integration for improved predictions
- Lexicon-based post-processing for enhanced accuracy

## Usage

1. Ensure all dependencies are installed
2. Prepare your data in the required format (audio files and transcripts)
3. Adjust paths and hyperparameters in the script as needed
4. Run the training script to train the models
5. Use the trained models for prediction on new audio files

## Dependencies

- TensorFlow 2.x
- Librosa
- Numpy
- Pandas
- JiWER (for WER calculation)
- Tqdm

## Future Improvements

- Experiment with more advanced model architectures (e.g., Transformer-based models)
- Implement more sophisticated data augmentation techniques
- Explore transfer learning from pre-trained models
- Optimize for inference speed

## Acknowledgements

- Thanks to the organizers of the Arabic MTC-competition
- Gratitude to the open-source community for providing essential tools and libraries

