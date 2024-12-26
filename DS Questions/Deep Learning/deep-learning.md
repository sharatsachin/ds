# Deep Learning 

## What is Deep Learning?
Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to progressively learn increasingly complex features from data. For example, in image recognition, early layers might learn simple edges and shapes, while deeper layers combine these to recognize complex objects. Deep learning excels at finding patterns in unstructured data like images, text, and audio.

## What is an RNN (recurrent neural network)?
Recurrent Neural Networks are specialized neural networks designed for processing sequential data. Unlike traditional neural networks, RNNs maintain an internal memory state that allows them to incorporate information from previous inputs. This makes them particularly effective for tasks like time series prediction, natural language processing, and speech recognition. The key feature is their ability to process inputs of variable length and maintain context through their hidden state.

## What is a CNN (convolutional neural network)?
Convolutional Neural Networks are a type of deep neural network that is well-suited for image recognition and computer vision tasks. CNNs use a specialized architecture that takes advantage of the spatial structure of image data. They consist of convolutional layers that apply filters to extract features, pooling layers that downsample the data, and fully connected layers that classify the features. CNNs have been highly successful in tasks like image classification, object detection, and image segmentation. 

## How do you work towards a random forest?
Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions. The process involves:
1. Bootstrap sampling - Creating multiple datasets by randomly sampling with replacement
2. Feature randomization - At each split, considering only a random subset of features
3. Building decision trees - Creating multiple uncorrelated trees using these samples
4. Aggregating predictions - Combining predictions through voting (classification) or averaging (regression)
This approach reduces overfitting and improves generalization compared to single decision trees.

## What is a computational graph?
A computational graph is a directed graph that represents a sequence of mathematical operations. In deep learning, it's used to model the flow of computations through a neural network. Nodes represent operations (like matrix multiplication or activation functions), and edges represent the flow of data. This representation is crucial for automatic differentiation and efficient backpropagation during training.

## What are auto-encoders?
Autoencoders are neural networks that learn to compress (encode) data into a lower-dimensional representation and then reconstruct (decode) it back to its original form. The network consists of an encoder that compresses the input into a latent space representation, and a decoder that reconstructs the input from this representation. They're useful for dimensionality reduction, feature learning, and anomaly detection.

## What are Exploding Gradients and Vanishing Gradients?
These are problems that occur during neural network training:
- Vanishing gradients occur when gradients become extremely small as they're propagated back through layers, making it difficult for early layers to learn
- Exploding gradients happen when gradients become extremely large, causing unstable updates
Solutions include gradient clipping, proper initialization, and architectures like LSTM networks that are designed to mitigate these issues.

I'll continue with the remaining questions.

## What is the difference between generative and discriminative models?
Discriminative models learn the boundary between classes by modeling P(y|x) - the probability of a label given the input. They focus on what distinguishes different classes. Examples include logistic regression and neural networks for classification.

Generative models learn the underlying distribution of the data P(x,y) or P(x) - how the data was generated. They can generate new samples and model the actual distribution of each class. Examples include GANs, VAEs, and naive Bayes.

## What is forward and backward propagation in deep learning?
Forward propagation is the process of passing input through the neural network to generate predictions. Data flows from input layer through hidden layers to output layer, applying weights, biases, and activation functions.

Backward propagation (backprop) calculates gradients of the loss function with respect to each weight by applying the chain rule backwards through the network. These gradients are then used to update weights and improve the model's performance.

## Describe the use of Markov models in sequential data analysis?
Markov models are probabilistic models that assume future states depend only on the current state (Markov property). They're used for:
- Time series analysis and prediction
- Natural language processing (n-gram models)
- Speech recognition
- Gene sequence analysis
Hidden Markov Models (HMMs) extend this by including hidden states that generate observable outputs.

## What is generative AI?
Generative AI refers to artificial intelligence systems that can create new content similar to their training data. These systems learn patterns and structures from existing data to generate novel:
- Images (DALL-E, Stable Diffusion)
- Text (GPT models)
- Music
- Code
- Videos
They use various architectures including GANs, transformers, and variational autoencoders.

## What are different neural network architectures used to generate artificial data?
Key architectures for data generation include:
1. GANs (Generative Adversarial Networks)
2. VAEs (Variational Autoencoders)
3. Diffusion Models
4. Autoregressive Models (like GPT)
5. Flow-based Models
Each has unique characteristics: GANs excel at sharp, realistic images; VAEs provide more stable training; diffusion models offer high-quality generation but slower inference.

## What is deep reinforcement learning technique?
Deep reinforcement learning combines deep learning with reinforcement learning to create agents that learn optimal actions through trial and error. It uses deep neural networks to approximate either the value function or policy. Key components include:
- State representation through neural networks
- Policy networks for action selection
- Value networks for state evaluation
- Experience replay for efficient learning
Applications include game playing (AlphaGo), robotics, and autonomous systems.

## What is transfer learning, and how is it applied in deep learning?
Transfer learning is the technique of using knowledge learned from one task to improve performance on another related task. In deep learning, this typically involves:
1. Taking a pre-trained model (e.g., on ImageNet)
2. Freezing early layers (which learn general features)
3. Retraining later layers for the new task
This approach reduces training time and data requirements, especially useful when labeled data is scarce.

## What is the difference between object detection and image segmentation?
Object Detection identifies and localizes objects in images using bounding boxes, providing object class and location.

Image Segmentation assigns each pixel to a specific class or object, creating a pixel-level understanding. It comes in two types:
- Semantic segmentation: Labels each pixel with a class
- Instance segmentation: Distinguishes between different instances of the same class

## Explain the concept of word embeddings in NLP
Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points. They:
- Capture semantic relationships between words
- Enable mathematical operations on words
- Reduce dimensionality compared to one-hot encoding
Popular methods include Word2Vec, GloVe, and FastText.

## What is seq2seq model?
Sequence-to-sequence (seq2seq) models are architectures designed to transform one sequence into another, like translation or summarization. They consist of:
- Encoder: Processes input sequence into a context vector
- Decoder: Generates output sequence from context vector
Often implemented with attention mechanisms to handle long sequences better.

## What are artificial neural networks?
Artificial Neural Networks (ANNs) are computing systems inspired by biological neural networks. They consist of:
- Input layer: Receives raw data
- Hidden layers: Process information through weighted connections
- Output layer: Produces final output
- Activation functions: Add non-linearity
They learn by adjusting weights and biases through backpropagation, forming the foundation of deep learning.