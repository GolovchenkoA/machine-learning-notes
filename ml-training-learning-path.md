# Learning How to Train ML Models: Video Path

A short path for a software engineer with basic AI knowledge: intuition first, then training from scratch, then the day-to-day PyTorch workflow.

## 1. 3Blue1Brown: Neural Networks (chapters 1-4)

- **Platform:** YouTube
- **Length:** ~1 hour
- **Why:** Visual intuition for what a neural network is, how gradient descent works, and what backpropagation does.
- **Link:** https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

## 2. Andrej Karpathy: The spelled-out intro to neural networks and backpropagation (micrograd)

- **Platform:** YouTube
- **Why:** You build a tiny autograd engine in plain Python, so you see exactly what `loss.backward()` does. First video of the "Neural Networks: Zero to Hero" series.
- **Link:** https://www.youtube.com/watch?v=VMj-3S1tku0

## 3. Andrej Karpathy: Building makemore Part 2: MLP

- **Platform:** YouTube
- **Why:** Your first real training loop: batches, learning rate, train/validation split, overfitting.
- **Link:** https://www.youtube.com/watch?v=TCH_1BHY58I
- **Next:** Part 3 (activations, gradients, BatchNorm) explains why training sometimes fails: https://www.youtube.com/watch?v=P6sfmUTpUmc

## 4. freeCodeCamp: PyTorch for Deep Learning & Machine Learning - Full Course

- **Platform:** YouTube
- **Why:** Code-along course covering the standard PyTorch workflow: data, model, loss, optimizer, training, evaluation, saving models.
- **Link:** Search the title on YouTube and pick the freeCodeCamp upload.

## 5. LinkedIn Learning: PyTorch Essential Training: Deep Learning

- **Instructor:** Terezija Semenski
- **Platform:** LinkedIn Learning
- **Length:** ~1h 22m
- **Why:** Short, structured recap of the PyTorch workflow with hands-on coding challenges.
- **Link:** https://www.linkedin.com/learning/pytorch-essential-training-deep-learning-23753149

## Bonus

**Andrej Karpathy: Let's reproduce GPT-2 (124M)**: training a real language model end to end.
https://www.youtube.com/watch?v=l8pRSuU81PU

## Suggested order

1. Watch videos 1-3 first.
2. Work through 4 and 5 while building your own small project (e.g. an MNIST or CIFAR-10 image classifier).
3. Move on to the bonus video when you're comfortable.
