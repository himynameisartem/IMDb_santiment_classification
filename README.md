# IMDb Sentiment Classification

[English](#english) | [Русский](#русский)

---

<a name="english"></a>
# English

## Table of Contents

- [Overview](#overview)
- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Regularization Techniques](#regularization-techniques)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Overview

This project implements a binary sentiment classification model for IMDb movie reviews using deep learning techniques. The model classifies reviews as either positive or negative sentiment using a Bidirectional LSTM neural network with pre-trained GloVe word embeddings.

## Project Description

The project tackles the classic NLP task of sentiment analysis on the IMDb dataset, which contains 50,000 movie reviews labeled as positive or negative. The solution employs:

- **Pre-trained GloVe embeddings** (100-dimensional vectors) for word representation
- **Bidirectional LSTM** layers to capture contextual information from both directions
- **Two-stage training** approach: first with frozen embeddings, then fine-tuning
- **Advanced regularization** techniques to prevent overfitting

## Features

- Binary sentiment classification (positive/negative)
- Pre-trained word embeddings (GloVe 6B.100d)
- Bidirectional LSTM architecture
- Two-phase training strategy
- Comprehensive regularization to prevent overfitting
- Early stopping and learning rate scheduling
- Visualization of training progress

## Technologies Used

### Deep Learning Framework
- **TensorFlow/Keras** - Neural network framework

### Libraries
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization

### Data Sources
- **IMDb Dataset** - 50,000 movie reviews (25,000 train, 25,000 test)
- **GloVe Embeddings** - Pre-trained 100-dimensional word vectors (Stanford NLP)

## Dataset

- **Training Set**: 20,000 samples
- **Validation Set**: 5,000 samples (split from training)
- **Test Set**: 25,000 samples
- **Vocabulary Size**: 20,000 most frequent words
- **Sequence Length**: 400 tokens (padded)

## Model Architecture

The model consists of the following layers:

1. **Embedding Layer**
   - Input dimension: 20,000 (vocabulary size)
   - Output dimension: 100 (GloVe embedding dimension)
   - Pre-trained weights from GloVe
   - Initially frozen, then fine-tuned

2. **SpatialDropout1D** (0.4)
   - Drops entire embedding vectors during training
   - Prevents overfitting in the embedding space

3. **Bidirectional LSTM** (64 units, return_sequences=True)
   - Captures contextual information from both directions
   - Dropout: 0.3 (input dropout)
   - Recurrent dropout: 0.3 (recurrent dropout)

4. **Dropout** (0.3)
   - Regularization between LSTM layers

5. **Bidirectional LSTM** (64 units, return_sequences=True)
   - Second LSTM layer for deeper feature extraction
   - Dropout: 0.3
   - Recurrent dropout: 0.3

6. **GlobalMaxPooling1D**
   - Extracts the most important features from the sequence

7. **Dense Layer** (128 units, ReLU activation)
   - L2 regularization (1e-4)
   - Fully connected layer for classification

8. **Dropout** (0.6)
   - High dropout rate to prevent overfitting

9. **Output Layer** (1 unit, Sigmoid activation)
   - Binary classification output (0 = negative, 1 = positive)

**Total Parameters**: ~1.5M trainable parameters

## Training Strategy

The model uses a **two-stage training approach**:

### Stage 1: Frozen Embeddings
- **Learning Rate**: 1e-3
- **Epochs**: 5 (with early stopping)
- **Batch Size**: 64
- **Embedding Layer**: Frozen (trainable=False)
- **Purpose**: Learn to utilize pre-trained GloVe embeddings effectively

### Stage 2: Fine-tuning
- **Learning Rate**: 5e-4 (reduced for fine-tuning)
- **Epochs**: 8 (with early stopping)
- **Batch Size**: 64
- **Embedding Layer**: Unfrozen (trainable=True)
- **Purpose**: Adapt embeddings to the specific task

### Callbacks

- **EarlyStopping**:
  - Monitor: `val_loss`
  - Patience: 4 epochs
  - Restore best weights: True
  - Min delta: 0.0001

- **ReduceLROnPlateau**:
  - Monitor: `val_loss`
  - Factor: 0.5 (reduce LR by half)
  - Patience: 2 epochs
  - Min delta: 0.0001

## Regularization Techniques

To prevent overfitting, the following regularization techniques are employed:

1. **SpatialDropout1D** (0.4) - Regularizes embedding layer
2. **LSTM Dropout** (0.3) - Input dropout in LSTM layers
3. **LSTM Recurrent Dropout** (0.3) - Recurrent dropout in LSTM layers
4. **Dropout between layers** (0.3) - Additional regularization
5. **Dense Layer Dropout** (0.6) - High dropout in fully connected layer
6. **L2 Regularization** (1e-4) - Weight decay in dense layer
7. **Lower Learning Rates** - More conservative optimization
8. **Early Stopping** - Prevents overtraining

## Results

- **Validation Accuracy**: 90.28%
- **Test Accuracy**: 89.87%

The model demonstrates good generalization with stable training loss that doesn't increase on validation set, indicating successful overfitting prevention.

## Installation

### Prerequisites

```bash
pip install tensorflow numpy matplotlib
```

### Download GloVe Embeddings

The GloVe embeddings will be automatically downloaded when running the notebook, or you can download manually:

```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
```

## Usage

1. Open the Jupyter notebook `Untitled.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Download and preprocess the IMDb dataset
   - Download GloVe embeddings (if not present)
   - Create embedding matrix
   - Build and train the model
   - Evaluate on test set
   - Display training curves
  
     
---

<a name="русский"></a>
# Русский

## Содержание

- [Обзор](#обзор)
- [Описание проекта](#описание-проекта)
- [Возможности](#возможности)
- [Используемые технологии](#используемые-технологии)
- [Датасет](#датасет)
- [Архитектура модели](#архитектура-модели)
- [Стратегия обучения](#стратегия-обучения)
- [Техники регуляризации](#техники-регуляризации)
- [Результаты](#результаты)
- [Установка](#установка)
- [Использование](#использование)

## Обзор

Этот проект реализует модель бинарной классификации отзывов IMDb о фильмах с использованием методов глубокого обучения. Модель классифицирует отзывы как положительные или отрицательные, используя двунаправленную LSTM нейронную сеть с предобученными GloVe эмбеддингами слов.

## Описание проекта

Проект решает классическую задачу NLP - анализ на датасете IMDb, который содержит 50,000 отзывов о фильмах, помеченных как положительные или отрицательные. Решение использует:

- **Предобученные GloVe эмбеддинги** (100-мерные векторы) для представления слов
- **Двунаправленные LSTM** слои для захвата контекстной информации в обоих направлениях
- **Двухэтапное обучение**: сначала с замороженными эмбеддингами, затем дообучение
- **Продвинутые техники регуляризации** для предотвращения переобучения

## Возможности

- Бинарная классификация отзывов (положительная/отрицательная)
- Предобученные эмбеддинги слов (GloVe 6B.100d)
- Архитектура на основе двунаправленных LSTM
- Двухфазная стратегия обучения
- Комплексная регуляризация для предотвращения переобучения
- Ранняя остановка и планирование скорости обучения
- Визуализация процесса обучения

## Используемые технологии

### Фреймворк глубокого обучения
- **TensorFlow/Keras** - Фреймворк для нейронных сетей

### Библиотеки
- **NumPy** - Численные вычисления
- **Matplotlib** - Визуализация данных

### Источники данных
- **IMDb Dataset** - 50,000 отзывов о фильмах (25,000 для обучения, 25,000 для теста)
- **GloVe Embeddings** - Предобученные 100-мерные векторы слов (Stanford NLP)

## Датасет

- **Обучающая выборка**: 20,000 примеров
- **Валидационная выборка**: 5,000 примеров (выделена из обучающей)
- **Тестовая выборка**: 25,000 примеров
- **Размер словаря**: 20,000 самых частых слов
- **Длина последовательности**: 400 токенов (с паддингом)

## Архитектура модели

Модель состоит из следующих слоев:

1. **Слой Embedding**
   - Размерность входа: 20,000 (размер словаря)
   - Размерность выхода: 100 (размерность GloVe эмбеддингов)
   - Предобученные веса из GloVe
   - Изначально заморожен, затем дообучается

2. **SpatialDropout1D** (0.4)
   - Удаляет целые векторы эмбеддингов во время обучения
   - Предотвращает переобучение в пространстве эмбеддингов

3. **Bidirectional LSTM** (64 единицы, return_sequences=True)
   - Захватывает контекстную информацию в обоих направлениях
   - Dropout: 0.3 (input dropout)
   - Recurrent dropout: 0.3 (рекуррентный dropout)

4. **Dropout** (0.3)
   - Регуляризация между LSTM слоями

5. **Bidirectional LSTM** (64 единицы, return_sequences=True)
   - Второй LSTM слой для более глубокой экстракции признаков
   - Dropout: 0.3
   - Recurrent dropout: 0.3

6. **GlobalMaxPooling1D**
   - Извлекает наиболее важные признаки из последовательности

7. **Полносвязный слой** (128 единиц, активация ReLU)
   - L2 регуляризация (1e-4)
   - Полносвязный слой для классификации

8. **Dropout** (0.6)
   - Высокий уровень dropout для предотвращения переобучения

9. **Выходной слой** (1 единица, активация Sigmoid)
   - Выход бинарной классификации (0 = отрицательный, 1 = положительный)

**Всего параметров**: ~1.5M обучаемых параметров

## Стратегия обучения

Модель использует **двухэтапный подход обучения**:

### Этап 1: Замороженные эмбеддинги
- **Скорость обучения**: 1e-3
- **Эпохи**: 5 (с ранней остановкой)
- **Размер батча**: 64
- **Слой Embedding**: Заморожен (trainable=False)
- **Цель**: Научиться эффективно использовать предобученные GloVe эмбеддинги

### Этап 2: Дообучение
- **Скорость обучения**: 5e-4 (уменьшена для тонкой настройки)
- **Эпохи**: 8 (с ранней остановкой)
- **Размер батча**: 64
- **Слой Embedding**: Разморожен (trainable=True)
- **Цель**: Адаптировать эмбеддинги под конкретную задачу

### Callbacks

- **EarlyStopping**:
  - Мониторинг: `val_loss`
  - Терпение: 4 эпохи
  - Восстановление лучших весов: True
  - Минимальная дельта: 0.0001

- **ReduceLROnPlateau**:
  - Мониторинг: `val_loss`
  - Фактор: 0.5 (уменьшение LR вдвое)
  - Терпение: 2 эпохи
  - Минимальная дельта: 0.0001

## Техники регуляризации

Для предотвращения переобучения используются следующие техники регуляризации:

1. **SpatialDropout1D** (0.4) - Регуляризация слоя эмбеддингов
2. **LSTM Dropout** (0.3) - Входной dropout в LSTM слоях
3. **LSTM Recurrent Dropout** (0.3) - Рекуррентный dropout в LSTM слоях
4. **Dropout между слоями** (0.3) - Дополнительная регуляризация
5. **Dropout в полносвязном слое** (0.6) - Высокий dropout в полносвязном слое
6. **L2 Регуляризация** (1e-4) - Затухание весов в полносвязном слое
7. **Пониженные скорости обучения** - Более консервативная оптимизация
8. **Ранняя остановка** - Предотвращает переобучение

## Результаты

- **Точность на валидации**: 90.28%
- **Точность на тесте**: 89.87%

Модель демонстрирует хорошую генерализацию со стабильной потерей на обучении, которая не растет на валидационной выборке, что указывает на успешное предотвращение переобучения.

## Установка

### Требования

```bash
pip install tensorflow numpy matplotlib
```

### Загрузка GloVe эмбеддингов

GloVe эмбеддинги будут автоматически загружены при запуске ноутбука, или вы можете загрузить их вручную:

```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
```

## Использование

1. Откройте Jupyter ноутбук `Untitled.ipynb`
2. Запустите все ячейки последовательно
3. Ноутбук автоматически:
   - Загрузит и предобработает датасет IMDb
   - Загрузит GloVe эмбеддинги (если они отсутствуют)
   - Создаст матрицу эмбеддингов
   - Построит и обучит модель
   - Оценит на тестовой выборке
   - Отобразит кривые обучения



