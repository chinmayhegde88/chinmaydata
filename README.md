# Amazon Fine Food Reviews: Sentiment Analysis (Deep Learning / NLP)

A binary sentiment classifier built with TensorFlow/Keras on the Amazon Fine Food Reviews dataset. The project compares a SimpleRNN baseline with LSTM models, tunes hyperparameters with a random search, and reaches **88% accuracy on 10,000 held-out reviews**.

## Results

| Model | Validation accuracy | Notes |
|---|---|---|
| SimpleRNN baseline (128-dim embedding, 64 units) | about 53% | Barely better than chance in this setup |
| LSTM baseline (128-dim embedding, 64 units) | unstable, 56% to 74% across epochs | Validation accuracy fluctuated between epochs |
| **Tuned LSTM** (best of 8 random-search trials) | **88.4%** | Embedding 16, 128 units, dropout 0.1, learning rate 0.01 |

**Final test accuracy: 88%** on 10,000 held-out reviews (balanced classes).

## Dataset

- Source: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (Kaggle, `snap/amazon-fine-food-reviews`), 568,454 reviews.
- Rating distribution is heavily skewed: about 64% are 5-star reviews and about 9% are 1-star.

## Approach

1. **Labels:** removed 3-star reviews as ambiguous (525,814 left). Ratings of 4 to 5 became positive and ratings of 1 to 2 became negative.
2. **Class imbalance:** the data had 443,777 positive and 82,037 negative reviews. I downsampled the positive class to 82,037 so both classes were equal.
3. **Sampling:** drew a 50,000-review sample from the balanced set to keep training time manageable.
4. **Text cleaning:** lowercased text, removed HTML tags and normalized whitespace.
5. **Tokenization:** Keras `Tokenizer` with a 10,000-word vocabulary and an out-of-vocabulary token; sequences padded or truncated to 100 tokens.
6. **Split:** stratified 80/20 train/test split, with 20% of the training data held out for validation.
7. **Models:** compared a SimpleRNN and an LSTM, each with an embedding layer, dropout and a dense head.
8. **Tuning:** random search over 8 trials (model type, embedding size, units, dropout rate, learning rate), 3 epochs each, chosen by validation accuracy.
9. **Final training:** retrained the best configuration for up to 10 epochs with `EarlyStopping` (on validation loss) and `ModelCheckpoint` (on validation accuracy).
10. **Inference:** saved the model and tokenizer, and wrote a `predict_sentiment()` helper. Example: "great food, amazing quality can do better but for now i am happy" is predicted Positive with probability 0.99.

## Tech stack

Python, TensorFlow/Keras, pandas, NumPy, scikit-learn, Matplotlib, KaggleHub.

## How to run

1. Open `Sentiment_Analysis.ipynb` in Google Colab (where it was developed) or in a local Jupyter environment.
2. Install the dependencies if needed: `pip install tensorflow pandas numpy scikit-learn matplotlib kagglehub`.
3. Run all cells. The notebook downloads the dataset through `kagglehub`.
4. Running the notebook creates `best_amazon_sentiment_model.h5` and `amazon_sentiment_tokenizer.pkl`. These files are not stored in this repository.

## Limitations

- **Results vary between runs.** The train/test split and sampling have no fixed random seed, so exact numbers will differ slightly each time.
- **Only accuracy is reported.** The test set is balanced, so accuracy is meaningful, but precision, recall and a confusion matrix would give a fuller picture.
- **Balanced test set.** Real reviews are about 84% positive after removing 3-star reviews, so performance on natural data may look different.
- **Small search.** Only 8 random-search trials with 3 epochs each, selected by validation accuracy.
- **Punctuation is not removed.** The cleaning step has a regex bug (missing brackets), so only HTML tags and extra whitespace are actually removed.
- **Star ratings as labels.** Ratings are a proxy for sentiment, and the model cannot predict a neutral class.
- **Subset used.** Training used a 50,000-review sample of the 164,074 balanced reviews.

## Possible improvements

- Fix the cleaning regex, set random seeds and report precision, recall and F1.
- Evaluate on the natural (imbalanced) distribution as well as the balanced one.
- Try pretrained embeddings (GloVe) or a transformer model such as DistilBERT.
- Deploy the saved model as a small API.

## Author

Chinmay Hegde, M.Sc. Statistics, University of Mysore.
[LinkedIn](https://www.linkedin.com/in/chinmaych) | [chinmayhegde88@gmail.com]
