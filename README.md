# Comment Informativeness Ranking

A bilingual NLP tool that ranks comments by their **relative informativeness** using TF-IDF.

The project supports both **English and Persian** comments and provides an interactive Gradio interface for testing the ranking algorithm.

## Demo

🚀 **Try the interactive demo:**
[Hugging Face Space](#)

> The demo link will be added after deployment.

## How It Works

The system processes comments according to their language and converts them into TF-IDF vectors.

For English, the preprocessing includes tokenization, stopword removal, and lemmatization.

For Persian, the text is normalized using **Hazm**, followed by tokenization, stopword removal, and lemmatization.

An importance score is then calculated from each comment's TF-IDF vector, and comments are ranked from highest to lowest score.

## Example

Given comments such as:

```text
Great video!
Very helpful explanation.
How can TF-IDF be improved for sentiment classification?
```

The third comment is likely to receive a higher score because it contains more distinctive terms.

The score is **relative to the provided set of comments** and does not represent the actual quality or usefulness of a comment.

## Limitations & Future Work

This project uses TF-IDF as a simple proxy for comment informativeness. Therefore, it does not understand the semantic meaning of comments and does not use labeled training data.

Future improvements could include:

* Human-annotated evaluation data
* Semantic embeddings and transformer-based models
* Additional ranking features
* Better evaluation of ranking quality

## Sample Data

The repository includes sample English and Persian comments for demonstration.

The sample comments are synthetically generated and are intended for testing rather than benchmarking.

## Technologies

* Python
* scikit-learn
* NLTK
* Hazm
* Gradio

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
