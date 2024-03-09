import gradio as gr
from transformers import pipeline


# Code Comment: Sentiment Analysis Model Selection

'''
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer-based 
natural language processing model that captures contextualized word representations by considering 
bidirectional context in input sequences, enabling its effectiveness across a wide range of downstream 
tasks.ep in mind that both models have their strengths and trade-offs. Assess your specific requirements, 
available resources, and desired level of analysis to choose the most appropriate model for your sentiment 
analysis tasks.


FinBERT is a BERT model pre-trained on financial communication text created to enhance financial NLP research and practice.

finbert-tone model is the FinBERT model fine-tuned on 10,000 manually annotated (positive, negative, neutral) sentences from analyst reports.

distilroberta-finetuned-financial-news-sentiment-analysis is a distilled version of RoBERTa fine-tuned for sentiment analysis on financial news. 
It is smaller and faster but will likely sacrifice some accurancy. 

'''

# End of Code Comment


def sentiment_analysis_generate_text(text):
    # to use the larger model
    model_name = "yiyanghkust/finbert-tone"

    # to use the distilled version 
    # model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

    # Create the pipeline, this defines the task and sets the model
    nlp = pipeline("sentiment-analysis", model=model_name)
    
    # Split the input text into individual sentences
    sentences = text.split('|')
    # Create a list of outputs and fill with output from the pipline
    results = nlp(sentences)
    output = []
    # iterate over a pair of inputs and outputs to structure the output
    for sentence, result in zip(sentences, results):
        output.append(f"Text: {sentence.strip()}\nSentiment: {result['label']}, Score: {result['score']:.4f}\n")

    # Join the results into a single string 
    return "\n".join(output)


def sentiment_analysis_generate_table(text):
    # Define the model
    # to use the larger model
    model_name = "yiyanghkust/finbert-tone"

    # to use the distilled version 
    # model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

    # pipeline
    nlp = pipeline("sentiment-analysis", model=model_name)
    # Split the input text into individual sentences
    sentences = text.split('|')

    # Generate the HTML table with enhanced colors and bold headers
    html = """
    <html>
    <head>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/css/bootstrap.min.css">
    <style>
    .label {
        transition: .15s;
        border-radius: 8px;
        padding: 5px 10px;
        font-size: 14px;
        text-transform: uppercase;
    }
    .positive {
        background-color: rgb(54, 176, 75);
        color: white;
    }
    .negative {
        background-color: rgb(237, 83, 80);
        color: white;
    }
    .neutral {
        background-color: rgb(52, 152, 219);
        color: white;
    }
    th {
        font-weight: bold;
        color: rgb(106, 38, 198);
    }
    </style>
    </head>
    <body>
    <table class="table table-striped">
    <thead>
        <tr>
            <th scope="col">Text</th>
            <th scope="col">Score</th>
            <th scope="col">Sentiment</th>
        </tr>
    </thead>
    <tbody>
    """
    for sentence in sentences:
        result = nlp(sentence.strip())[0]
        text = sentence.strip()
        score = f"{result['score']:.4f}"
        sentiment = result['label']

        # Determine the sentiment class
        if sentiment == "Positive":
            sentiment_class = "positive"
        elif sentiment == "Negative":
            sentiment_class = "negative"
        else:
            sentiment_class = "neutral"

        # Generate table rows
        html += f'<tr><td>{text}</td><td>{score}</td><td><span class="label {sentiment_class}">{sentiment}</span></td></tr>'

    html += """
    </tbody>
    </table>
    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    # uncomment below code for using the code in text results
    # iface = gr.Interface(
    #     fn=sentiment_analysis_generate_text, 
    #     inputs="text", 
    #     outputs="text", 
    #     title="Financial Sentiment Analysis",
    #     description="<p>A sentiment analysis model fine-tuned on financial news.</p>"
    #                 "<p>Enter some financial text to see whether the sentiment is positive, neutral or negative.</p>"
    #                 "<p><strong>Note:</strong> Separate multiple sentences with a '|'.",
    #     )

    # generate the result in html format
    iface = gr.Interface(
        sentiment_analysis_generate_table,
        gr.Textbox(placeholder="Enter input here..."),
        ["html"],
        title="Financial Sentiment Analysis Demo",
        description="<p>Enter some financial related text below. </p>"
                    "<p>Separate multiple input sentences with a '|' if desired. </p>",
        examples=[
            ['The company reported robust earnings, and the cash flow is excellent.'],
            ['We anticipate a decline in revenue due to market uncertainties.'],
            ['The company is located in California'],
            ['The company reported robust earnings, and the cash flow is excellent. |  We anticipate a decline in revenue due to market uncertainties. | The company is located in California '],
        ],
        allow_flagging=False,
        examples_per_page=4,
    )

    iface.launch()  