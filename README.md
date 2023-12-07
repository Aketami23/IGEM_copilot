# iGEM tokyo tech LLM

I created a chat bot based on chatGPT that is familiar with iGEM parts and named it "iGEM copilot".To make it easier for beginners to enter iGEM, participants will be able to efficiently select PARTS using natural language search on a problem-based basis.
This is a 99% modified version of the Scrapbox ChatGPT Connector.
    Visit https://scrapbox.io/villagepump/Scrapbox_ChatGPT_Connector


## How to install

Clone the GitHub repository.

Run the following commands to install the required libraries.

$ pip install -r requirements.txt

## How to use
Obtain an OpenAI API token and save it in an .env file.

```
 OPENAI_API_KEY=sk-...
```

Make index.

$ python vector_store.py

It outputs like below:

code::
 % python vector_store.py
  97%|███████████████████████████████████████████████████████████████████████████████████████████████████▉ | 846/872 [07:06<00:10, 2.59 It/s]The server is currently overloaded with other requests. Sorry about that! You can retry your request, or contact us through our help center at help. openai.com if the error persists.
 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 872/872 [07:45<00:00, 1 .87it/s] 

Ask. 

$ python ask_parts.py

It outputs like below:

```
>>>> What is the most important question?
> The most important question is to know ourselves.
```

License
The Scrapbox ChatGPT Connector is distributed under the MIT License. See the LICENSE file for more information.

how to use
    ask_parts.py
    >>Embedded in parts to answer questions

    ask_query_project.py
    >>Embedded in project to answer questions

    looking_parts.py
    >>Part search by cos similarity

    vector_store.py
    >>Create a vector index