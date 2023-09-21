import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Start the fine-tuning process
response = openai.FineTuningJob.create(
    training_file="file-UtPmMrkdiWiBrt1RWXiOQMdO",
    validation_file="file-K5pCXAy6pLdxIknLI2U53g8D",
    model="ft:gpt-3.5-turbo-0613:ultraware:1:80tWpCey",
    hyperparameters={"n_epochs": 1},
    suffix="2"
)

print(response)
