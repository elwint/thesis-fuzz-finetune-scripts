import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Start the fine-tuning process
response = openai.FineTuningJob.create(
    training_file="file-tTFTZl4L9VN8Y3rjGE6dVMBi",
    validation_file="file-M0xragyw4CFhK2FevedSvE73",
    model="davinci-002",
    hyperparameters={"n_epochs": 1},
    suffix="base-1"
)

print(response)
