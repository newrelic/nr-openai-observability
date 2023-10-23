import boto3
import json 
import newrelic.agent
import openai
import os
import sys


# For this example to work, you should set the following environment variables
# to values that work for your specific environment.
#
#   NEW_RELIC_APP_NAME 
#   NEW_RELIC_LICENSE_KEY
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   OPENAI_API_KEY
#   


@newrelic.agent.background_task()
@newrelic.agent.function_trace(name="openai")
def runOpenAI(prompt, user_request):
    # change to 'gpt-4' if you have access to it 
    model="gpt-3.5-turbo" 
    
    print(f"Running OpenAI with model {model}")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": user_request,
            }
        ]
    )

    print("Response from ChatGPT\n")
    print(response['choices'][0]['message']['content'])
    print("\n\n")


@newrelic.agent.background_task()
@newrelic.agent.function_trace(name="bedrock")
def runBedrock(prompt, user_request):
    """
    Run a query with AWS Bedrock using Anthropic Claude v2.
    """
    bedrock_runtime = boto3.client('bedrock-runtime', 'us-east-1')
    full_prompt = f'{prompt}\n{user_request}'
    prompt_data = f"Human: ${full_prompt}\n\nAssistant:"
    body = json.dumps({"prompt": prompt_data, "max_tokens_to_sample": 500})
    modelId = "anthropic.claude-v2"
    accept = "application/json"
    contentType = "application/json"

    print(f"Running Bedrock with model {modelId}")

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print("Response from Bedrock")
    print(response_body.get("completion"))
    print("\n\n")


if __name__ == "__main__":
    app_name = os.getenv('NEW_RELIC_APP_NAME')
    if not app_name:
        print("You must set the NEW_RELIC_APP_NAME environment variable.")
        exit(1)

    # Enable New Relic Python agent
    newrelic.agent.initialize()
    newrelic.agent.register_application(name=app_name, timeout=10)

    # Enable New Relic observability for LLMs
    from nr_openai_observability import monitor
    monitor.initialization(app_name)

    # prompt is credited to https://prompts.chat/#act-as-a-math-teacher
    prompt = """I want you to act as a math teacher. 
    I will provide some mathematical equations or concepts, 
    and it will be your job to explain them in easy-to-understand terms. 
    This could include providing step-by-step instructions for solving a problem, 
    demonstrating various techniques with visuals or suggesting online 
    resources for further study."""
    user_request = 'My first request is “I need help understanding how probability works."' 

    runOpenAI(prompt, user_request)
    runBedrock(prompt, user_request)

    # Allow the New Relic agent to send final messages as part of shutdown
    # The agent by default can send data up to a minute later
    newrelic.agent.shutdown_agent(60)
