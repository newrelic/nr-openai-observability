import boto3
import json 
import newrelic.agent
import os
import sys


# For this example to work, you should set the following environment variables
# to values that work for your specific environment.
#
#   NEW_RELIC_APP_NAME
#   NEW_RELIC_LICENSE_KEY
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#


# When testing changes within the SDK, we need to load the changes from a local
# directory. These lines allow for this. Unless you are testing changes to the
# SDK itself, leave these lines commented out.
# 
# Add vendor directory to module search path
parent_dir = os.path.abspath(os.path.dirname(__file__))
vendor_dir = os.path.join(parent_dir, '../src')

sys.path.append(vendor_dir)
# End adding SDK

@newrelic.agent.background_task()
@newrelic.agent.function_trace(name="titan")
def runTitan(bedrock_runtime):
    # Run a query with Amazon Titan
    prompt_data = """
    Command: Write me a blog about making strong business decisions as a leader.

    Blog:
    """

    body = json.dumps({"inputText": prompt_data})
    # Titan modelIds:
    #  - amazon.titan-text-express-v1
    modelId = "amazon.titan-text-express-v1" 
    accept = "application/json"
    contentType = "application/json"

    print(f'Test with AWS Titan model {modelId}')
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body.get("results")[0].get("outputText"))
    print()


@newrelic.agent.background_task()
@newrelic.agent.function_trace(name="anthropic")
def runAnthropic(bedrock_runtime):
    # Run a query with Anthropic Claude
    prompt_data = """Human: Write me a blog about making strong business decisions as a leader.

    Assistant:
    """

    body = json.dumps({"prompt": prompt_data, "max_tokens_to_sample": 500})
    # Anthropic modelIds: 
    #  - anthropic.claude-instant-v1 (Instant v1.2)
    #  - anthropic.claude-v1         (v1.3)
    #  - anthropic.claude-v2         (v2)
    modelId = "anthropic.claude-instant-v1"
    accept = "application/json"
    contentType = "application/json"

    print(f'Test with Anthropic model {modelId}')
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body.get("completion"))
    print()


@newrelic.agent.background_task()
@newrelic.agent.function_trace(name="ai21")
def runAi21(bedrock_runtime):
    # Run a query with AI21 Jurassic
    prompt_data = """Write me a blog about making strong business decisions as a leader."""

    body = json.dumps({"prompt": prompt_data, "maxTokens": 200})
    # AI21 Jurassic modelIds:
    # - ai21.j2-mid-v1
    # - ai21.j2-ultra-v1
    modelId = "ai21.j2-mid-v1" 
    accept = "application/json"
    contentType = "application/json"

    print(f'Test with AI21 model {modelId}')
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body.get("completions")[0].get("data").get("text"))
    print()


@newrelic.agent.background_task()
@newrelic.agent.function_trace(name="cohere")
def runCohere(bedrock_runtime):
    # Run a query with Cohere
    prompt_data = """Write me a blog about making strong business decisions as a leader."""

    body = json.dumps({"prompt": prompt_data, "max_tokens": 200, "temperature": 0.75})
    # Cohere modelIds:
    # - cohere.command-text-v14
    modelId = "cohere.command-text-v14"
    accept = "application/json"
    contentType = "application/json"

    print(f'Test with Cohere model {modelId}')
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body.get("generations")[0].get("text"))
    print()


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

    # AWS split Bedrock client into two parts. The 'bedrock' client has management
    # functionality while the 'bedrock-runtime' can invoke specific LLMs.
    # Use the following to see what type of models are available.
    #
    # bedrock = boto3.client('bedrock', 'us-east-1')
    # print(json.dumps(bedrock.list_foundation_models(), indent=2))

    bedrock_runtime = boto3.client('bedrock-runtime', 'us-east-1')

    runTitan(bedrock_runtime)
    runAnthropic(bedrock_runtime)
    runAi21(bedrock_runtime)
    runCohere(bedrock_runtime)

    # Allow the New Relic agent to send final messages as part of shutdown
    # The agent by default can send data up to a minute later
    newrelic.agent.shutdown_agent(60)
