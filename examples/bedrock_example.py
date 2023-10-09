import boto3
import json 
import newrelic.agent


app_name = 'aws-bedrock-sample'

# Uncomment and set if these are not already in your environment
# import os
# os.environ['AWS_ACCESS_KEY_ID'] = '...'
# os.environ['AWS_SECRET_ACCESS_KEY'] = '...'

# When testing changes within the SDK, we need to load the changes from a local
# directory. These lines allow for this.
import os
import sys

# # Add vendor directory to module search path
parent_dir = os.path.abspath(os.path.dirname(__file__))
vendor_dir = os.path.join(parent_dir, '../src')

sys.path.append(vendor_dir)
# End adding SDK

@newrelic.agent.background_task()
def main():
    bedrock = boto3.client('bedrock', 'us-east-1')
    #print(bedrock.list_foundation_models())

    bedrock_runtime = boto3.client('bedrock-runtime', 'us-east-1')

    # Run a query with Amazon Titan
    prompt_data = """
    Command: Write me a blog about making strong business decisions as a leader.

    Blog:
    """

    body = json.dumps({"inputText": prompt_data})
    modelId = "amazon.titan-tg1-large"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    print(f"titan large response: {response}")
    response_body = json.loads(response.get("body").read())
    print(f"titan large body: {response_body}")

    print(response_body.get("results")[0].get("outputText"))

    # Run a query with Anthropic Claude
    # prompt_data = """Human: Write me a blog about making strong business decisions as a leader.

    # Assistant:
    # """

    # body = json.dumps({"prompt": prompt_data, "max_tokens_to_sample": 500})
    # modelId = "anthropic.claude-instant-v1"  # change this to use a different version from the model provider
    # accept = "application/json"
    # contentType = "application/json"

    # response = bedrock_runtime.invoke_model(
    #     body=body, modelId=modelId, accept=accept, contentType=contentType
    # )
    # print(f"claude response: {response}")
    # response_body = json.loads(response.get("body").read())
    # print(f"claude body: {response_body}")

    # print(response_body.get("completion"))


    # Run a query with AI21 Jurassic
    # prompt_data = """Write me a blog about making strong business decisions as a leader."""

    # body = json.dumps({"prompt": prompt_data, "maxTokens": 200})
    # modelId = "ai21.j2-grande-instruct"  # change this to use a different version from the model provider
    # accept = "application/json"
    # contentType = "application/json"

    # response = bedrock_runtime.invoke_model(
    #     body=body, modelId=modelId, accept=accept, contentType=contentType
    # )
    # print(f"jurassic response: {response}")
    # response_body = json.loads(response.get("body").read())
    # print(f"jurassic body: {response_body}")

    # print(response_body.get("completions")[0].get("data").get("text"))


if __name__ == "__main__":
    # Enable New Relic Python agent
    newrelic.agent.initialize()
    newrelic.agent.register_application(name=app_name, timeout=10)

    # Enable New Relic observability for LLMs
    from nr_openai_observability import monitor
    monitor.initialization(app_name)

    main()

    # Allow the New Relic agent to send final messages as part of shutdown
    # The agent by default can send data up to a minute later
    newrelic.agent.shutdown_agent(60)
