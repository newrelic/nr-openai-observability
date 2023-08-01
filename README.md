
[![Community Project header](https://github.com/newrelic/open-source-office/raw/master/examples/categories/images/Community_Project.png)](https://github.com/newrelic/open-source-office/blob/master/examples/categories/index.md#category-community-project)

# OpenAI Observability

A lightweight tool to monitor your OpenAI workload.

## Installation
**With `pip`**

```bash
pip install nr-openai-observability
```

## Getting Started

#### STEP 1: Set Your Environment Variables 
* [Get your License key](https://one.newrelic.com/launcher/api-keys-ui.api-keys-launcher) (also referenced as `ingest - license`) and set it as environment variable: `NEW_RELIC_LICENSE_KEY`.
[Click here](https://docs.newrelic.com/docs/apis/intro-apis/new-relic-api-keys/#license-key) for more details and instructions.

**`Bash`**

```bash
export NEW_RELIC_LICENSE_KEY=<license key>
```

**`Python`**

```python
import os
os.environ["NEW_RELIC_LICENSE_KEY"] = "<license key>"
```
`NEW_RELIC_LICENSE_KEY` can also be sent as a parameter at the `monitor.initialization()`
 call.

* Are you reporting data to the New Relic EU region? click [here](#eu-account-users) for more instructions.

#### STEP 2: Add the following two lines to your code

```python
from nr_openai_observability import monitor
monitor.initialization(
    application_name="OpenAI observability example"
)
```

#### Code example:

```python

import os

import openai
from nr_openai_observability import monitor

monitor.initialization(
    application_name="OpenAI observability example"
)

openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Write a rhythm about observability",
        },
    ],
)
print(response["choices"][0]["message"]["content"])
```

#### STEP 3: Follow the instruction [here](https://one.newrelic.com/launcher/catalog-pack-details.launcher/?pane=eyJuZXJkbGV0SWQiOiJjYXRhbG9nLXBhY2stZGV0YWlscy5jYXRhbG9nLXBhY2stY29udGVudHMiLCJxdWlja3N0YXJ0SWQiOiI1ZGIyNWRiZC1hNmU5LTQ2ZmMtYTcyOC00Njk3ZjY3N2ZiYzYifQ==) to add the dashboard to your New Relic account.

### EU Account Users:

If you are using an EU region account, you should also set your `EVENT_CLIENT_HOST`:

**`Bash`**

```bash
export EVENT_CLIENT_HOST="insights-collector.eu01.nr-data.net"
```

**`Python`**

```python
import os
os.environ["EVENT_CLIENT_HOST"] = "insights-collector.eu01.nr-data.net"
```
    
`EVENT_CLIENT_HOST` can also be sent as a parameter at the `monitor.initialization()`
 call.

## Support

New Relic hosts and moderates an online forum where customers can interact with New Relic employees as well as other customers to get help and share best practices. Like all official New Relic open source projects, there's a related Community topic in the New Relic Explorers Hub. You can find this project's topic/threads here:

## Contribute

We encourage your contributions to improve nr-openai-observability! Keep in mind that when you submit your pull request, you'll need to sign the CLA via the click-through using CLA-Assistant. You only have to sign the CLA one time per project.

If you have any questions, or to execute our corporate CLA (which is required if your contribution is on behalf of a company), drop us an email at opensource@newrelic.com.

**A note about vulnerabilities**

As noted in our [security policy](../../security/policy), New Relic is committed to the privacy and security of our customers and their data. We believe that providing coordinated disclosure by security researchers and engaging with the security community are important means to achieve our security goals.

If you believe you have found a security vulnerability in this project or any of New Relic's products or websites, we welcome and greatly appreciate you reporting it to New Relic through [HackerOne](https://hackerone.com/newrelic).

If you would like to contribute to this project, review [these guidelines](./CONTRIBUTING.md).

To all contributors, we thank you!  Without your contribution, this project would not be what it is today.

## License
nr-openai-observability is licensed under the [Apache 2.0](http://apache.org/licenses/LICENSE-2.0.txt) License.
The nr-openai-observability also uses source code from third-party libraries. You can find full details on which libraries are used and the terms under which they are licensed in the third-party notices document.
