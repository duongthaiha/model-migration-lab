import requests
from typing_extensions import Self
from typing import TypedDict
from promptflow.tracing import trace


class ModelEndpoints:
    def __init__(self: Self, env: dict, model_type: str) -> str:
        self.env = env
        self.model_type = model_type

    class Response(TypedDict):
        query: str
        response: str

    @trace
    def __call__(self: Self, query: str) -> Response:
        if self.model_type == "gpt4":
            output = self.call_gpt4_endpoint(query)
        elif self.model_type == "gpt35-turbo":
            output = self.call_gpt35_turbo_endpoint(query)
        else:
            output = self.call_default_endpoint(query)

        return output

    def query(self: Self, endpoint: str, headers: str, payload: str) -> str:
        response = requests.post(url=endpoint, headers=headers, json=payload)
        return response.json()

    def call_gpt4_endpoint(self: Self, query: str) -> Response:
        endpoint = self.env["gpt4"]["endpoint"]
        key = self.env["gpt4"]["key"]

        headers = {"Content-Type": "application/json", "api-key": key}

        payload = {"messages": [{"role": "user", "content": query}], "max_tokens": 500}

        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        print(f'output{output}')
        response = output["choices"][0]["message"]["content"]
        return {"query": query, "response": response}

    def call_gpt35_turbo_endpoint(self: Self, query: str) -> Response:
        endpoint = self.env["gpt35-turbo"]["endpoint"]
        key = self.env["gpt35-turbo"]["key"]

        headers = {"Content-Type": "application/json", "api-key": key}

        payload = {"messages": [{"role": "user", "content": query}], "max_tokens": 500}

        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        response = output["choices"][0]["message"]["content"]
        return {"query": query, "response": response}


    def call_default_endpoint(self: Self, query: str) -> Response:
        return {"query": "What is the capital of France?", "response": "Paris"}
