import urllib.request
import json
import os
import ssl
from typing_extensions import Self
from typing import TypedDict


class ApplicationEndpoint:
    def __init__(self: Self, env: dict) -> None:
        self.env = env
        print(self.env)

    class Response(TypedDict):
        query: str
        response: str

    # @trace
    def __call__(self: Self, query: str, context: str) -> Response:
       

        def allowSelfSignedHttps(allowed):
            # bypass the server certificate verification on client side
            if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
                ssl._create_default_https_context = ssl._create_unverified_context

        allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

        # Replace this with the call to the application endpoint
        data = {"chat_input" : f"{query}","chat_history": []}

        body = str.encode(json.dumps(data))

        url = 'xxxxxx'  # Replace this with the API URL for the endpoint
        # Replace this with the api key
        api_key = 'xxxxxx'
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req)

            result = response.read()
            result_json = json.loads(result.decode('utf-8'))

            print(result_json)
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))
        return {"query": query, "response": result_json['chat_output']}
