from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Initializing the App and Gemini API
working_dir = os.path.dirname(os.path.abspath(__file__))

# path of config_data file
config_file_path = os.path.join(working_dir, "config.json")
config_data = json.load(open(config_file_path))

# loading the GOOGLE_API_KEY
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# configuring google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
CORS(app)

config = {"temperature": 0, "top_k": 20, "top_p": 0.9, "max_output_tokens": 2048}

model = genai.GenerativeModel(model_name="gemini-1.5-flash")


class CodeReviewer:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_review(self, code):
        try:
            prompt = (
                f"Act as an expert code reviewer. Analyze the following code:\n\n{code}\n\n"
                "Report your observations on the following parameters in detail:\n"
                "1. Code Issues: Identify any logical errors, type errors, runtime errors, and syntax errors.\n"
                "2. Performance and Optimization Issues: Highlight potential bottlenecks, inefficient algorithms, and suboptimal practices affecting performance. Include issues with time and space complexity, loops, data structures, memory management, SQL queries, I/O operations, object creation, garbage collection, memory leaks, concurrency, parallelism, and thread management.\n"
                "3. Security Vulnerabilities: Look for OWASP 10 vulnerabilities such as injection, broken access control, cryptographic failures, sensitive data exposure, XXS, XXE, insecure deserialization, and any other security concerns. Check for user inputs that are not validated, hardcoded credentials, and logging of sensitive information.\n"
                "4. Scalability Issues: Identify inefficient database queries, improper indexing, unoptimized query plans, resource management issues, and caching problems.\n"
                "5. Engineering Practices: Check for adherence to coding standards, meaningful naming conventions, proper commenting and documentation, modularity, separation of concerns, use of design patterns, and proper error handling.\n"
                "For each issue, provide a description and criticality level (1-5), and suggest refactored code if applicable.\n"
                "Return the response in the following JSON format:\n"
                "{"
                '  "codeIssues": ['
                '    {"issueType": "Logical error", "description": "", "criticalityLevel": 3},'
                '    {"issueType": "Runtime error", "description": "", "criticalityLevel": 2}'
                "  ],"
                '  "performanceOptimizationIssues": ['
                '    {"issueType": "Unoptimized Time and Space Complexity of Algorithms", "description": "", "criticalityLevel": 1},'
                '    {"issueType": "Unoptimized use of Loops", "description": "POxyz", "criticalityLevel": 2}'
                "  ],"
                '  "securityVulnerabilityIssues": ['
                '    {"issueType": "OWASP 10 security vulnerabilities", "description": "", "criticalityLevel": 3},'
                '    {"issueType": "hardcoded credentials or API keys in code", "description": "", "criticalityLevel": 4}'
                "  ],"
                '  "scalabilityIssues": ['
                '    {"issueType": "Inefficient use of database queries", "description": "", "criticalityLevel": 5},'
                '    {"issueType": "improper indexing", "description": "", "criticalityLevel": 1}'
                "  ],"
                '  "engineeringPracticesIssues": ['
                '    {"issueType": "Non-adherence to coding standards", "description": "", "criticalityLevel": 3},'
                '    {"issueType": "non-meaningful and non-descriptive constructs", "description": "", "criticalityLevel": 4}'
                "  ],"
                '  "refactoredCode": "refactored/updated/correct code"'
                "}"
                "If no issues are found, return an empty array for each parameter."
            )

            response = model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )

            print(f"Raw response: {response}")

            if response and response.parts:
                generated_review = response.parts[0].text
                # Extract only the JSON part of the response
                json_match = re.search(
                    r'\{.*"refactoredCode":', generated_review, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group()
                    json_str += '""}'

                    try:
                        review_dict = json.loads(json_str)

                        # Now extract the refactored code separately
                        refactored_code_match = re.search(
                            r'"refactoredCode":(.*)$', generated_review, re.DOTALL
                        )
                        if refactored_code_match:
                            review_dict["refactoredCode"] = refactored_code_match.group(
                                1
                            ).strip()[
                                1:-1
                            ]  # Remove surrounding quotes

                        return review_dict
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        return {
                            "error": "Error decoding JSON response",
                            "raw_response": json_str,
                        }
                else:
                    return {
                        "error": "No JSON found in response",
                        "raw_response": generated_review,
                    }
            else:
                print(f"Unexpected response format: {response}")
                return {
                    "error": "Unexpected response format",
                    "raw_response": str(response),
                }

        except Exception as e:
            print(f"Error generating review: {e}")
            return {"error": f"Error generating review: {e}"}


class CodeAssistant:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_response(self, code, user_prompt):
        try:
            prompt = (
                f"Act as a code assistant. Your task is to help users with code related queries.\n"
                f"The code that you need to refer is following: {code}\n"
                f"The user Prompt that you need to answer is: {user_prompt}\n"
                f"If the user prompt is not related to the code reply back saying 'Please enter a valid query related to the code'\n"
                f"Just directly return the response in the following JSON format nothing else in JSON apart from JSON:\n"
                f'{{\n  "Response": "provide your response to the prompt"\n}}'
            )

            response = model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )

            print(f"Raw response: {response}")

            if response and response.parts:
                generated_response = response.parts[0].text
                # Remove markdown formatting and decode JSON
                generated_response = (
                    generated_response.replace("```json\n", "")
                    .replace("```", "")
                    .strip()
                )
                print(f"gen response: {generated_response}")

                try:
                    return json.loads(generated_response)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    return {
                        "error": "Error decoding JSON response",
                        "raw_response": generated_response,
                    }
            else:
                print(f"Unexpected response format: {response}")
                return {
                    "error": "Unexpected response format",
                    "raw_response": str(response),
                }

        except Exception as e:
            print(f"Error generating response: {e}")
            return {"error": f"Error generating response: {e}"}


@app.route("/review_code", methods=["POST"])
def review_code():
    try:
        request_data = request.json
        code = request_data.get("code")

        print(f"Received code: {code}")

        reviewer = CodeReviewer(api_key=GOOGLE_API_KEY)
        review = reviewer.generate_review(code)

        print(f"Review: {review}")

        return jsonify({"review": review})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        request_data = request.json
        code = request_data.get("code")
        user_prompt = request_data.get("userPrompt")

        print(f"Received code: {code}")
        print(f"Received user prompt: {user_prompt}")

        assistant = CodeAssistant(api_key=GOOGLE_API_KEY)
        response = assistant.generate_response(code, user_prompt)

        print(f"Response: {response}")

        return jsonify({"response": response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def hello_world():
    return "Hii"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
