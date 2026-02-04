import os
import requests
import random
import base64
import time # Added import
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS # Import CORS
import logging
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from dotenv import load_dotenv
import google.generativeai as genai # Added import
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='../build', static_url_path='/')
CORS(app) # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load OTX API Key
OTX_API_KEY = os.getenv('OTX_API_KEY')
OTX_BASE_URL = 'https://otx.alienvault.com/api/v1'

# IBM X-Force Exchange API credentials
IBM_API_KEY = os.getenv('IBM_API_KEY')
IBM_API_PASSWORD = os.getenv('IBM_API_PASSWORD')
IBM_API_URL = 'https://api.xforce.ibmcloud.com/vulnerabilities'

# Configure Gemini API Key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # Added Gemini API Key config
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

def get_ibm_auth_header():
    token = base64.b64encode(f"{IBM_API_KEY}:{IBM_API_PASSWORD}".encode('utf-8')).decode('ascii')
    return {'Authorization': f'Basic {token}'}

def fetch_ibm_data(params=None):
    headers = get_ibm_auth_header()
    response = requests.get(IBM_API_URL, headers=headers, params=params)
    print(f"IBM X-Force response: {response.json()}")  # Log full response for debugging
    return response.json()

# Get the absolute path to the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained models and tokenizers using absolute paths
model_path_spam = os.path.join(script_dir, 'saved_model')
model_path_detailed = os.path.join(script_dir, 'saved_model_detailed')

tokenizer_spam = DistilBertTokenizerFast.from_pretrained(model_path_spam)
model_spam = DistilBertForSequenceClassification.from_pretrained(model_path_spam, num_labels=2)

tokenizer_detailed = DistilBertTokenizerFast.from_pretrained(model_path_detailed)
model_detailed = DistilBertForSequenceClassification.from_pretrained(model_path_detailed, num_labels=7)

# Define labels for detailed classification
detailed_labels = ["Social Engineering", "DDoS", "Zero-day Exploit", "SQL Injection", "Phishing", "Malware", "Ransomware"]

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    data = request.get_json()
    text = data.get('text', '')

    inputs = tokenizer_spam(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model_spam(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    label = 'likely Spam' if prediction == 1 else 'likely Not Spam'
    return jsonify({'prediction': label})

@app.route('/predict_detailed', methods=['POST'])
def predict_detailed():
    data = request.get_json()
    text = data.get('text', '')

    inputs = tokenizer_detailed(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model_detailed(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    predicted_label = detailed_labels[prediction]
    scores = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    score_dict = dict(zip(detailed_labels, scores))

    return jsonify({'prediction': predicted_label, 'scores': score_dict})

# New AlienVault OTX-based API Endpoints

def fetch_otx_data(endpoint, params=None):
    """Utility function to make API requests to OTX"""
    headers = {'X-OTX-API-KEY': OTX_API_KEY}
    url = f'{OTX_BASE_URL}/{endpoint}'
    response = requests.get(url, headers=headers, params=params)
    return response.json()

@app.route('/api/threat-trends', methods=['GET'])
def threat_trends():
    try:
        response = fetch_otx_data('pulses/subscribed')
        trends = [{'date': pulse['modified'], 'threats': len(pulse['indicators'])} for pulse in response.get('results', [])]
        return jsonify(trends)
    except Exception as e:
        print(f"Error fetching threat trends: {e}")
        return jsonify({"error": "Failed to fetch threat trends"}), 500

@app.route('/api/top-threats', methods=['GET'])
@app.route('/api/top-threats', methods=['GET'])
def top_threats():
    try:
        # Fetch subscribed pulses from OTX
        response = fetch_otx_data('pulses/subscribed')
        pulses = response.get('results', [])

        indicator_details = []
        for pulse in pulses:
            for indicator_data in pulse.get('indicators', []):
                indicator_value = indicator_data.get('indicator')
                indicator_type = indicator_data.get('type')
                if indicator_value and indicator_type:
                    indicator_details.append({
                        "value": indicator_value,
                        "type": indicator_type
                    })
        
        # Count occurrences of each specific indicator (value + type combination)
        specific_indicator_counts = {}
        for ind in indicator_details:
            key = (ind["value"], ind["type"]) # Use a tuple of (value, type) as key
            specific_indicator_counts[key] = specific_indicator_counts.get(key, 0) + 1

        # Prepare the list for JSON response, sorted by count
        # Each item will be like: {"value": "1.2.3.4", "type": "IPv4", "count": 5}
        top_threats_list = []
        for (value, type_name), count in specific_indicator_counts.items():
            top_threats_list.append({
                "value": value,
                "type": type_name,
                "count": count
            })
        
        # Sort by count in descending order and take top 10 (or more, if desired)
        top_threats_sorted = sorted(top_threats_list, key=lambda x: x['count'], reverse=True)
        
        # Log the prepared top threats data
        # print(f"Top threats data being sent: {top_threats_sorted[:10]}") 

        return jsonify(top_threats_sorted[:20]) # Return top 20 specific indicators

    except Exception as e:
        print(f"Error fetching top threats: {e}")
        # Log the full traceback for detailed debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to fetch top threats", "details": str(e)}), 500

def generate_realistic_sector_data(threat_level=None): # Add threat_level parameter
    sectors = {
        "Finance": {"base": 25, "variance": 5},
        "Healthcare": {"base": 20, "variance": 4},
        "Technology": {"base": 18, "variance": 3},
        "Government": {"base": 15, "variance": 3},
        "Education": {"base": 12, "variance": 2},
        "Retail": {"base": 10, "variance": 2}
    }

    multiplier = 1.0
    if threat_level:
        if threat_level.lower() == "low":
            multiplier = 0.8
        elif threat_level.lower() == "medium":
            multiplier = 1.0
        elif threat_level.lower() == "high":
            multiplier = 1.2
        elif threat_level.lower() == "critical":
            multiplier = 1.5

    total = 0
    impact_data = []
    
    for sector, values in sectors.items():
        # Adjust base impact by the multiplier
        adjusted_base = values["base"] * multiplier
        # Ensure variance doesn't make base negative or overly exaggerated
        current_variance = values["variance"] * multiplier
        
        impact = adjusted_base + random.uniform(-current_variance / 2, current_variance / 2) # Smaller variance range
        impact = max(0, min(impact, 100)) 
        total += impact
        impact_data.append({"sector": sector, "impact": impact})
    
    if total == 0: # Avoid division by zero if all impacts are 0
        if impact_data:
             equal_impact = 100 / len(impact_data)
             for item in impact_data:
                item["impact"] = round(equal_impact,2)
        else: # No sectors defined, return empty
            return []
    else: # Normalize to ensure total is 100%
        for item in impact_data:
            item["impact"] = round((item["impact"] / total) * 100, 2)
    
    impact_data.sort(key=lambda x: x["impact"], reverse=True)
    return impact_data

@app.route('/api/sector-impact', methods=['GET'])
def sector_impact():
    threat_level = request.args.get('threatLevel') # Get threatLevel from query params
    try:
        # Simulate IBM X-Force API failure for now
        print("IBM X-Force API call is being skipped. Using generated data influenced by threat level.")
        industry_data = {} # Simulate empty/failed response

        if industry_data: # This block will be skipped in this simulation
            total = sum(industry_data.values())
            impact_data = [
                {
                    'sector': industry,
                    'impact': round((count / total) * 100, 2)
                }
                for industry, count in sorted(industry_data.items(), key=lambda x: x[1], reverse=True)[:6]
            ]
            is_default = False
        else:
            print("No industry data found or API failed, using generated default data")
            impact_data = generate_realistic_sector_data(threat_level) # Pass threat_level
            is_default = True

        print("Sector Impact Data:", impact_data)
        return jsonify({'data': impact_data, 'is_default': is_default})
    except Exception as e:
        print(f"Error fetching sector impact: {e}")
        print("API error, using generated default data influenced by threat level")
        impact_data = generate_realistic_sector_data(threat_level) # Pass threat_level in case of exception too
        return jsonify({'data': impact_data, 'is_default': True}), 500

# New endpoint for OTX indicator type distribution
@app.route('/api/otx-indicator-types', methods=['GET'])
def otx_indicator_types():
    try:
        response = fetch_otx_data('pulses/subscribed') # Assuming fetch_otx_data is defined elsewhere
        indicators = []
        for pulse in response.get('results', []):
            indicators.extend(pulse.get('indicators', []))
        
        indicator_counts = {}
        for indicator in indicators:
            indicator_type = indicator.get('type')
            if indicator_type: # Ensure type is not None
                indicator_counts[indicator_type] = indicator_counts.get(indicator_type, 0) + 1
        
        # Convert to list of objects for recharts
        indicator_data = [{'name': name, 'count': count} for name, count in indicator_counts.items()]
        return jsonify(indicator_data)
    except Exception as e:
        print(f"Error fetching OTX indicator types: {e}")
        return jsonify({"error": "Failed to fetch OTX indicator types"}), 500

# New Gemini-based Threat Analysis Endpoint
@app.route('/analyze_threat', methods=['POST'])
def analyze_threat():
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key not configured"}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided for analysis"}), 400

    try:
        # Use a currently available model name
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"Analyze the following text for potential cyber threats. Provide a summary of the threat, potential impact, and recommended mitigation steps:\n\n{text}"
        print(f"Sending prompt to Gemini: {prompt}") # Log the prompt
        response = model.generate_content(prompt)
        print(f"Received raw response from Gemini: {response}") # Log the raw response

        # Check if the response has the expected structure
        if response and hasattr(response, 'text'):
             analysis_result = response.text
        elif response and hasattr(response, 'parts') and response.parts:
             # Handle potential multi-part responses if necessary
             analysis_result = "".join(part.text for part in response.parts)
        else:
             # Log the unexpected response structure for debugging
             print(f"Unexpected Gemini response structure: {response}")
             analysis_result = "Analysis could not be generated due to an unexpected response format."

        # Remove asterisks from the result
        analysis_result = analysis_result.replace('*', '')

        print(f"Processed analysis result (asterisks removed): {analysis_result}") # Log the processed result
        return jsonify({'analysis': analysis_result})

    except Exception as e:
        print(f"Error during Gemini analysis: {e}")
        # Provide more specific error feedback if possible
        error_message = f"Failed to analyze threat using Gemini: {str(e)}"
        # Check for specific API errors if the SDK provides them
        # Example: if isinstance(e, genai.APIError): error_message = ...
        return jsonify({"error": error_message}), 500

@app.route('/api/cves', methods=['GET'])
def get_cves():
    start_date_str = request.args.get('startDate')
    end_date_str = request.args.get('endDate')

    if not start_date_str or not end_date_str:
        return jsonify({"error": "startDate and endDate are required"}), 400

    try:
        start_date_nvd = f"{start_date_str}T00:00:00.000Z"
        end_date_nvd = f"{end_date_str}T23:59:59.999Z"

        nvd_api_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {
            "pubStartDate": start_date_nvd,
            "pubEndDate": end_date_nvd,
            "resultsPerPage": 50
        }
        
        time.sleep(0.6) 

        headers = {
            'User-Agent': 'CyberThreatDashboard/1.0 (contact: your-email@example.com)'
        }
        
        print(f"Requesting NVD API v2.0 with params: {params} from URL: {nvd_api_url}")
        response = requests.get(nvd_api_url, params=params, headers=headers, timeout=20)
        
        print(f"NVD API Response Status: {response.status_code}")
        response.raise_for_status()
        
        cve_data = response.json()
        cves_list = []

        if 'vulnerabilities' in cve_data:
            for cve_item_wrapper in cve_data['vulnerabilities']:
                cve = cve_item_wrapper.get('cve', {})
                cve_id = cve.get('id', 'N/A')
                description = "N/A"
                if cve.get('descriptions'):
                    eng_desc = next((d['value'] for d in cve['descriptions'] if d.get('lang') == 'en'), None)
                    description = eng_desc if eng_desc else (cve['descriptions'][0]['value'] if cve['descriptions'] else "N/A")

                published_date = cve.get('published', 'N/A')
                severity = "N/A"

                # CVSS v3.1 (preferred)
                if cve.get('metrics', {}).get('cvssMetricV31') and len(cve['metrics']['cvssMetricV31']) > 0:
                    severity = cve['metrics']['cvssMetricV31'][0]['cvssData'].get('baseSeverity', 'N/A')
                # CVSS v3.0
                elif cve.get('metrics', {}).get('cvssMetricV30') and len(cve['metrics']['cvssMetricV30']) > 0:
                    severity = cve['metrics']['cvssMetricV30'][0]['cvssData'].get('baseSeverity', 'N/A')
                # CVSS v2
                elif cve.get('metrics', {}).get('cvssMetricV2') and len(cve['metrics']['cvssMetricV2']) > 0:
                    cvss_v2_metrics = cve['metrics']['cvssMetricV2'][0]
                    severity = cvss_v2_metrics.get('baseSeverity', 'N/A')
                
                cves_list.append({
                    "id": cve_id,
                    "description": description,
                    "publishedDate": published_date,
                    "severity": severity
                })
        else:
            print(f"CVE data received from NVD API v2.0, but in an unexpected format or empty. Response: {cve_data}")

        return jsonify(cves_list)

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching CVEs from NVD: {http_err}")
        print(f"Response content: {response.text if 'response' in locals() else 'Response object not available'}")
        return jsonify({"error": f"NVD API request failed: {http_err}", "details": response.text if 'response' in locals() else 'No response text'}), response.status_code if 'response' in locals() else 500
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CVEs from NVD: {e}")
        return jsonify({"error": f"Failed to fetch CVEs from NVD: {str(e)}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred in get_cves: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred while fetching CVEs"}), 500

# Placeholder for actual LLM call
def get_llm_recommendation_from_model(threat_level, threats_summary):
    """
    Generates a security recommendation based on threat level and details.
    Uses Gemini API for generating recommendations.
    """
    app.logger.info(f"Generating LLM recommendation for Level: {threat_level}, Details: {threats_summary}")

    if not GEMINI_API_KEY:
        app.logger.error("GEMINI_API_KEY not configured. Cannot fetch LLM recommendation.")
        return "AI Advisor is currently unavailable due to a configuration issue."

    prompt_parts = [
        f"Generate a concise, actionable security recommendation for a non-technical user. The file scan threat level is '{threat_level}'. "
    ]

    if threats_summary:
        prompt_parts.append("The following threats were identified (up to 3 shown):")
        for i, threat in enumerate(threats_summary):
            if i < 3:
                prompt_parts.append(f"- File '{threat.get('name', 'Unknown')}' status: {threat.get('status', 'N/A')}.")
            else:
                prompt_parts.append(f"- And {len(threats_summary) - 3} more...")
                break
    
    prompt_parts.append(
        "The recommendation should be easy to understand and guide the user on immediate next steps. Format it with clear, numbered or bulleted points if multiple steps are involved. Ensure the output is plain text and directly usable. Avoid markdown like asterisks for bolding, use newlines (\\\\n) for separation if needed."
    )
    final_prompt = "\\n".join(prompt_parts)
    app.logger.info(f"Constructed LLM Prompt for recommendation: {final_prompt}")

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or your preferred Gemini model
        response = model.generate_content(final_prompt)
        
        recommendation = ""
        if response and hasattr(response, 'text'):
             recommendation = response.text
        elif response and hasattr(response, 'parts') and response.parts:
             recommendation = "".join(part.text for part in response.parts)
        else:
             app.logger.error(f"Unexpected Gemini response structure for recommendation: {response}")
             recommendation = "Could not generate AI recommendation due to an unexpected response format."

        # Basic cleanup, e.g., removing leading/trailing asterisks or unwanted markdown
        recommendation = recommendation.strip().replace('*', '')
        app.logger.info(f"Raw recommendation from Gemini: {recommendation}")
        return recommendation

    except Exception as e:
        app.logger.error(f"Error during Gemini call for recommendation: {e}", exc_info=True)
        # Fallback to a simpler, predefined message if Gemini fails
        if threat_level == 'High':
            return ("High threat detected! Critical action advised: Isolate the system, run full antivirus scan, and consult IT security if available. Avoid using suspicious files.")
        elif threat_level == 'Medium':
            return ("Medium threat detected. Caution advised: Review suspicious files, perform targeted antivirus scan, and monitor system behavior. Be wary of downloads and attachments.")
        elif threat_level == 'Low':
            return ("Low threat detected. Maintain good security hygiene: Keep software updated, use strong passwords, and be cautious online.")
        elif threat_level == 'Error':
            return "Scan Error: Could not determine the threat level. Please check scan details and try again."
        else:
            return "No specific threats identified. Continue practicing safe computing habits."

@app.route('/api/llm/recommendation', methods=['POST'])
def llm_recommendation_endpoint():
    try:
        data = request.get_json()
        if not data:
            app.logger.error("LLM Recommendation: No data provided in request")
            return jsonify({"error": "No data provided"}), 400

        threat_level = data.get('threatLevel')
        # Threats details are optional, provide an empty list if not present
        threats_details = data.get('threats', []) 

        if not threat_level:
            app.logger.error("LLM Recommendation: 'threatLevel' is required but not provided")
            return jsonify({"error": "threatLevel is required"}), 400

        app.logger.info(f"Received LLM recommendation request: Level='{threat_level}', Threats Count='{len(threats_details)}'")
        
        recommendation_text = get_llm_recommendation_from_model(threat_level, threats_details)
        
        app.logger.info(f"Generated LLM Recommendation: {recommendation_text}")
        return jsonify({"recommendation": recommendation_text})

    except Exception as e:
        app.logger.error(f"Error in LLM recommendation endpoint: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate recommendation", "details": str(e)}), 500

@app.route('/api/analyze_email_phishing', methods=['POST'])
def analyze_email_phishing_endpoint():
    if not GEMINI_API_KEY:
        app.logger.error("Phishing Analyzer: GEMINI_API_KEY not configured.")
        return jsonify({"error": "Phishing Analyzer is currently unavailable due to a configuration issue."}), 500

    data = request.get_json()
    email_text = data.get('email_text', '')

    if not email_text:
        app.logger.warn("Phishing Analyzer: No email_text provided.")
        return jsonify({"error": "No email text provided for analysis"}), 400

    app.logger.info(f"Phishing Analyzer: Received email text for analysis (length: {len(email_text)}).")

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = (
            "Analyze the following email text for signs of phishing. "
            "Provide a phishing likelihood (Low, Medium, High). "
            "List specific suspicious elements found (e.g., urgent requests, generic greetings, mismatched links if described, grammatical errors, suspicious attachments mentioned). "
            "Offer a brief explanation for your assessment. "
            "Format the output as plain text, using newlines (\\\\n) for separation. Example:\\n"
            "Likelihood: High\\\\n"
            "Suspicious Elements:\\\\n"
            "- Urges immediate action to avoid account suspension.\\\\n"
            "- Generic greeting 'Dear User'.\\\\n"
            "- Contains a link to 'http://example-login-totally-not-fake.com'.\\\\n"
            "Explanation: The email creates a false sense of urgency and uses a suspicious link, common tactics in phishing attacks.\\\\n"
            "---BEGIN EMAIL TEXT ANALYZED---\\\\n"
            f"{email_text}"
            "---END EMAIL TEXT ANALYZED---"
        )
        
        app.logger.info(f"Constructed Phishing Analysis Prompt: {prompt[:500]}...") # Log a snippet
        response = model.generate_content(prompt)
        
        analysis_result = ""
        if response and hasattr(response, 'text'):
             analysis_result = response.text
        elif response and hasattr(response, 'parts') and response.parts:
             analysis_result = "".join(part.text for part in response.parts)
        else:
             app.logger.error(f"Unexpected Gemini response structure for phishing analysis: {response}")
             analysis_result = "Could not generate phishing analysis due to an unexpected response format."

        analysis_result = analysis_result.strip().replace('*', '') # Basic cleanup
        app.logger.info(f"Raw Phishing Analysis from Gemini: {analysis_result}")
        
        # Attempt to parse the structured response (optional, but good for frontend)
        parsed_result = {"raw_text": analysis_result, "likelihood": "N/A", "elements": [], "explanation": "N/A"}
        try:
            lines = analysis_result.split('\\\\n')
            for line in lines:
                if line.startswith("Likelihood:"):
                    parsed_result["likelihood"] = line.split(":", 1)[1].strip()
                elif line.startswith("Suspicious Elements:"):
                    # This part would need more robust parsing if elements are multi-line or complex
                    pass # For now, keep it simple, frontend can display raw_text
                elif line.startswith("Explanation:"):
                    parsed_result["explanation"] = line.split(":", 1)[1].strip()
            # A more robust parsing would be needed here if strict structure is required.
            # For now, sending the raw text is safer and the frontend can display it.
        except Exception as parse_err:
            app.logger.error(f"Could not parse phishing analysis: {parse_err}")


        return jsonify({"analysis": analysis_result}) # Send the raw, cleaned text for now

    except Exception as e:
        app.logger.error(f"Error during Gemini call for phishing analysis: {e}", exc_info=True)
        return jsonify({"error": "Failed to analyze email for phishing", "details": str(e)}), 500

@app.route('/api/llm/explain_file_threat', methods=['POST'])
def explain_file_threat_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        file_name = data.get('fileName')
        threat_status = data.get('threatStatus')
        # file_hash = data.get('fileHash') # Optional, can be used for more context

        if not file_name or not threat_status:
            return jsonify({"error": "Missing fileName or threatStatus"}), 400

        app.logger.info(f"Received request to explain threat for file: {file_name}, status: {threat_status}")

        # Prompt engineering for Gemini
        prompt = f"""Explain the security implications for an average computer user for a file named '{file_name}' that has been identified with a status of '{threat_status}'.
        Keep the explanation concise (2-3 sentences), easy to understand, and actionable if possible.
        For example, if status is 'Malicious', explain what that means and what the user should generally do.
        If status is 'Suspicious', explain the uncertainty and recommended caution.
        If status is 'Clean', briefly reassure the user.
        Do not use markdown in your response.
        """

        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest') # Changed model
            response = model.generate_content(prompt)
            explanation = response.text
            app.logger.info(f"Gemini explanation for {file_name} ({threat_status}): {explanation}")
            return jsonify({"explanation": explanation})
        except Exception as e:
            app.logger.error(f"Gemini API error while explaining file threat: {e}")
            # Fallback explanation
            fallback_explanation = f"Could not get a detailed AI explanation for {file_name} ({threat_status}). "
            if threat_status == "Malicious":
                fallback_explanation += "This status means the file is considered dangerous. Do not open it and consider deleting it immediately."
            elif threat_status == "Suspicious":
                fallback_explanation += "This status means the file has some characteristics that could be harmful. It's best to be cautious and avoid opening it unless you are certain of its safety."
            else:
                fallback_explanation += "This status generally means the file is considered safe by the scan."
            return jsonify({"explanation": fallback_explanation})

    except Exception as e:
        app.logger.error(f"Error in /api/llm/explain_file_threat: {e}")
        return jsonify({"error": "Failed to get explanation due to an internal server error.", "details": str(e)}), 500

@app.route('/api/llm/explain_ip_threat', methods=['POST'])
def explain_ip_threat_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        ip_address = data.get('ipAddress')
        abuse_score = data.get('abuseScore')
        country = data.get('country')
        isp = data.get('isp')
        usage_type = data.get('usageType')
        domain = data.get('domain')

        if not ip_address or abuse_score is None: # abuse_score can be 0, so check for None
            return jsonify({"error": "Missing ipAddress or abuseScore"}), 400

        app.logger.info(f"Received request to explain IP threat for: {ip_address}")

        prompt = f"""Explain the security risk associated with the IP address '{ip_address}' for an average computer user.
        This IP has an abuse confidence score of {abuse_score}% (0-100%, higher is worse).
        Additional details: Country: {country or 'N/A'}, ISP: {isp or 'N/A'}, Usage Type: {usage_type or 'N/A'}, Associated Domain: {domain or 'N/A'}.
        Keep the explanation concise (2-4 sentences), easy to understand, and suggest general precautions if the score is high.
        For example, if the score is high, mention risks like association with spam, malware, or other malicious activities and advise caution with any unsolicited communication from or interaction with this IP.
        If the score is low, indicate it appears to be low risk based on the score.
        Do not use markdown in your response.
        """

        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest') # Changed model
            response = model.generate_content(prompt)
            explanation = response.text
            app.logger.info(f"Gemini explanation for IP {ip_address}: {explanation}")
            return jsonify({"explanation": explanation})
        except Exception as e:
            app.logger.error(f"Gemini API error while explaining IP threat: {e}")
            fallback_explanation = f"Could not get a detailed AI explanation for IP {ip_address} (Score: {abuse_score}%). "
            if abuse_score > 75:
                fallback_explanation += "This IP has a high abuse score, suggesting it may be involved in malicious activities. Exercise extreme caution."
            elif abuse_score > 25:
                fallback_explanation += "This IP has a moderate abuse score. It might be risky, so be cautious."
            else:
                fallback_explanation += "This IP has a low abuse score, suggesting it is likely safe based on current data."
            return jsonify({"explanation": fallback_explanation})

    except Exception as e:
        app.logger.error(f"Error in /api/llm/explain_ip_threat: {e}")
        return jsonify({"error": "Failed to get IP explanation due to an internal server error.", "details": str(e)}), 500

@app.route('/api/llm/explain_cve', methods=['POST'])
def explain_cve_route():
    try:
        data = request.get_json()
        if not data:
            app.logger.error("Explain CVE: No data provided")
            return jsonify({"error": "No data provided"}), 400

        cve_id = data.get('cveId')
        description = data.get('description')
        severity = data.get('severity')

        if not cve_id or not description: # Severity can be N/A
            app.logger.error(f"Explain CVE: Missing cveId or description. Received: ID='{cve_id}', Desc='{description}', Severity='{severity}'")
            return jsonify({"error": "Missing cveId or description"}), 400

        app.logger.info(f"Received request to explain CVE: {cve_id}")

        prompt = f"""Explain the security implications of CVE '{cve_id}' for an average computer user or small business.
        The CVE description is: "{description}"
        The reported severity is: '{severity or 'Not Available'}'.
        Keep the explanation concise (3-5 sentences), easy to understand, and suggest general protective measures if applicable.
        Focus on what this vulnerability means in practical terms and what kind of risks it might pose.
        Do not use markdown in your response. Output plain text.
        Example of what to explain: What kind of systems might be affected? What could an attacker do if they exploit this? What general advice can you give (e.g., update software, be cautious of phishing)?
        """

        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest') # Changed model
            response = model.generate_content(prompt)
            explanation = response.text.strip().replace('*', '')
            app.logger.info(f"Gemini explanation for CVE {cve_id}: {explanation}")
            return jsonify({"explanation": explanation})
        except Exception as e:
            app.logger.error(f"Gemini API error while explaining CVE {cve_id}: {e}")
            fallback_explanation = f"Could not get a detailed AI explanation for {cve_id} (Severity: {severity or 'N/A'}). "
            if severity and severity.upper() in ["CRITICAL", "HIGH"]:
                fallback_explanation += f"This CVE is rated as {severity}, indicating a significant vulnerability. It's crucial to check for patches from software vendors and apply them as soon as possible. Avoid exposing affected systems to untrusted networks."
            elif severity and severity.upper() == "MEDIUM":
                fallback_explanation += f"This CVE is rated as {severity}. Users should review vendor advisories for patches and apply them. Monitor systems for any unusual activity."
            else:
                fallback_explanation += "Review vendor advisories for this CVE to understand its impact and apply any recommended updates or mitigations."
            return jsonify({"explanation": fallback_explanation})

    except Exception as e:
        app.logger.error(f"Error in /api/llm/explain_cve: {e}", exc_info=True)
        return jsonify({"error": "Failed to get CVE explanation due to an internal server error.", "details": str(e)}), 500

@app.route('/api/llm/explain_threatscape_event', methods=['POST'])
def explain_threatscape_event_route():
    try:
        data = request.get_json()
        if not data:
            app.logger.error("Explain ThreatScape Event: No data provided")
            return jsonify({"error": "No data provided"}), 400

        threat_type = data.get('type')
        severity = data.get('severity')
        source_city = data.get('source', {}).get('city')
        target_city = data.get('target', {}).get('city')
        description = data.get('description')

        if not all([threat_type, severity, source_city, target_city, description]):
            app.logger.error(f"Explain ThreatScape Event: Missing required fields. Received: {data}")
            return jsonify({"error": "Missing required fields (type, severity, source.city, target.city, description)"}), 400

        app.logger.info(f"Received request to explain ThreatScape event: {description}")

        prompt = f"""\
Explain the security implications of the following cyber threat event for an average computer user or small business:
        Description: "{description}"
        Threat Type: {threat_type}
        Severity: {severity}
        Source City: {source_city}
        Target City: {target_city}

        Keep the explanation concise (3-5 sentences), easy to understand, and suggest general protective measures if applicable.
        Focus on what this event means in practical terms and what kind of risks it might pose.
        Do not use markdown in your response. Output plain text.
        Example of what to explain: What kind of attack is this? What are the typical goals? What general advice can you give (e.g., be cautious of phishing, ensure systems are updated, use strong passwords)?
        """

        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt)
            explanation = response.text.strip().replace('*', '')
            app.logger.info(f"Gemini explanation for ThreatScape event '{description}': {explanation}")
            return jsonify({"explanation": explanation})
        except Exception as e:
            app.logger.error(f"Gemini API error while explaining ThreatScape event '{description}': {e}")
            fallback_explanation = f"Could not get a detailed AI explanation for the event: {description}. "
            if severity.lower() in ["critical", "high"]:
                fallback_explanation += f"This event is rated as {severity}, indicating a significant threat. It is important to ensure all security measures are up-to-date and to be vigilant against suspicious activities related to this type of event."
            else:
                fallback_explanation += "Please ensure your systems are updated and be cautious of any unusual activity related to this type of event."
            return jsonify({"explanation": fallback_explanation})

    except Exception as e:
        app.logger.error(f"Error in /api/llm/explain_threatscape_event: {e}", exc_info=True)
        return jsonify({"error": "Failed to get ThreatScape event explanation due to an internal server error.", "details": str(e)}), 500

# Mock data for ThreatScape
def generate_mock_threat_data(count=50):
    threats = []
    threat_types = ["phishing", "malware", "ddos", "botnet_activity", "exploit_kit"]
    severities = ["low", "medium", "high", "critical"]
    
    cities = [
        {"name": "New York", "lat": 40.7128, "lng": -74.0060, "country": "USA"},
        {"name": "London", "lat": 51.5074, "lng": -0.1278, "country": "UK"},
        {"name": "Tokyo", "lat": 35.6895, "lng": 139.6917, "country": "Japan"},
        {"name": "Paris", "lat": 48.8566, "lng": 2.3522, "country": "France"},
        {"name": "Berlin", "lat": 52.5200, "lng": 13.4050, "country": "Germany"},
        {"name": "Moscow", "lat": 55.7558, "lng": 37.6173, "country": "Russia"},
        {"name": "Beijing", "lat": 39.9042, "lng": 116.4074, "country": "China"},
        {"name": "Sydney", "lat": -33.8688, "lng": 151.2093, "country": "Australia"},
        {"name": "Sao Paulo", "lat": -23.5505, "lng": -46.6333, "country": "Brazil"},
        {"name": "Cairo", "lat": 30.0444, "lng": 31.2357, "country": "Egypt"},
        {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777, "country": "India"},
        {"name": "Lagos", "lat": 6.5244, "lng": 3.3792, "country": "Nigeria"},
        {"name": "Mexico City", "lat": 19.4326, "lng": -99.1332, "country": "Mexico"},
        {"name": "Buenos Aires", "lat": -34.6037, "lng": -58.3816, "country": "Argentina"},
        {"name": "Toronto", "lat": 43.6532, "lng": -79.3832, "country": "Canada"}
    ]

    for i in range(count):
        src_city = random.choice(cities)
        possible_targets = [city for city in cities if city["name"] != src_city["name"]]
        tgt_city = random.choice(possible_targets) if possible_targets else src_city

        threat_type_choice = random.choice(threat_types)
        threats.append({
            "id": f"threat_{i}_{datetime.now().timestamp()}",
            "type": threat_type_choice,
            "severity": random.choice(severities),
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 60*24))).isoformat() + "Z",
            "source": {"lat": src_city["lat"], "lng": src_city["lng"], "city": src_city["name"], "country": src_city["country"]},
            "target": {"lat": tgt_city["lat"], "lng": tgt_city["lng"], "city": tgt_city["name"], "country": tgt_city["country"]},
            "description": f"{threat_type_choice.replace('_', ' ').title()} event from {src_city['name']} to {tgt_city['name']}"
        })
    return threats

@app.route('/api/threatscape_data', methods=['GET'])
def threatscape_data_route(): # Renamed to avoid conflict if 'threatscape_data' is used elsewhere
    try:
        data = generate_mock_threat_data(random.randint(30, 70))
        return jsonify(data)
    except Exception as e_exc: # Changed variable name to avoid conflict with 'e' if used in a broader scope
        print(f"Error in /api/threatscape_data: {e_exc}")
        return jsonify({"error": "Failed to generate threat data", "details": str(e_exc)}), 500

if __name__ == '__main__':
    # Ensure the app runs on a specific port if needed, e.g., port=5001
    # Use host='0.0.0.0' to make it accessible on the network
    app.run(debug=True, host='0.0.0.0', port=5001) # Added host and port