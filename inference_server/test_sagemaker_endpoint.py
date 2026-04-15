"""
Test SageMaker endpoint with various payloads.
"""
import json
import argparse
import boto3


def test_endpoint(endpoint_name, region):
    """Test SageMaker endpoint with sample data"""
    
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Test cases
    test_cases = [
        {
            "name": "Single text via 'text' field",
            "payload": {"text": "I feel wonderful today!"},
        },
        {
            "name": "Multiple texts via 'texts' field",
            "payload": {
                "texts": [
                    "I'm so happy!",
                    "Feeling a bit down",
                    "Everything is terrible",
                    "I can't go on like this"
                ]
            },
        },
        {
            "name": "SageMaker instances format",
            "payload": {
                "instances": [
                    {"text": "Life is beautiful!"},
                    {"text": "I'm struggling with everything"}
                ]
            },
        },
    ]
    
    print("="*70)
    print(f"Testing SageMaker Endpoint: {endpoint_name}")
    print("="*70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-"*70)
        
        payload_json = json.dumps(test_case['payload'])
        print(f"Request: {payload_json}")
        
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Accept='application/json',
                Body=payload_json
            )
            
            result = json.loads(response['Body'].read().decode())
            print(f"\nResponse:")
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()
    
    # Test with plain text
    print(f"\nTest: Plain text input")
    print("-"*70)
    plain_text = "I am feeling great!"
    print(f"Request: {plain_text}")
    
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/plain',
            Accept='application/json',
            Body=plain_text
        )
        
        result = json.loads(response['Body'].read().decode())
        print(f"\nResponse:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SageMaker endpoint')
    parser.add_argument('--endpoint-name', required=True, help='SageMaker endpoint name')
    parser.add_argument('--region', default='ap-southeast-1', help='AWS region')
    
    args = parser.parse_args()
    
    test_endpoint(args.endpoint_name, args.region)
