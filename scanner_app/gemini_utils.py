import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import logging
import re
from typing import Dict, List, Any

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up logging
logger = logging.getLogger(__name__)

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY not found in environment variables")

def analyze_resume_with_gemini(resume_text: str, job_description: str) -> Dict[str, Any]:
    """
    Analyze resume against job description using Gemini AI
    
    Args:
        resume_text (str): Extracted text from resume
        job_description (str): Job description text
        
    Returns:
        Dict: Analysis results with score, strengths, weaknesses, suggestions
    """
    
    if not GEMINI_API_KEY:
        logger.error("Gemini API key not configured")
        return get_fallback_analysis()
    
    try:
        # Create enhanced prompt
        prompt = create_analysis_prompt(resume_text, job_description)
        
        # Initialize model with safety settings
        model = genai.GenerativeModel(
            "gemini-3.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
        )
        
        # Generate analysis
        logger.info("Sending request to Gemini API")
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            logger.error("Empty response from Gemini API")
            return get_fallback_analysis()
        
        # Parse response
        analysis = parse_gemini_response(response.text)
        
        # Validate and clean analysis
        validated_analysis = validate_analysis(analysis)
        
        logger.info(f"Analysis completed successfully with score: {validated_analysis.get('score', 0)}")
        return validated_analysis
        
    except Exception as e:
        logger.error(f"Error in Gemini analysis: {str(e)}", exc_info=True)
        return get_fallback_analysis()

def create_analysis_prompt(resume_text: str, job_description: str) -> str:
    """Create detailed analysis prompt for Gemini"""
    
    prompt = f"""
You are an expert HR professional and resume analyst with 15+ years of experience in talent acquisition and candidate assessment.

TASK: Analyze the following resume against the job description and provide a comprehensive assessment.

RESUME TEXT:
{resume_text[:4000]}  # Limit text to avoid token limits

JOB DESCRIPTION:
{job_description[:2000]}  # Limit text to avoid token limits

ANALYSIS REQUIREMENTS:
1. Calculate a match score from 0-100 based on:
   - Skills alignment (40% weight)
   - Experience relevance (30% weight)  
   - Education/qualifications (15% weight)
   - Keywords match (15% weight)

2. Identify 3-5 key strengths that make this candidate suitable
3. Identify 2-4 areas where the candidate may be lacking
4. Provide 3-5 actionable suggestions for resume improvement

RESPONSE FORMAT (must be valid JSON):
{{
  "score": <number between 0-100>,
  "strengths": [
    "Specific strength with context",
    "Another strength with evidence"
  ],
  "weaknesses": [
    "Specific gap or weakness",
    "Another area for improvement"
  ],
  "suggestions": [
    "Actionable suggestion for improvement",
    "Another concrete recommendation"
  ]
}}

GUIDELINES:
- Be specific and provide context for each point
- Focus on job-relevant skills and experience
- Suggestions should be actionable and specific
- Maintain professional tone
- Base assessment on factual analysis, not assumptions
- If information is unclear, indicate uncertainty appropriately

Provide only the JSON response, no additional text.
"""
    
    return prompt

def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """Parse Gemini response and extract JSON"""
    
    try:
        # Clean response text
        cleaned_response = response_text.strip()
        
        # Remove any markdown code blocks
        cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
        cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = cleaned_response
        
        # Parse JSON
        analysis = json.loads(json_str)
        return analysis
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Response text: {response_text}")
        
        # Try to extract information manually
        return extract_info_manually(response_text)
    
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        return get_fallback_analysis()

def extract_info_manually(response_text: str) -> Dict[str, Any]:
    """Manually extract information if JSON parsing fails"""
    
    analysis = {
        "score": 50,  # Default score
        "strengths": [],
        "weaknesses": [],
        "suggestions": []
    }
    
    try:
        # Extract score
        score_match = re.search(r'score["\s:]+(\d+)', response_text, re.IGNORECASE)
        if score_match:
            analysis["score"] = int(score_match.group(1))
        
        # Extract lists (simplified approach)
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'strength' in line.lower():
                current_section = 'strengths'
            elif 'weakness' in line.lower() or 'gap' in line.lower():
                current_section = 'weaknesses'  
            elif 'suggest' in line.lower() or 'recommend' in line.lower():
                current_section = 'suggestions'
            elif current_section and line.startswith(('-', '•', '*', '1.', '2.')):
                clean_line = re.sub(r'^[-•*\d\.]\s*', '', line)
                if clean_line and len(clean_line) > 10:
                    analysis[current_section].append(clean_line)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Manual extraction failed: {str(e)}")
        return get_fallback_analysis()

def validate_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean analysis results"""
    
    validated = {
        "score": 0,
        "strengths": [],
        "weaknesses": [],
        "suggestions": []
    }
    
    # Validate score
    score = analysis.get("score", 0)
    if isinstance(score, str):
        score = re.sub(r'[^\d.]', '', score)
        try:
            score = float(score)
        except ValueError:
            score = 0
    
    validated["score"] = max(0, min(100, float(score)))
    
    # Validate lists
    for key in ["strengths", "weaknesses", "suggestions"]:
        items = analysis.get(key, [])
        
        if isinstance(items, str):
            # Split string into list
            items = [item.strip() for item in items.split(',') if item.strip()]
        
        if isinstance(items, list):
            # Clean and validate each item
            cleaned_items = []
            for item in items[:5]:  # Limit to 5 items
                if isinstance(item, str) and len(item.strip()) > 5:
                    cleaned_items.append(item.strip()[:200])  # Limit length
            
            validated[key] = cleaned_items
    
    return validated

def get_fallback_analysis() -> Dict[str, Any]:
    """Provide fallback analysis if AI fails"""
    
    return {
        "score": 50,
        "strengths": [
            "Resume format is readable and well-structured",
            "Contains relevant work experience information",
            "Educational background is clearly stated"
        ],
        "weaknesses": [
            "Unable to perform detailed analysis due to technical issues",
            "Recommend manual review by HR professional"
        ],
        "suggestions": [
            "Ensure resume includes relevant keywords from job description",
            "Quantify achievements with specific numbers and metrics",
            "Consider professional resume review for optimization"
        ]
    }

def test_gemini_connection() -> bool:
    """Test if Gemini API is working"""
    
    if not GEMINI_API_KEY:
        return False
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Test connection. Respond with 'OK'.")
        return bool(response and response.text)
        
    except Exception as e:
        logger.error(f"Gemini connection test failed: {str(e)}")
        return False