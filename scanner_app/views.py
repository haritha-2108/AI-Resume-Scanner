from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Resume
from .utils import extract_text_from_file
from .gemini_utils import analyze_resume_with_gemini
import logging

# Set up logging
logger = logging.getLogger(__name__)

def index(request):
    if request.method == "POST":
        name = request.POST.get("name")
        job_description = request.POST.get("job_description")
        file = request.FILES.get("resume")

        if not file:
            messages.error(request, "Please upload a resume file.")
            return render(request, "index.html")
            
        if not name:
            messages.error(request, "Please provide a candidate name.")
            return render(request, "index.html")

        try:
            # Extract resume text
            resume_text = extract_text_from_file(file)
            
            if not resume_text or len(resume_text.strip()) < 50:
                messages.error(request, "Could not extract sufficient text from the resume. Please ensure it's a valid document.")
                return render(request, "index.html")

            # Gemini AI analysis
            analysis = analyze_resume_with_gemini(resume_text, job_description)
            
            # Validate and process the analysis results
            if not analysis or not isinstance(analysis, dict):
                messages.error(request, "Analysis failed. Please try again.")
                return render(request, "index.html")
                
            # Ensure score is numeric 0-100
            score = analysis.get("score", 0)
            if not isinstance(score, (int, float)):
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0
                    
            score = max(0, min(100, score))
            
            # Ensure lists for strengths, weaknesses, suggestions
            strengths = analysis.get("strengths", [])
            if not isinstance(strengths, list):
                strengths = [strengths] if strengths else []
                
            weaknesses = analysis.get("weaknesses", [])
            if not isinstance(weaknesses, list):
                weaknesses = [weaknesses] if weaknesses else []
                
            suggestions = analysis.get("suggestions", [])
            if not isinstance(suggestions, list):
                suggestions = [suggestions] if suggestions else []

            # Save to DB
            resume_instance = Resume.objects.create(
                name=name,
                file=file,
                job_description=job_description,
                score=score,
                strengths=", ".join(strengths) if strengths else "No strengths detected",
                weaknesses=", ".join(weaknesses) if weaknesses else "No weaknesses detected",
                suggestions=", ".join(suggestions) if suggestions else "No suggestions provided"
            )

            return render(request, "result.html", {
                "resume": resume_instance,
                "analysis": {
                    "score": score,
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "suggestions": suggestions
                }
            })

        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            messages.error(request, f"An error occurred while processing your resume: {str(e)}")
            return render(request, "index.html")

    return render(request, "index.html")