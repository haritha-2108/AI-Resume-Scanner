from django.db import models

class Resume(models.Model):
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='resumes/')
    job_description = models.TextField()
    score = models.FloatField()
    strengths = models.TextField()
    weaknesses = models.TextField()
    suggestions = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name