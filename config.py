"""Main configuration file"""
from dotenv import load_dotenv
from os import getenv, path

# Getting the current env
current_env = getenv('ISHIELD_ENV')

# If env is not set
if not path.exists("{}.env".format(current_env)):
    raise EnvironmentError("ISHIELD_ENV not set properly.")

# Loading the env file
load_dotenv('{}.env'.format(current_env))

# Getting values from env file
TOKEN_SECRET = getenv('TOKEN_SECRET')
LOG_LEVEL = getenv('LOG_LEVEL')
PORT = getenv('PORT')
HOST = getenv('HOST')

OPENAI_API_KEY = getenv('OPENAI_API_KEY')
GET_ACTIVITY_TYPE_CONFIG = getenv('GET_ACTIVITY_TYPE_CONFIG')
GET_CONTENT_TEMPLATE_CONFIG = getenv('GET_CONTENT_TEMPLATE_CONFIG')
GET_CAMPAIGN_CONSTRUCT_CONFIG = getenv('GET_CAMPAIGN_CONSTRUCT_CONFIG')
JD_EXTRACTION_MODEL = getenv('JD_EXTRACTION_MODEL')
RESUME_EXTRACTION_MODEL = getenv('RESUME_EXTRACTION_MODEL')
RESUME_EXTRACTION_MODEL_ALT = getenv('RESUME_EXTRACTION_MODEL_ALT')
CONTENT_EXTRACTION_MODEL = getenv('CONTENT_EXTRACTION_MODEL')
RESUME_EXTRACTION_TOKEN_THRESHOLD = 6000
CONTENT_GENERATION_TOKEN_BUFFER = 1.4
GOOGLE_API_KEY = getenv('GOOGLE_API_KEY')

# Env variables for the Qdrant
QDRANT_URL = getenv('QDRANT_URL')
QDRANT_API_KEY = getenv('QDRANT_API_KEY')
QDRANT_COLLECTION_NAME = "ishield_jd_collection_v2"
QDRANT_RESUME_COLLECTION_NAME = 'ishield_resume_collection_v1'
QDRANT_CONTENT_COLLECTION_NAME = 'ishield_content_collection_v1'
QDRANT_RESUME_MAX_SEARCH_LIMIT = 60
QDRANT_RESUME_MAX_OUTPUT_LIMIT = 30

EMBEDDING_MODEL = 'text-embedding-ada-002'

RESUME_METADATA_PROMPT = """Given a candidate resume in the triple quotes below.
{{}}
Summarize each section as a list of topics, with each topic being a short phrase consisting of content words such that the context is retained and the topic does not contain more than 10 words.
Return the output in JSON format and do not return the entire sentences, rather generate summarized phrases in each section. 
In the 'projectExperiences' section, convert all dates to YYYY-MM format and calculate the elapsed  time between date ranges in number of months for each each child element, and add this integer value as a key "elapsedTime" to each child element.
Return everything in the following structure only if the valid resume is provided otherwise return NA.

{
      "profile": {
          "name": "",
          "LinkedIn_URL": ""
      },
      "contact": {
          "email": [],
          "phone": []
      },
      "education": [
          {
              "qualification": "",
              "subject": ""
          }
	],
    "mostRelevantTechnicalAndOperationalSkills": [],
    "mostRelevantManagementAndBusinessSkills": [],
    "mostRelevantTrainingAndCertifications": [],
    "mostRelevantRolesPlayed": [],
    "mostRelevantAchievements": [],
    "lessRelevantTechnicalAndOperationalSkills": [],
    "lessRelevantManagementAndBusinessSkills": [],
    "lessRelevantTrainingAndCertifications": [],
    "lessRelevantRolesPlayed": [],
    "lessRelevantAchievements": []
    "projectExperiences": [
      {
        "companyName": "",
        "jobRole": "",
        "duration": "",
        "elapsedTime": "",
        "mostRelevantTechnicalAndOperationalSkills": [],
        "mostRelevantManagementAndBusinessSkills": [],
        "lessRelevantTechnicalAndOperationalSkills": [],
        "lessRelevantManagementAndBusinessSkills": []
      }
    ]
  }
Output:"""

MOST_RELEVANT_FEATURES_IN_RESUME = ['mostRelevantTechnicalAndOperationalSkills',
                                    'mostRelevantManagementAndBusinessSkills',
                                    'mostRelevantTrainingAndCertifications', 'mostRelevantRolesPlayed',
                                    'mostRelevantAchievements']
LESS_RELEVANT_FEATURES_IN_RESUME = ['lessRelevantTechnicalAndOperationalSkills',
                                    'lessRelevantManagementAndBusinessSkills',
                                    'lessRelevantTrainingAndCertifications', 'lessRelevantRolesPlayed',
                                    'lessRelevantAchievements']
RELEVANT_EXPERIENCE_THRESHOLD = 0.60
MOST_RECENT_EXPERIENCE_THRESHOLD = 36
DEFAULT_RELEVANT_EXPERIENCE_RANGE = [0, 600]

JD_METADATA_PROMPT = """Extract the important short phrases of not more than 15 words into relevant sections from the Job Description provided in triple quotes below.
{{}}
Return the output in JSON format and do not return the entire sentences, rather generate summarized important phrases of significant length in each section and not too short as well. Consider the value of each key in JSON as a list of phrases. Also exclude any Salary Ranges and compensation figures in the extracted sections. Also, identify and mask any organization name with {{ORG}} in the generated output.
Return everything in the following structure only if the valid job description is provided otherwise return NA.
{
    "aboutRole": [],
    "responsibilities": [],
    "mandatoryRequirements": [],
    "preferredRequirements": [],
    "aboutBrand": [],
    "perksAndBenefits": [],
    "deiStatement": [],
    "languageAndTone": [],
    "mandatoryTechnicalAndOperationalSkills": [],
    "mandatoryManagementAndBusinessSkills": [],
    "mandatoryTrainingAndCertifications": [],
    "mandatoryWorkExperience": [],
    "mandatoryAccomplishments": [],
    "preferredTechnicalAndOperationalSkills": [],
    "preferredManagementAndBusinessSkills": [],
    "preferredTrainingAndCertifications": [],
    "preferredWorkExperience": [],
    "preferredAccomplishments": []
}
Output:"""

MANDATORY_FEATURES_IN_JD = ["mandatoryTechnicalAndOperationalSkills", "mandatoryManagementAndBusinessSkills",
                            "mandatoryTrainingAndCertifications", "mandatoryWorkExperience", "mandatoryAccomplishments"]
PREFERRED_FEATURES_IN_JD = ["preferredTechnicalAndOperationalSkills", "preferredManagementAndBusinessSkills",
                            "preferredTrainingAndCertifications", "preferredWorkExperience", "preferredAccomplishments"]

JD_FEATURES_TO_PREPARE_KEYWORDS = ['aboutRole', 'responsibilities', 'mandatoryRequirements', 'preferredRequirements',
                                   'aboutBrand', 'perksAndBenefits', 'deiStatement', 'languageAndTone']

RESUME_RANKING_WEIGHTS = [0.5, 0.2, 0.2, 0.1]

SET_PERSONA_CONTEXT = """Extract the most prominent persona reflective of the brand that is posting these, from the LinkedIn posts provided within the triple quotes. The persona should reflect the brand identity, the topics that the brand is communicating about, and the tone that is used, in the below output format

persona: an identity representing the employer posting these on LinkedIn, and not the company name, person name, role or target audience.
themes: List of topics discussed.
tone: the kind of tone used in the post.
LinkedIn Posts:
{{}}

Return the persona in the following JSON structure and do not return the sample output. Also, remove any hastags from the output:
Sample output:
{
    "persona": "Social and Healthy Eating Enthusiast",
    "themes":  ["Healthy lifestyle tips", "articles on how to stay active and social", "organic and local food options"],
    "tone":  ["Supportive", "aspirational", "motivating"]
}

Output:"""

CONTENT_FEATURES_FOR_KEYWORDS = ['title', 'description', 'content']

CONTENT_CLASSIFICATION_METADATA_PROMPT = """
Classify the LinkedIn post in triple quotes into any one of the following company, competitive or industry event types ("eventType"):
1. NewMarketOrLocationLaunch
2. ProductOrServiceLaunch
3. OrganizationStructureChange
4. PartnershipAnnouncement
5. EmployeeEngagement
6. RecruitingAndStaffing
7. DiversityInitiative
8. IndustryConferenceOrEvent
9. WorkplaceSafetyAndHealth
10. CompensationAndBenefits
11. HumanResourcesCompliance

LinkedIn post:
{{}}
If the post does not fall into any of the above event types, classify it as 'Other'
Summarize the content as a list of topics, with each topic being a short phrase consisting of content words such that the context is retained and each topic does not contain more than 15 words.
Determine if the content is relevant for job candidates and employees.
Provide the output in the following JSON format.
{
  "eventType": "",
  "topics": [],
  "relevantforCandidatesOrEmployees": High/Medium/Low
}
Output:"""


CAMPAIGN_SUGGESTION_METADATA_PROMPT = """
Suggest a relevant and eye-catching title (in up to 10 words) and brief description (in up to 50 words) for a campaign that the brand can run addressing their talent pool and employees, based on the below inputs:
  - Include the topics: {{topics}}
  - Personify the content author's identity as: {{persona_name}}
  - Incorporate the themes: {{themes}}
  - Embody the tone: {{tone}}
Provide the output in the following JSON format.
{
  "campaignTitle": "",
  "campaignDescription": ""
}
Output:"""

CRITERIA_MATCH_THRESHOLDS = {
    'High': 0.9,
    'Medium': 0.8,
    'Low': 0.7
}

BRAND_CRITERIA_MATCH_THRESHOLD = 0.15
BRAND_NAME_MISSING_PENALTY = 0.1
CONTENT_RESULTS_LIMIT = 25

TOP_PERSONA_EXTRACTION_PROMPT = """Extract {{count}} most prominent but distinct persona reflective of the brand that is posting content addressing their talent pool, from the social media posts within the triple quotes below. 
Each persona should reflect the following:
- personaName:  The persona adopted by the company that is relevant to prospective talent, eg. Employee nurturing company; Sustainability Evangelist; Tech pioneer and innovator; Social Impact Trailblazer; Well-being Advocate; Socially Conscious Innovator; etc. It should not include the company name, person name, role or target audience.
- themes: list of up to 5 most prominent topics included in the content focusing on the value proposition for candidates, employees and the communities that the company interacts with.
- tone: the kind of tone used in the post.

Posts:
{{posts}}
Return the persona as a list in the following JSON structure, and do not return the sample output. Also, remove any hastags from the output:
Sample output:
[{
    "personaName": "Employer with culture of Technology Innovation",
    "themes": [
      "Work on best-in-class products",
      "Make an impact",
      "join a culture of innovation",
      "technology career development"
    ],
    "tone": [
      "creative",
      "inspirational",
      "authoritative"
    ]
}]

Output:"""

PERSONA_EXTRACTION_BATCH_SIZE = 20
CLUSTERING_INPUT_MIN_COUNT = 2

PERSONA_INCLUSION_IN_GENERATE_PROMPT = """
Incorporate at least one of the following themes, however do not include these explicitly:
{themes}
Incorporate at least one of the following tones in the language:
{tone}
"""

STYLE_INCLUSION_IN_GENERATE_PROMPT = """
Incorporate the following writing style and flow:
{style_description}
"""


TOP_KEYWORDS_PROMPT = """give the list of trending keywords/ phrases/ terms related to {{data}} in the following JSON structure. Optionally, if any metadata (ex. monthly search volume) is available for each keyword, include it within an object with the key (metadata). 
                            
Only return the output in the following JSON structure, and do not return the sample output.:

Sample output:
{
  "data": {
    "keywords": [
      {
        "keyword": "Blockchain",
        "metadata": {
            "monthly_search_volume": 450000
        }
      },
      {
        "keyword": "Stablecoin",
        "metadata": {
            "monthly_search_volume": 24000
        }
      }
    ]
  }
}


Output:"""