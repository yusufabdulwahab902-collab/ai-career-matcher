import os
import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Career & Job Matcher Pro",
    page_icon="🚀",
    layout="wide"
)

# --- 2. DATA LAYER (Mock + Live API Ready) ---
@st.cache_data
def load_fallback_jobs():
    """Curated fallback dataset of open job listings."""
    return pd.DataFrame([
        {
            "title": "Junior Python Developer",
            "company": "TechCorp Solutions",
            "location": "Remote",
            "requirements": "Python, Flask, SQL, Git, Bachelor's degree in Computer Science or related field",
            "url": "https://example.com/job/1"
        },
        {
            "title": "Data Analyst",
            "company": "Insight Metrics",
            "location": "New York, NY",
            "requirements": "SQL, Python, Pandas, Tableau, Statistics, Bachelor's degree in Mathematics, Statistics, or Analytics",
            "url": "https://example.com/job/2"
        },
        {
            "title": "Frontend React Developer",
            "company": "WebScale Inc.",
            "location": "San Francisco, CA",
            "requirements": "JavaScript, React, HTML, CSS, UI/UX principles, High School Diploma or equivalent with portfolio",
            "url": "https://example.com/job/3"
        },
        {
            "title": "Machine Learning Engineer",
            "company": "AI Innovations",
            "location": "Remote",
            "requirements": "Python, Scikit-Learn, PyTorch, Natural Language Processing, Master's or Bachelor's in Computer Science",
            "url": "https://example.com/job/4"
        },
        {
            "title": "Digital Marketing Specialist",
            "company": "Growth Hackers",
            "location": "Chicago, IL",
            "requirements": "SEO, Google Analytics, Content Strategy, Social Media, Bachelor's degree in Marketing or Communications",
            "url": "https://example.com/job/5"
        }
    ])

def fetch_live_jsearch_jobs(query):
    """
    Fetches real-time open jobs using JSearch API on RapidAPI.
    To use live data, set your RapidAPI key in Streamlit secrets or environment variables.
    """
    rapid_api_key = st.secrets.get("RAPID_API_KEY") if hasattr(st, "secrets") else os.getenv("RAPID_API_KEY")
    
    if not rapid_api_key:
        return [] # Fall back gracefully if no key is provided yet
        
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": query, "page": "1", "num_pages": "1"}
    headers = {
        "X-RapidAPI-Key": rapid_api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=5)
        if response.status_code == 200:
            data = response.json().get("data", [])
            formatted_jobs = []
            for item in data:
                formatted_jobs.append({
                    "title": item.get("job_title", "Position Available"),
                    "company": item.get("employer_name", "Confidential"),
                    "location": f"{item.get('job_city', '')}, {item.get('job_country', 'Remote')}",
                    "requirements": item.get("job_description", "Check application link for qualifications.")[:250] + "...",
                    "url": item.get("job_apply_link", item.get("job_google_link", "https://google.com"))
                })
            return formatted_jobs
    except Exception:
        pass
    return []

# --- 3. UI LAYOUT & USER INPUTS ---
st.title("🚀 AI Career Advisor & Open Job Matcher")
st.markdown("Input your technical skills and educational certificates below to generate customized advice and instantly match with open positions.")

with st.form("profile_form"):
    col1, col2 = st.columns(2)
    with col1:
        user_skills = st.text_area(
            "Your Skills (comma-separated)",
            placeholder="e.g., Python, SQL, Project Management, React, Data Visualization",
            help="List the tools, programming languages, or core competencies you have."
        )
    with col2:
        user_education = st.text_area(
            "Educational Background & Certifications",
            placeholder="e.g., B.Sc. Computer Science, AWS Certified Cloud Practitioner, Google Data Analytics",
            help="Include degrees, diplomas, bootcamps, or industry-recognized certs."
        )
    
    submitted = st.form_submit_button("Analyze Profile & Find Open Jobs")

# --- 4. PROCESSING & RECOMMENDATION ENGINE ---
if submitted:
    if not user_skills.strip() and not user_education.strip():
        st.warning("⚠️ Please provide at least some skills or educational history.")
    else:
        with st.spinner("Analyzing candidate profile against open job market matrices..."):
            user_profile = f"{user_skills} {user_education}"
            
            # Combine mock + any live API data retrieved based on user inputs
            base_jobs_df = load_fallback_jobs()
            
            # Try grabbing live listings relevant to the primary skill typed
            primary_query = user_skills.split(",")[0] if user_skills else "Developer"
            live_jobs = fetch_live_jsearch_jobs(primary_query)
            
            if live_jobs:
                live_df = pd.DataFrame(live_jobs)
                jobs_df = pd.concat([base_jobs_df, live_df], ignore_index=True)
            else:
                jobs_df = base_jobs_df
            
            # Vectorize text using TF-IDF and compute semantic Cosine Similarity
            job_corpus = (jobs_df["title"] + " " + jobs_df["requirements"]).tolist()
            corpus = job_corpus + [user_profile]
            
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            user_vector = tfidf_matrix[-1]
            job_vectors = tfidf_matrix[:-1]
            similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()
            
            jobs_df["Match Score"] = similarity_scores * 100
            ranked_jobs = jobs_df.sort_values(by="Match Score", ascending=False).reset_index(drop=True)
            
            # --- 5. RENDER RESULTS ---
            st.success("Analysis complete! Here is your custom roadmap and matching jobs.")
            
            # AI Career Advice Generator block
            st.subheader("💡 Tailored AI Career Advice")
            
            advice_container = st.container()
            with advice_container:
                skills_lower = user_skills.lower()
                edu_lower = user_education.lower()
                
                if "python" in skills_lower or "sql" in skills_lower:
                    st.info("🔹 **Technical Track Detected:** Your programming/data background is heavily sought after. Consider building an end-to-end portfolio project (like a dashboard or automated API backend) to back up your credentials.")
                elif "marketing" in skills_lower or "seo" in skills_lower:
                    st.info("🔹 **Growth & Marketing Track:** Focus heavily on metrics and verifiable campaign ROI in your resume. Certifications in Google Analytics or Hubspot will give you an instant edge.")
                else:
                    st.info("🔹 **General Optimization Tip:** Make sure to explicitly align keywords from target job specs directly into your certifications and skills blocks to clear Automated Applicant Tracking Systems (ATS).")
                
                if "bachelor" not in edu_lower and "master" not in edu_lower and "degree" not in edu_lower:
                    st.warning("⚠️ **Tip on Education:** Since formal degrees weren't explicitly highlighted, prioritize building a strong GitHub repo or project portfolio to validate your practical skills to employers.")

            # Open Job Listings Section
            st.subheader(f"🔥 Open Positions Ranked For You")
            
            for idx, row in ranked_jobs.iterrows():
                score = row["Match Score"]
                
                with st.container():
                    col_left, col_right = st.columns([4, 1])
                    with col_left:
                        st.markdown(f"### {row['title']} @ **{row['company']}**")
                        st.markdown(f"📍 **Location:** {row['location']}")
                        st.markdown(f"📋 **Requirements/Details:** {row['requirements']}")
                    with col_right:
                        st.metric(label="Match Fit", value=f"{score:.1f}%")
                        if row["url"] and row["url"].startswith("http"):
                            st.markdown(f"[Apply Direct]({row['url'])")
                    st.markdown("---")